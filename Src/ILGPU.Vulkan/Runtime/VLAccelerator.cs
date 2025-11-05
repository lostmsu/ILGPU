// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLAccelerator.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.IL;
using ILGPU;
using ILGPU.Resources;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using VkDevice = Silk.NET.Vulkan.Device;
using System;
using System.Reflection;
using ILGPU.Util;
using System.Diagnostics;
using System.Reflection.Emit;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Vulkan accelerator scaffold.
/// </summary>
[Obsolete("Not implemented")]
public sealed unsafe class VLAccelerator : KernelAccelerator<ILGPU.Backends.Vulkan.VLCompiledKernel, VLKernel>
{
    private static bool IsAnyArrayViewType(Type t)
    {
        if (!t.IsGenericType) return false;
        var g = t.GetGenericTypeDefinition();
        return g == typeof(ArrayView<>) ||
               g == typeof(ArrayView1D<,>) ||
               g == typeof(ArrayView2D<,>) ||
               g == typeof(ArrayView3D<,>);
    }
    private readonly Vk _vk;
    private Instance _instance;
    private PhysicalDevice _physicalDevice;
    private VkDevice _device;
    private uint _queueFamilyIndex;
    private Queue _queue;
    private CommandPool _commandPool;
    // Validation layer enablement (no debug messenger wiring here to avoid extra deps)

    public VLAccelerator(Context context, VulkanDevice device)
        : base(context, device)
    {
        // Initialize Vulkan runtime objects for compute.
        _vk = Vk.GetApi();
        CreateInstance();
        PickPhysicalDeviceByName(device.Name);
        CreateDeviceAndQueue();
        CreateCommandPool();

        // Assign a native pointer for disposal assertions.
        NativePtr = new IntPtr(_device.Handle);

        // Create default stream.
        DefaultStream = CreateStreamInternal();

        // Initialize backend
        Init(new ILGPU.Backends.Vulkan.VLBackend(
            Context,
            (VulkanCapabilityContext)Device.Capabilities));
    }

    public override TExtension CreateExtension<TExtension, TExtensionProvider>(
        TExtensionProvider provider)
    {
        throw new NotImplementedException();
    }

    protected override AcceleratorStream CreateStreamInternal() => new VLStream(this);

    protected override void SynchronizeInternal()
    {
        if (_device.Handle != 0)
            _vk.DeviceWaitIdle(_device);
    }

    protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
    {
        return new VLMemoryBuffer(this, length, elementSize);
    }

    protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
        Kernel kernel,
        int groupSize,
        int dynamicSharedMemorySizeInBytes) => 1;

    protected override int EstimateGroupSizeInternal(
        Kernel kernel,
        Func<int, int> computeSharedMemorySize,
        int maxGroupSize,
        out int minGridSize)
    {
        minGridSize = 1;
        return Math.Min(maxGroupSize, Device.MaxNumThreadsPerGroup);
    }

    protected override int EstimateGroupSizeInternal(
        Kernel kernel,
        int dynamicSharedMemorySizeInBytes,
        int maxGroupSize,
        out int minGridSize)
    {
        minGridSize = 1;
        return Math.Min(maxGroupSize, Device.MaxNumThreadsPerGroup);
    }

    protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(
        IntPtr pinned,
        long numElements)
    {
        throw new NotImplementedException();
    }

    protected override void DisposeAccelerator_SyncRoot(bool disposing)
    {
        if (_device.Handle != 0)
        {
            _vk.DeviceWaitIdle(_device);
            if (_commandPool.Handle != 0)
                _vk.DestroyCommandPool(_device, _commandPool, null);
            _vk.DestroyDevice(_device, null);
            _device = default;
        }
        if (_instance.Handle != 0)
        {
            _vk.DestroyInstance(_instance, null);
            _instance = default;
        }
    }

    protected override void OnBind() { }
    protected override void OnUnbind() { }

    protected override MethodInfo GenerateKernelLauncherMethod(
        ILGPU.Backends.Vulkan.VLCompiledKernel kernel,
        int customGroupSize)
    {
        var entryPoint = kernel.EntryPoint;
        AdjustAndVerifyKernelGroupSize(ref customGroupSize, entryPoint);

        // For now, only implicitly grouped kernels are supported.
        if (entryPoint.HasByRefParameters)
            throw new NotSupportedException(ErrorMessages.NotSupportedByRefKernelParameters);

        using var scopedLock = entryPoint.CreateLauncherMethod(
            Context.RuntimeSystem,
            out var launcher);
        var emitter = new ILEmitter(launcher.ILGenerator);

        // Load kernel instance
        var kernelLocal = emitter.DeclareLocal(typeof(VLKernel));
        KernelLauncherBuilder.EmitLoadKernelArgument<VLKernel, ILEmitter>(
            Kernel.KernelInstanceParamIdx,
            emitter);
        emitter.Emit(LocalOperation.Store, kernelLocal);

        // Load stream
        KernelLauncherBuilder.EmitLoadAcceleratorStream<VLStream, ILEmitter>(
            Kernel.KernelStreamParamIdx,
            emitter);
        var streamLocal = emitter.DeclareLocal(typeof(VLStream));
        emitter.Emit(LocalOperation.Store, streamLocal);

        // Load RuntimeKernelConfig
        KernelLauncherBuilder.EmitLoadRuntimeKernelConfig(
            entryPoint,
            emitter,
            Kernel.KernelParamDimensionIdx,
            MaxGridSize,
            MaxGroupSize,
            customGroupSize);
        var configLocal = emitter.DeclareLocal(typeof(RuntimeKernelConfig));
        emitter.Emit(LocalOperation.Store, configLocal);

        // Build a plan with the Vulkan argument mapper
        var mapper = new ILGPU.Backends.Vulkan.VLArgumentMapper(Context);
        var plan = mapper.BuildPlan(entryPoint);

        var paramTypes = entryPoint.Parameters;

        var viewsLocal = emitter.DeclareLocal(typeof(IArrayView[]));
        var scalarsLocal = emitter.DeclareLocal(typeof(int[]));
        var cursorLocal = emitter.DeclareLocal(typeof(int));

        emitter.LoadIntegerConstant(plan.ViewCount);
        emitter.Emit(OpCodes.Newarr, typeof(IArrayView));
        emitter.Emit(LocalOperation.Store, viewsLocal);
        for (int i = 0; i < plan.ViewCount; i++)
        {
            int vIdx = plan.ViewParamIndices[i];
            emitter.Emit(LocalOperation.Load, viewsLocal);
            emitter.LoadIntegerConstant(i);
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + vIdx);
            // box ArrayView<T> and cast to IArrayView
            emitter.Emit(OpCodes.Box, paramTypes[vIdx]);
            emitter.Emit(OpCodes.Castclass, typeof(IArrayView));
            emitter.Emit(OpCodes.Stelem_Ref);
        }

        // Allocate scalars array sized to compiled PushConstantCount minus view lengths
        // kernelLocal is already stored; load its PushConstantCount property
        emitter.Emit(LocalOperation.Load, kernelLocal);
        var pcCountProp = typeof(VLKernel).GetProperty("PushConstantCount", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public).ThrowIfNull();
        var getter = pcCountProp.GetGetMethod(nonPublic: true).ThrowIfNull();
        emitter.EmitCall(getter);
        // subtract viewCount
        emitter.LoadIntegerConstant(plan.ViewCount);
        emitter.Emit(OpCodes.Sub);
        emitter.Emit(OpCodes.Newarr, typeof(int));
        emitter.Emit(LocalOperation.Store, scalarsLocal);

        // Initialize cursor to scalarIntCount (relative to scalars array)
        emitter.LoadIntegerConstant(plan.ScalarIntCount);
        emitter.Emit(LocalOperation.Store, cursorLocal);

        for (int i = 0; i < plan.ScalarIntCount; i++)
        {
            int sIdx = plan.ScalarIntParamIndices[i];
            emitter.Emit(LocalOperation.Load, scalarsLocal);
            emitter.LoadIntegerConstant(i);
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + sIdx);
            emitter.Emit(OpCodes.Stelem_I4);
        }

        // Append non-view parameters as raw 32-bit words using VLAPI.PackIntoInt32WordsObject
        var packObjMethod = typeof(VLAPI).GetMethod(nameof(VLAPI.PackIntoInt32WordsObject)).ThrowIfNull();
        for (int i = 0; i < paramTypes.Count; i++)
        {
            var pt = paramTypes[i];
            if (IsAnyArrayViewType(pt) || pt == typeof(int))
                continue;
            // value (boxed)
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + i);
            emitter.Emit(OpCodes.Box, pt);
            // array
            emitter.Emit(LocalOperation.Load, scalarsLocal);
            // start index from cursor
            emitter.Emit(LocalOperation.Load, cursorLocal);
            // call and get words written
            emitter.EmitCall(packObjMethod);
            // increment cursor by returned word count
            emitter.Emit(LocalOperation.Load, cursorLocal);
            emitter.Emit(OpCodes.Add);
            emitter.Emit(LocalOperation.Store, cursorLocal);
        }

        // Load preserved stream, kernel, config and call generic API
        emitter.Emit(LocalOperation.Load, streamLocal);
        emitter.Emit(LocalOperation.Load, kernelLocal);
        emitter.Emit(LocalOperation.Load, configLocal);
        emitter.Emit(LocalOperation.Load, viewsLocal);
        emitter.Emit(LocalOperation.Load, scalarsLocal);
        var genericApi = typeof(VLAPI).GetMethod(nameof(VLAPI.Launch), BindingFlags.NonPublic | BindingFlags.Static).ThrowIfNull();
        emitter.EmitCall(genericApi);

        emitter.Emit(OpCodes.Ret);
        emitter.Finish();
        var mi = launcher.Finish();
        ILGPU.Backends.Vulkan.ILDebug.ValidateAndDump(mi);
        return mi;
    }

    protected override VLKernel CreateKernel(ILGPU.Backends.Vulkan.VLCompiledKernel compiledKernel)
    {
        return new VLKernel(this, compiledKernel, null);
    }

    protected override VLKernel CreateKernel(
        ILGPU.Backends.Vulkan.VLCompiledKernel compiledKernel,
        MethodInfo launcher)
    {
        return new VLKernel(this, compiledKernel, launcher);
    }

    protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) => false;
    protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) { }
    protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) { }

    #region Internal accessors

    internal Vk Vk => _vk;
    internal Instance Instance => _instance;
    internal PhysicalDevice PhysicalDevice => _physicalDevice;
    internal VkDevice LogicalDevice => _device;
    internal Queue ComputeQueue => _queue;
    internal uint ComputeQueueFamilyIndex => _queueFamilyIndex;
    internal CommandPool CommandPool => _commandPool;

    #endregion

    #region Initialization helpers

    private void CreateInstance()
    {
        ApplicationInfo appInfo = new()
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)SilkMarshal.StringToPtr("ILGPU.Vulkan"),
            PEngineName = (byte*)SilkMarshal.StringToPtr("ILGPU"),
            ApiVersion = Vk.Version12,
        };
        try
        {
            // Discover available layers and extensions
            bool wantValidation = true; // enable by default during backend bring-up
            // Query layers
            uint layerCount = 0;
            _vk.EnumerateInstanceLayerProperties(ref layerCount, null);
            var layers = new LayerProperties[layerCount > 0 ? layerCount : 0];
            if (layerCount > 0)
            {
                fixed (LayerProperties* p = layers)
                    _vk.EnumerateInstanceLayerProperties(ref layerCount, p);
            }
            bool hasValidation = false;
            foreach (ref var lp in layers.AsSpan())
            {
                var name = PtrToString(ref lp.LayerName[0]);
                if (name == "VK_LAYER_KHRONOS_validation") { hasValidation = true; break; }
            }
            // Query extensions
            uint extCount = 0;
            _vk.EnumerateInstanceExtensionProperties((byte*)null, ref extCount, null);
            var exts = new ExtensionProperties[extCount > 0 ? extCount : 0];
            if (extCount > 0)
            {
                fixed (ExtensionProperties* p = exts)
                    _vk.EnumerateInstanceExtensionProperties((byte*)null, ref extCount, p);
            }
            bool hasDebugUtils = false;
            foreach (ref var ep in exts.AsSpan())
            {
                var name = PtrToString(ref ep.ExtensionName[0]);
                if (name == "VK_EXT_debug_utils") { hasDebugUtils = true; break; }
            }

            // Prepare name arrays
            var enabledLayers = stackalloc byte*[1];
            uint enabledLayerCount = 0;
            if (wantValidation && hasValidation)
            {
                enabledLayers[enabledLayerCount++] = (byte*)SilkMarshal.StringToPtr("VK_LAYER_KHRONOS_validation");
            }
            var enabledExts = stackalloc byte*[1];
            uint enabledExtCount = 0;
            if (hasDebugUtils)
                enabledExts[enabledExtCount++] = (byte*)SilkMarshal.StringToPtr("VK_EXT_debug_utils");

            InstanceCreateInfo ci = new()
            {
                SType = StructureType.InstanceCreateInfo,
                PApplicationInfo = &appInfo,
                EnabledLayerCount = enabledLayerCount,
                PpEnabledLayerNames = enabledLayerCount > 0 ? enabledLayers : null,
                EnabledExtensionCount = enabledExtCount,
                PpEnabledExtensionNames = enabledExtCount > 0 ? enabledExts : null,
            };
            _vk.CreateInstance(in ci, null, out _instance).ThrowOnError();

            // Free allocated layer/extension strings
            for (int i = 0; i < enabledLayerCount; i++) SilkMarshal.Free((nint)enabledLayers[i]);
            for (int i = 0; i < enabledExtCount; i++) SilkMarshal.Free((nint)enabledExts[i]);

            // Note: debug messenger not created to avoid additional extension bindings here.
        }
        finally
        {
            SilkMarshal.Free((nint)appInfo.PApplicationName);
            SilkMarshal.Free((nint)appInfo.PEngineName);
        }
    }

    private static unsafe uint DebugCallback(
        DebugUtilsMessageSeverityFlagsEXT severity,
        DebugUtilsMessageTypeFlagsEXT types,
        DebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        try
        {
            string msg = SilkMarshal.PtrToString((nint)data->PMessage) ?? string.Empty;
            Debug.WriteLine($"[VK] {severity} {types}: {msg}");
        }
        catch { /* best-effort logging */ }
        return Vk.False;
    }

    private void PickPhysicalDeviceByName(string desiredName)
    {
        uint count = 0;
        _vk.EnumeratePhysicalDevices(_instance, ref count, null);
        if (count == 0)
            throw new InvalidOperationException("No Vulkan physical devices");
        var pds = new PhysicalDevice[count];
        unsafe
        {
            fixed (PhysicalDevice* p = pds)
                _vk.EnumeratePhysicalDevices(_instance, ref count, p);
        }

        // Pick by name if found, else first with compute queue support.
        PhysicalDevice? selected = null;
        foreach (var pd in pds)
        {
            _vk.GetPhysicalDeviceProperties(pd, out var props);
            string name = PtrToString(ref props.DeviceName[0]);
            if (string.Equals(name, desiredName, StringComparison.Ordinal))
            {
                selected = pd; break;
            }
        }
        if (selected is null)
            selected = pds[0];
        _physicalDevice = selected.Value;
    }

        #region Launcher helpers
        // No specialized helpers needed; single generic path is used.
        #endregion

    private void CreateDeviceAndQueue()
    {
        // Find a compute-capable queue family
        uint familyCount = 0;
        _vk.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref familyCount, null);
        if (familyCount == 0)
            throw new InvalidOperationException("No queue families");
        var families = new QueueFamilyProperties[familyCount];
        unsafe
        {
            fixed (QueueFamilyProperties* p = families)
                _vk.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref familyCount, p);
        }
        uint computeIndex = uint.MaxValue;
        for (uint i = 0; i < familyCount; i++)
        {
            if ((families[i].QueueFlags & QueueFlags.QueueComputeBit) != 0)
            {
                computeIndex = i; break;
            }
        }
        if (computeIndex == uint.MaxValue)
            throw new NotSupportedException("No compute queue family found");

        float priority = 1.0f;
        DeviceQueueCreateInfo dq = new()
        {
            SType = StructureType.DeviceQueueCreateInfo,
            QueueFamilyIndex = computeIndex,
            QueueCount = 1,
            PQueuePriorities = &priority,
        };

        // Enable shaderInt64 feature if supported (required by some tests)
        PhysicalDeviceFeatures feats;
        _vk.GetPhysicalDeviceFeatures(_physicalDevice, out feats);
        var enableFeatures = new PhysicalDeviceFeatures();
        if (feats.ShaderInt64)
            enableFeatures.ShaderInt64 = Vk.True;

        DeviceCreateInfo dci = new()
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &dq,
            PEnabledFeatures = &enableFeatures,
        };

        _vk.CreateDevice(_physicalDevice, in dci, null, out _device).ThrowOnError();
        _vk.GetDeviceQueue(_device, computeIndex, 0, out _queue);
        _queueFamilyIndex = computeIndex;
    }

    private void CreateCommandPool()
    {
        CommandPoolCreateInfo cpi = new()
        {
            SType = StructureType.CommandPoolCreateInfo,
            QueueFamilyIndex = _queueFamilyIndex,
            Flags = CommandPoolCreateFlags.CommandPoolCreateResetCommandBufferBit,
        };
        _vk.CreateCommandPool(_device, in cpi, null, out _commandPool).ThrowOnError();
    }

    private static unsafe string PtrToString(ref byte first)
    {
        fixed (byte* p = &first)
        {
            return SilkMarshal.PtrToString((nint)p) ?? string.Empty;
        }
    }

    #endregion
}
