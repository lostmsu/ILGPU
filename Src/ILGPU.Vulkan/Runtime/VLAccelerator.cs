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
using System.Reflection.Emit;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Vulkan accelerator scaffold.
/// </summary>
[Obsolete("Not implemented")]
public sealed unsafe class VLAccelerator : KernelAccelerator<ILGPU.Backends.Vulkan.VLCompiledKernel, VLKernel>
{
    #region Static

    private static readonly MethodInfo Launch1OpenGeneric =
        typeof(VLAccelerator).GetMethod(
            nameof(Launch1),
            BindingFlags.NonPublic | BindingFlags.Static)
        .ThrowIfNull();

    private static readonly MethodInfo Launch3OpenGeneric =
        typeof(VLAccelerator).GetMethod(
            nameof(Launch3),
            BindingFlags.NonPublic | BindingFlags.Static)
        .ThrowIfNull();

    #endregion
    private readonly Vk _vk;
    private Instance _instance;
    private PhysicalDevice _physicalDevice;
    private VkDevice _device;
    private uint _queueFamilyIndex;
    private Queue _queue;
    private CommandPool _commandPool;

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

        // Load kernel
        emitter.Emit(LocalOperation.Load, kernelLocal);

        // Load RuntimeKernelConfig
        KernelLauncherBuilder.EmitLoadRuntimeKernelConfig(
            entryPoint,
            emitter,
            Kernel.KernelParamDimensionIdx,
            MaxGridSize,
            MaxGroupSize,
            customGroupSize);

        // Load ArrayView<T> args based on parameter count
        var paramTypes = entryPoint.Parameters;
        int numArgs = paramTypes.Count;
        if (numArgs < 1)
            throw new NotSupportedException("Kernels without view parameters not yet supported");

        // Currently support 1 and 3 view-parameter kernels
        if (numArgs == 1)
        {
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + 0);
            var viewType = paramTypes[0];
            var elemType = viewType.GetGenericArguments()[0];
            var apiMethod = Launch1OpenGeneric.MakeGenericMethod(elemType);
            emitter.EmitCall(apiMethod);
        }
        else if (numArgs >= 3)
        {
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + 0);
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + 1);
            emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + 2);
            var viewType = paramTypes[0];
            var elemType = viewType.GetGenericArguments()[0];
            var apiMethod = Launch3OpenGeneric.MakeGenericMethod(elemType);
            emitter.EmitCall(apiMethod);
        }
        else
        {
            throw new NotImplementedException("Only 1 or 3 view parameters supported");
        }

        emitter.Emit(OpCodes.Ret);
        emitter.Finish();
        return launcher.Finish();
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
            InstanceCreateInfo ci = new()
            {
                SType = StructureType.InstanceCreateInfo,
                PApplicationInfo = &appInfo,
            };
            _vk.CreateInstance(in ci, null, out _instance).ThrowOnError();
        }
        finally
        {
            SilkMarshal.Free((nint)appInfo.PApplicationName);
            SilkMarshal.Free((nint)appInfo.PEngineName);
        }
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
    private static void Launch1<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a)
        where T : unmanaged
        => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a);

    private static void Launch3<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> c)
        where T : unmanaged
        => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a, b, c);
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

        DeviceCreateInfo dci = new()
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &dq,
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
