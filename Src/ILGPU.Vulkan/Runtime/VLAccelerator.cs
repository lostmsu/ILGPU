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
using System.Linq;

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

    private static readonly MethodInfo Launch2OpenGeneric =
        typeof(VLAccelerator).GetMethod(
            nameof(Launch2),
            BindingFlags.NonPublic | BindingFlags.Static)
        .ThrowIfNull();

    private static readonly MethodInfo Launch3ScalarOpenGeneric =
        typeof(VLAccelerator).GetMethod(
            nameof(Launch3Scalar),
            BindingFlags.NonPublic | BindingFlags.Static)
        .ThrowIfNull();

    #endregion
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

        // Build a plan with the Vulkan argument mapper
        var mapper = new ILGPU.Backends.Vulkan.VLArgumentMapper(Context);
        var plan = mapper.BuildPlan(entryPoint);

        var paramTypes = entryPoint.Parameters;
        // If all view element types are identical, we can use generic helpers; otherwise
        // fall back to mixed-type non-generic launchers using IArrayView.
        bool AllSameElemType()
        {
            if (plan.ViewCount <= 1) return true;
            var first = paramTypes[plan.ViewParamIndices[0]].GetGenericArguments()[0];
            for (int i = 1; i < plan.ViewCount; i++)
            {
                var t = paramTypes[plan.ViewParamIndices[i]].GetGenericArguments()[0];
                if (!ReferenceEquals(t, first)) return false;
            }
            return true;
        }

        if (!AllSameElemType())
        {
            if (plan.ScalarIntCount == 0 && plan.ViewCount == 3)
            {
                // Generic heterogeneous 3-view path: VLAPI.LaunchKernelWithStreamBinding<T0,T1,T2>
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                int v2 = plan.ViewParamIndices[2];
                var t0 = paramTypes[v0].GetGenericArguments()[0];
                var t1 = paramTypes[v1].GetGenericArguments()[0];
                var t2 = paramTypes[v2].GetGenericArguments()[0];
                var api = typeof(VLAPI).GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                    .First(m => m.Name == nameof(VLAPI.LaunchKernelWithStreamBinding)
                        && m.IsGenericMethodDefinition
                        && m.GetGenericArguments().Length == 3
                        && m.GetParameters().Length == 6);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v2);
                emitter.EmitCall(api.MakeGenericMethod(t0, t1, t2));
            }
            else if (plan.ScalarIntCount == 0 && plan.ViewCount == 2)
            {
                // Generic heterogeneous 2-view path: VLAPI.LaunchKernelWithStreamBinding<T0,T1>
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                var t0 = paramTypes[v0].GetGenericArguments()[0];
                var t1 = paramTypes[v1].GetGenericArguments()[0];
                var api = typeof(VLAPI).GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                    .First(m => m.Name == nameof(VLAPI.LaunchKernelWithStreamBinding)
                        && m.IsGenericMethodDefinition
                        && m.GetGenericArguments().Length == 2
                        && m.GetParameters().Length == 5);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.EmitCall(api.MakeGenericMethod(t0, t1));
            }
            else
            {
                throw new NotImplementedException($"Unsupported heterogeneous layout: views={plan.ViewCount}, scalars={plan.ScalarIntCount}");
            }
        }
        else
        {
            // Original generic paths
            if (plan.ViewCount == 1 && plan.ScalarIntCount == 0)
            {
                int p0 = plan.ViewParamIndices[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + p0);
                var elemType = paramTypes[p0].GetGenericArguments()[0];
                var apiMethod = Launch1OpenGeneric.MakeGenericMethod(elemType);
                emitter.EmitCall(apiMethod);
            }
            else if (plan.ViewCount == 2 && plan.ScalarIntCount == 1)
            {
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                int s0 = plan.ScalarIntParamIndices[0];
                var elemType = paramTypes[v0].GetGenericArguments()[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + s0);
                var api = typeof(VLAccelerator).GetMethod(nameof(Launch2Scalar), BindingFlags.NonPublic | BindingFlags.Static).ThrowIfNull();
                emitter.EmitCall(api.MakeGenericMethod(elemType));
            }
            else if (plan.ViewCount == 2 && plan.ScalarIntCount == 0)
            {
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                var elemType = paramTypes[v0].GetGenericArguments()[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.EmitCall(Launch2OpenGeneric.MakeGenericMethod(elemType));
            }
            else if (plan.ViewCount == 3 && plan.ScalarIntCount == 0)
            {
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                int v2 = plan.ViewParamIndices[2];
                var elemType = paramTypes[v0].GetGenericArguments()[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v2);
                emitter.EmitCall(Launch3OpenGeneric.MakeGenericMethod(elemType));
            }
            else if (plan.ViewCount == 3 && plan.ScalarIntCount == 1)
            {
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                int v2 = plan.ViewParamIndices[2];
                int s0 = plan.ScalarIntParamIndices[0];
                var elemType = paramTypes[v0].GetGenericArguments()[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v2);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + s0);
                emitter.EmitCall(Launch3ScalarOpenGeneric.MakeGenericMethod(elemType));
            }
            else if (plan.ViewCount == 3 && plan.ScalarIntCount == 2)
            {
                int v0 = plan.ViewParamIndices[0];
                int v1 = plan.ViewParamIndices[1];
                int v2 = plan.ViewParamIndices[2];
                int s0 = plan.ScalarIntParamIndices[0];
                int s1 = plan.ScalarIntParamIndices[1];
                var elemType = paramTypes[v0].GetGenericArguments()[0];
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v1);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + v2);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + s0);
                emitter.Emit(ArgumentOperation.Load, Kernel.KernelParameterOffset + s1);
                var api = typeof(VLAccelerator).GetMethod(nameof(Launch3Scalar2), BindingFlags.NonPublic | BindingFlags.Static).ThrowIfNull();
                emitter.EmitCall(api.MakeGenericMethod(elemType));
            }
            else
            {
                throw new NotImplementedException($"Unsupported parameter layout: views={plan.ViewCount}, scalars={plan.ScalarIntCount}");
            }
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

    private static void Launch2<T>(
          VLStream stream,
          VLKernel kernel,
          RuntimeKernelConfig config,
          ArrayView<T> a,
          ArrayView<T> b)
          where T : unmanaged
          => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a, b);

    private static void Launch2Scalar<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a,
        ArrayView<T> b,
        int s0)
        where T : unmanaged
        => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a, b, s0);

    private static void Launch3Scalar<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> c,
        int s0)
        where T : unmanaged
        => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a, b, c, s0);

    private static void Launch3Scalar2<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> c,
        int s0,
        int s1)
        where T : unmanaged
        => VLAPI.LaunchKernelWithStreamBinding<T>(stream, kernel, config, a, b, c, s0, s1);
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
