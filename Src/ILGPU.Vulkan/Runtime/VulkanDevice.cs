// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VulkanDevice.cs
// ---------------------------------------------------------------------------------------

using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// A minimal Vulkan device wrapper for ILGPU device enumeration.
/// </summary>
[DeviceType(AcceleratorType.Vulkan)]
public sealed unsafe class VulkanDevice : Device
{
    #region Static

    /// <summary>
    /// Detects Vulkan devices and registers them.
    /// </summary>
    internal static void GetDevices(
        Predicate<VulkanDevice> predicate,
        DeviceRegistry registry)
    {
        if (registry is null)
            throw new ArgumentNullException(nameof(registry));
        if (predicate is null)
            throw new ArgumentNullException(nameof(predicate));

        var vk = Vk.GetApi();

        // Create minimal instance
        Instance instance = default;
        ApplicationInfo appInfo = new()
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)SilkMarshal.StringToPtr("ILGPU.Vulkan"),
            // Leave versions at default
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
            vk.CreateInstance(in ci, null, out instance).ThrowOnError();

            uint count = 0;
            vk.EnumeratePhysicalDevices(instance, ref count, null);
            if (count == 0)
                return;

            var pds = new PhysicalDevice[count];
            fixed (PhysicalDevice* p = pds)
                vk.EnumeratePhysicalDevices(instance, ref count, p);

            foreach (var pd in pds)
            {
                var device = CreateFromPhysicalDevice(vk, pd);
                registry.Register(device, predicate);
            }
        }
        finally
        {
            if (instance.Handle != 0)
                vk.DestroyInstance(instance, null);
            SilkMarshal.Free((nint)appInfo.PApplicationName);
            SilkMarshal.Free((nint)appInfo.PEngineName);
        }
    }

    private static VulkanDevice CreateFromPhysicalDevice(Vk vk, PhysicalDevice pd)
    {
        vk.GetPhysicalDeviceProperties(pd, out var props);
        vk.GetPhysicalDeviceMemoryProperties(pd, out var memProps);
        vk.GetPhysicalDeviceFeatures(pd, out var _);

        string name = PtrToString(ref props.DeviceName[0]);

        // Memory size: sum of device-local heaps
        long memBytes = 0;
        for (int i = 0; i < memProps.MemoryHeapCount; i++)
        {
            var heap = memProps.MemoryHeaps[i];
            if ((heap.Flags & MemoryHeapFlags.DeviceLocalBit) != 0)
                memBytes += (long)heap.Size;
        }

        // Limits
        var limits = props.Limits;
        static int ClampUint(uint v) => v >= int.MaxValue ? int.MaxValue : (int)v;
        int maxInvocations = ClampUint(limits.MaxComputeWorkGroupInvocations);
        var maxGroup = new Index3D(
            ClampUint(limits.MaxComputeWorkGroupSize[0]),
            ClampUint(limits.MaxComputeWorkGroupSize[1]),
            ClampUint(limits.MaxComputeWorkGroupSize[2]));
        var maxGrid = new Index3D(
            ClampUint(limits.MaxComputeWorkGroupCount[0]),
            ClampUint(limits.MaxComputeWorkGroupCount[1]),
            ClampUint(limits.MaxComputeWorkGroupCount[2]));

        var dev = new VulkanDevice(name)
        {
            MemorySize = memBytes,
            MaxGroupSize = maxGroup,
            MaxGridSize = maxGrid,
            MaxNumThreadsPerGroup = maxInvocations,
            MaxSharedMemoryPerGroup = (int)limits.MaxComputeSharedMemorySize,
            MaxConstantMemory = 0,
            WarpSize = 64,
            NumMultiprocessors = 1,
            MaxNumThreadsPerMultiprocessor = maxInvocations,
        };
        return dev;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static string PtrToString(ref byte first)
    {
        fixed (byte* p = &first)
        {
            return SilkMarshal.PtrToString((nint)p) ?? string.Empty;
        }
    }

    #endregion

    #region Instance

    internal VulkanDevice(string name)
    {
        Name = name;
        // Capabilities will be refined later; for now use defaults
        Capabilities = new VulkanCapabilityContext();
    }

    #endregion

    #region Properties
    // Uses base Device properties with protected setters.
    #endregion

    #region Methods

    public override Accelerator CreateAccelerator(Context context) =>
        new VLAccelerator(context, this);

    #endregion
}

internal static class VkResultExtensions
{
    [DebuggerHidden]
    [StackTraceHidden]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowOnError(this Result res)
    {
        if (res != Result.Success)
            throw new InvalidOperationException($"Vulkan error: {res}");
    }
}
