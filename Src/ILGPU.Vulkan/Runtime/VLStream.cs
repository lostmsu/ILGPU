// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLStream.cs
// ---------------------------------------------------------------------------------------

using System;
using ILGPU.Resources;
using Silk.NET.Vulkan;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Minimal Vulkan stream scaffold.
/// </summary>
[Obsolete("Not implemented")]
public sealed class VLStream : AcceleratorStream
{
    private readonly VLAccelerator _acc;
    private CommandBuffer _cmd;
    private Fence _fence;
    private bool _submitted;

    public VLStream(Accelerator accelerator) : base(accelerator)
    {
        _acc = (VLAccelerator)accelerator;
        AllocateCommandResources();
    }

    public override void Synchronize()
    {
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;
        if (_fence.Handle != 0 && _submitted)
        {
            vk.WaitForFences(dev, 1, in _fence, Vk.True, ulong.MaxValue);
            vk.ResetFences(dev, 1, in _fence);
            _submitted = false;
        }
    }

    protected override unsafe void DisposeAcceleratorObject(bool disposing)
    {
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;
        if (_cmd.Handle != 0)
        {
            vk.FreeCommandBuffers(dev, _acc.CommandPool, 1, in _cmd);
            _cmd = default;
        }
        if (_fence.Handle != 0)
        {
            vk.DestroyFence(dev, _fence, null);
            _fence = default;
        }
    }

    protected override ProfilingMarker AddProfilingMarkerInternal() =>
        throw new NotSupportedException(RuntimeErrorMessages.NotSupportedProfilingMarker);

    internal CommandBuffer CommandBuffer => _cmd;
    internal Fence Fence => _fence;
    internal void MarkSubmitted() => _submitted = true;

    private unsafe void AllocateCommandResources()
    {
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        CommandBufferAllocateInfo ai = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = _acc.CommandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1,
        };
        vk.AllocateCommandBuffers(dev, in ai, out _cmd);

        FenceCreateInfo fi = new()
        {
            SType = StructureType.FenceCreateInfo,
            Flags = 0,
        };
        vk.CreateFence(dev, in fi, null, out _fence);
    }
}
