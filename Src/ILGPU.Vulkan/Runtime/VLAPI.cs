// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLAPI.cs
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Backends.Vulkan;
using System;
using Silk.NET.Vulkan;

namespace ILGPU.Runtime.Vulkan;

internal static class VLAPI
{
    // Builds buffer descriptor info for a given view
    internal static unsafe DescriptorBufferInfo GetBufferInfo<T>(ArrayView<T> view)
        where T : unmanaged
    {
        var buf = (VLMemoryBuffer)((IArrayView)view).Buffer;
        var offset = ((IContiguousArrayView)view).IndexInBytes;
        return new DescriptorBufferInfo
        {
            Buffer = buf.BufferHandle,
            Offset = (ulong)offset,
            Range = (ulong)view.LengthInBytes,
        };
    }

    // Note: Additional overloads for ArrayView1D/2D/3D can be added later if needed.

    // Records and submits a compute dispatch with optional push-constants
    private static unsafe void BindAndDispatch(
        Vk vk,
        Silk.NET.Vulkan.Device dev,
        VLStream stream,
        VLKernel kernel,
        DescriptorSet dset,
        RuntimeKernelConfig config,
        int* pcValues,
        uint pcCount)
    {
        var cmd = stream.CommandBuffer;
        vk.ResetCommandBuffer(cmd, 0);
        CommandBufferBeginInfo bi = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        vk.BeginCommandBuffer(cmd, in bi).ThrowOnError();
        vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, kernel.Pipeline);
        DescriptorSet* pSet = stackalloc DescriptorSet[1];
        pSet[0] = dset;
        vk.CmdBindDescriptorSets(
            cmd,
            PipelineBindPoint.Compute,
            kernel.PipelineLayout,
            0,
            1,
            pSet,
            0,
            null);
        if (pcCount > 0)
        {
            vk.CmdPushConstants(
                cmd,
                kernel.PipelineLayout,
                ShaderStageFlags.ComputeBit,
                0,
                (uint)(pcCount * sizeof(int)),
                pcValues);
        }
        vk.CmdDispatch(cmd, (uint)config.GridDim.X, (uint)config.GridDim.Y, (uint)config.GridDim.Z);
        vk.EndCommandBuffer(cmd).ThrowOnError();

        SubmitInfo si = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmd,
        };
        vk.QueueSubmit(((VLAccelerator)stream.Accelerator).ComputeQueue, 1, in si, stream.Fence).ThrowOnError();
        stream.MarkSubmitted();
    }

    // Allocates a transient descriptor pool and set, writes buffer infos
    private static unsafe DescriptorSet AllocateAndWrite(
        Vk vk,
        Silk.NET.Vulkan.Device dev,
        DescriptorSetLayout layout,
        in DescriptorBufferInfo* infos,
        uint count,
        out DescriptorPool pool)
    {
        DescriptorPoolSize poolSize = new()
        {
            Type = DescriptorType.StorageBuffer,
            DescriptorCount = count,
        };
        DescriptorPoolCreateInfo dpci = new()
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            MaxSets = 1,
            PoolSizeCount = 1,
            PPoolSizes = &poolSize,
        };
        vk.CreateDescriptorPool(dev, in dpci, null, out pool).ThrowOnError();

        DescriptorSetLayout layoutLocal = layout;
        DescriptorSetAllocateInfo dsai = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = pool,
            DescriptorSetCount = 1,
            PSetLayouts = &layoutLocal,
        };
        vk.AllocateDescriptorSets(dev, in dsai, out var dset).ThrowOnError();

        var writes = stackalloc WriteDescriptorSet[(int)count];
        for (uint i = 0; i < count; i++)
        {
            writes[i] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = dset,
                DstBinding = i,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo = &infos[i],
            };
        }
        vk.UpdateDescriptorSets(dev, count, writes, 0, null);
        return dset;
    }

    internal static unsafe void Launch(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        IArrayView[] views,
        int[] scalars)
    {
        var acc = (VLAccelerator)stream.Accelerator;
        var vk = acc.Vk;
        var dev = acc.LogicalDevice;
        kernel.EnsurePipeline();

        // Build descriptors and push constants
        var count = views?.Length ?? 0;
        var infos = new DescriptorBufferInfo[count];
        var pc = new int[count + (scalars?.Length ?? 0)];
        for (int i = 0; i < count; i++)
        {
            var v = views[i];
            var mem = v.Buffer;
            var buf = (VLMemoryBuffer)mem;
            long indexInBytes = 0;
            if (v is IContiguousArrayView cav)
            {
                indexInBytes = cav.IndexInBytes;
            }
            else
            {
                // Try to resolve BaseView to fetch a contiguous origin
                var t = v.GetType();
                var baseViewProp = t.GetProperty("BaseView");
                if (baseViewProp != null)
                {
                    var baseViewObj = baseViewProp.GetValue(v);
                    if (baseViewObj is IContiguousArrayView baseCav)
                        indexInBytes = baseCav.IndexInBytes;
                }
            }
            long lengthInBytes = mem.LengthInBytes - indexInBytes;
            infos[i] = new DescriptorBufferInfo
            {
                Buffer = buf.BufferHandle,
                Offset = (ulong)indexInBytes,
                Range = (ulong)lengthInBytes,
            };
            pc[i] = (int)v.Length;
        }
        if (scalars != null && scalars.Length > 0)
            Array.Copy(scalars, 0, pc, count, scalars.Length);

        // Trace (optional) the push-constant layout and values
        VLTrace.Log($"LaunchKernelGeneric: views={count} scalars={(scalars?.Length ?? 0)} pc=[" + string.Join(",", pc) + "]");

        fixed (DescriptorBufferInfo* pInfos = infos)
        fixed (int* pPc = pc)
        {
            var dset = AllocateAndWrite(vk, dev, kernel.DescriptorSetLayout, pInfos, (uint)infos.Length, out var pool);
            try
            {
                BindAndDispatch(vk, dev, stream, kernel, dset, config, pPc, (uint)pc.Length);
            }
            finally
            {
                vk.DestroyDescriptorPool(dev, pool, null);
            }
        }
    }

    // Unified span-based dispatch entry (internal helper)
    internal static unsafe void LaunchKernel(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ReadOnlySpan<DescriptorBufferInfo> buffers,
        ReadOnlySpan<int> pushConstants)
    {
        var acc = (VLAccelerator)stream.Accelerator;
        var vk = acc.Vk;
        var dev = acc.LogicalDevice;
        kernel.EnsurePipeline();

        DescriptorBufferInfo* pInfos = stackalloc DescriptorBufferInfo[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) pInfos[i] = buffers[i];
        var dset = AllocateAndWrite(vk, dev, kernel.DescriptorSetLayout, pInfos, (uint)buffers.Length, out var pool);
        try
        {
            if (pushConstants.IsEmpty)
            {
                BindAndDispatch(vk, dev, stream, kernel, dset, config, null, 0u);
            }
            else
            {
                fixed (int* pPc = pushConstants)
                {
                    BindAndDispatch(vk, dev, stream, kernel, dset, config, pPc, (uint)pushConstants.Length);
                }
            }
        }
        finally
        {
            vk.DestroyDescriptorPool(dev, pool, null);
        }
    }

    // Specialized launchers removed; use LaunchKernelGeneric.

    // 2 views (heterogeneous element types)
    internal static unsafe void LaunchKernelWithStreamBinding<T0, T1>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T0> a,
        ArrayView<T1> b)
        where T0 : unmanaged
        where T1 : unmanaged
    {
        var acc = (VLAccelerator)stream.Accelerator;
        var vk = acc.Vk;
        var dev = acc.LogicalDevice;
        kernel.EnsurePipeline();

        var info = stackalloc DescriptorBufferInfo[2];
        info[0] = GetBufferInfo(a);
        info[1] = GetBufferInfo(b);
        var dset = AllocateAndWrite(vk, dev, kernel.DescriptorSetLayout, info, 2, out var pool);
        try
        {
            var vals = stackalloc int[2];
            vals[0] = (int)a.Length;
            vals[1] = (int)b.Length;
            BindAndDispatch(vk, dev, stream, kernel, dset, config, vals, 2u);
        }
        finally
        {
            vk.DestroyDescriptorPool(dev, pool, null);
        }
    }

    // All specialized launchers removed; use LaunchKernelGeneric.
}
