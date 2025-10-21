// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLAPI.cs
// ---------------------------------------------------------------------------------------

using Silk.NET.Vulkan;

namespace ILGPU.Runtime.Vulkan;

internal static class VLAPI
{
    internal static unsafe void LaunchKernelWithStreamBinding<T>(
        VLStream stream,
        VLKernel kernel,
        RuntimeKernelConfig config,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> c)
        where T : unmanaged
    {
        var acc = (VLAccelerator)stream.Accelerator;
        var vk = acc.Vk;
        var dev = acc.LogicalDevice;

        // Ensure pipeline created with 3 bindings
        kernel.EnsurePipeline(3);

        // Allocate descriptor pool + set
        DescriptorPoolSize poolSize = new()
        {
            Type = DescriptorType.StorageBuffer,
            DescriptorCount = 3,
        };
        var poolSizes = new DescriptorPoolSize[] { poolSize };
        fixed (DescriptorPoolSize* pSizes = poolSizes)
        {
            DescriptorPoolCreateInfo dpci = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = 1,
                PoolSizeCount = 1,
                PPoolSizes = pSizes,
            };
            vk.CreateDescriptorPool(dev, in dpci, null, out var dpool).ThrowOnError();
            try
            {
                // Allocate set
                var layouts = new DescriptorSetLayout[] { kernel.DescriptorSetLayout };
                fixed (DescriptorSetLayout* pLayouts = layouts)
                {
                    DescriptorSetAllocateInfo dsai = new()
                    {
                        SType = StructureType.DescriptorSetAllocateInfo,
                        DescriptorPool = dpool,
                        DescriptorSetCount = 1,
                        PSetLayouts = pLayouts,
                    };
                    vk.AllocateDescriptorSets(dev, in dsai, out var dset).ThrowOnError();

                    // Build buffer infos
                    var bufA = GetBufferInfo(a);
                    var bufB = GetBufferInfo(b);
                    var bufC = GetBufferInfo(c);
                    var infos = stackalloc DescriptorBufferInfo[3];
                    infos[0] = bufA; infos[1] = bufB; infos[2] = bufC;

                    var writes = stackalloc WriteDescriptorSet[3];
                    for (uint i = 0; i < 3; i++)
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
                    vk.UpdateDescriptorSets(dev, 3, writes, 0, null);

                    // Record + submit
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
                    vk.CmdDispatch(cmd, (uint)config.GridDim.X, (uint)config.GridDim.Y, (uint)config.GridDim.Z);
                    vk.EndCommandBuffer(cmd).ThrowOnError();

                    SubmitInfo si = new()
                    {
                        SType = StructureType.SubmitInfo,
                        CommandBufferCount = 1,
                        PCommandBuffers = &cmd,
                    };
                    // Queue submit with stream fence
                    vk.QueueSubmit(acc.ComputeQueue, 1, in si, stream.Fence).ThrowOnError();
                    stream.MarkSubmitted();
                }
            }
            finally
            {
                vk.DestroyDescriptorPool(dev, dpool, null);
            }
        }

        static DescriptorBufferInfo GetBufferInfo(ArrayView<T> view)
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
    }
}
