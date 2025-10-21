// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLMemoryBuffer.cs
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.CompilerServices;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;
using ILGPU.Resources;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Minimal Vulkan memory buffer scaffold.
/// </summary>
[Obsolete("Not implemented")]
public sealed unsafe class VLMemoryBuffer : MemoryBuffer
{
    private readonly VLAccelerator _acc;
    private VkBuffer _buffer;
    private DeviceMemory _memory;
    private long _sizeInBytes;
    private ulong _memoryOffset;

    public VLMemoryBuffer(Accelerator accelerator, long length, int elementSize)
        : base(accelerator, length, elementSize)
    {
        _acc = (VLAccelerator)accelerator;
        _sizeInBytes = LengthInBytes;
        if (_sizeInBytes == 0)
            return;

        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        BufferCreateInfo bi = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = (ulong)_sizeInBytes,
            Usage = BufferUsageFlags.BufferUsageStorageBufferBit |
                    BufferUsageFlags.BufferUsageTransferSrcBit |
                    BufferUsageFlags.BufferUsageTransferDstBit,
            SharingMode = SharingMode.Exclusive,
        };
        vk.CreateBuffer(dev, in bi, null, out _buffer).ThrowOnError();

        vk.GetBufferMemoryRequirements(dev, _buffer, out var req);
        var memTypeIndex = FindMemoryTypeIndex(
            req.MemoryTypeBits,
            MemoryPropertyFlags.MemoryPropertyHostVisibleBit |
            MemoryPropertyFlags.MemoryPropertyHostCoherentBit);
        if (memTypeIndex == uint.MaxValue)
        {
            throw new NotSupportedException(
                "Required HOST_VISIBLE | HOST_COHERENT memory not available");
        }
        MemoryAllocateInfo mai = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = req.Size,
            MemoryTypeIndex = memTypeIndex,
        };
        _acc.Vk.AllocateMemory(dev, in mai, null, out _memory).ThrowOnError();
        _memoryOffset = 0;
        vk.BindBufferMemory(dev, _buffer, _memory, _memoryOffset).ThrowOnError();
    }

    protected override void DisposeAcceleratorObject(bool disposing)
    {
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;
        if (_buffer.Handle != 0)
        {
            vk.DestroyBuffer(dev, _buffer, null);
            _buffer = default;
        }
        if (_memory.Handle != 0)
        {
            vk.FreeMemory(dev, _memory, null);
            _memory = default;
        }
    }

    internal VkBuffer BufferHandle => _buffer;

    protected internal override void MemSet(
        AcceleratorStream stream,
        byte value,
        in ArrayView<byte> targetView)
    {
        if (_sizeInBytes == 0 || targetView.LengthInBytes == 0)
            return;
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        var offset = ((IContiguousArrayView)targetView).IndexInBytes;
        void* mapped;
        vk.MapMemory(dev, _memory, (ulong)offset, (ulong)targetView.LengthInBytes, 0, &mapped)
            .ThrowOnError();
        Unsafe.InitBlockUnaligned(mapped, value, (uint)targetView.LengthInBytes);
        vk.UnmapMemory(dev, _memory);
    }

    protected internal override void CopyTo(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        // this -> target
        if (_sizeInBytes == 0 || targetView.LengthInBytes == 0)
            return;
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        var srcOffset = ((IContiguousArrayView)sourceView).IndexInBytes;
        // If target is CPU, copy via map
        var targetBuffer = ((IArrayView)targetView).Buffer;
        if (targetBuffer is ILGPU.Runtime.CPU.CPUMemoryBuffer cpuTarget)
        {
            void* mapped;
            vk.MapMemory(dev, _memory, (ulong)srcOffset, (ulong)targetView.LengthInBytes, 0, &mapped)
                .ThrowOnError();
            var dstPtr = (byte*)cpuTarget.NativePtr + ((IContiguousArrayView)targetView).IndexInBytes;
            Unsafe.CopyBlockUnaligned(dstPtr, mapped, (uint)targetView.LengthInBytes);
            vk.UnmapMemory(dev, _memory);
            return;
        }

        // If target is also Vulkan and host-visible, map both and copy
        if (targetBuffer is VLMemoryBuffer vlTarget)
        {
            var dstOffset = ((IContiguousArrayView)targetView).IndexInBytes;
            void* srcMapped;
            void* dstMapped;
            vk.MapMemory(dev, _memory, (ulong)srcOffset, (ulong)targetView.LengthInBytes, 0, &srcMapped)
                .ThrowOnError();
            vk.MapMemory(dev, vlTarget._memory, (ulong)dstOffset, (ulong)targetView.LengthInBytes, 0, &dstMapped)
                .ThrowOnError();
            Unsafe.CopyBlockUnaligned(dstMapped, srcMapped, (uint)targetView.LengthInBytes);
            vk.UnmapMemory(dev, vlTarget._memory);
            vk.UnmapMemory(dev, _memory);
            return;
        }

        throw new NotSupportedException(RuntimeErrorMessages.NotSupportedTargetAccelerator);
    }

    protected internal override void CopyFrom(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        // source -> this
        if (_sizeInBytes == 0 || sourceView.LengthInBytes == 0)
            return;
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        var dstOffset = ((IContiguousArrayView)targetView).IndexInBytes;
        var sourceBuffer = ((IArrayView)sourceView).Buffer;
        if (sourceBuffer is ILGPU.Runtime.CPU.CPUMemoryBuffer cpuSource)
        {
            void* mapped;
            vk.MapMemory(dev, _memory, (ulong)dstOffset, (ulong)sourceView.LengthInBytes, 0, &mapped)
                .ThrowOnError();
            var srcPtr = (byte*)cpuSource.NativePtr + ((IContiguousArrayView)sourceView).IndexInBytes;
            Unsafe.CopyBlockUnaligned(mapped, srcPtr, (uint)sourceView.LengthInBytes);
            vk.UnmapMemory(dev, _memory);
            return;
        }

        if (sourceBuffer is VLMemoryBuffer vlSource)
        {
            var srcOffset = ((IContiguousArrayView)sourceView).IndexInBytes;
            void* srcMapped;
            void* dstMapped;
            vk.MapMemory(dev, vlSource._memory, (ulong)srcOffset, (ulong)sourceView.LengthInBytes, 0, &srcMapped)
                .ThrowOnError();
            vk.MapMemory(dev, _memory, (ulong)dstOffset, (ulong)sourceView.LengthInBytes, 0, &dstMapped)
                .ThrowOnError();
            Unsafe.CopyBlockUnaligned(dstMapped, srcMapped, (uint)sourceView.LengthInBytes);
            vk.UnmapMemory(dev, _memory);
            vk.UnmapMemory(dev, vlSource._memory);
            return;
        }

        throw new NotSupportedException(RuntimeErrorMessages.NotSupportedTargetAccelerator);
    }

    private uint FindMemoryTypeIndex(uint typeBits, MemoryPropertyFlags props)
    {
        var vk = _acc.Vk;
        vk.GetPhysicalDeviceMemoryProperties(_acc.PhysicalDevice, out var memProps);
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
        {
            if (((typeBits >> (int)i) & 1) == 1 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & props) == props)
                return i;
        }
        return uint.MaxValue;
    }
}
