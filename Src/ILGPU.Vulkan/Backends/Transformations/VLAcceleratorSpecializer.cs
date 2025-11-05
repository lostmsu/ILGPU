// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLAcceleratorSpecializer.cs
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Transformations;
using ILGPU.IR.Types;
using ILGPU.Runtime;

namespace ILGPU.Backends.Vulkan.Transformations;

/// <summary>
/// Vulkan accelerator specializer. Mirrors CL/PTX design: pushes accelerator
/// constants (AcceleratorType, WarpSize, native pointer width) into IR during
/// backend transformer phase.
/// </summary>
internal sealed class VLAcceleratorSpecializer : AcceleratorSpecializer
{
    public VLAcceleratorSpecializer(
        PrimitiveType pointerType,
        bool enableIOOperations)
        : base(
              AcceleratorType.Vulkan,
              null,
              pointerType,
              enableAssertions: false,
              enableIOOperations)
    { }
}

