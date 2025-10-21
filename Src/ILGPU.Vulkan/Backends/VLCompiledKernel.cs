// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLCompiledKernel.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Represents a compiled Vulkan kernel holding SPIR-V bytes.
/// </summary>
public sealed class VLCompiledKernel : CompiledKernel
{
    public VLCompiledKernel(
        Context context,
        EntryPoint entryPoint,
        KernelInfo? info,
        uint[] spirv)
        : base(context, entryPoint, info)
    {
        SpirvWords = spirv;
    }

    /// <summary>
    /// The SPIR-V words for this kernel.
    /// </summary>
    public uint[] SpirvWords { get; }
}
