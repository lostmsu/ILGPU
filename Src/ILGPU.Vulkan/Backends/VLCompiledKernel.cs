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
/// Represents a compiled Vulkan kernel holding SPIR-V bytes and
/// resource-binding metadata for descriptors.
/// </summary>
public sealed class VLCompiledKernel : CompiledKernel
{
    public readonly struct BindingInfo
    {
        public BindingInfo(int paramIndex, uint binding)
        {
            ParamIndex = paramIndex;
            Binding = binding;
        }

        public int ParamIndex { get; }
        public uint Binding { get; }
    }

    public VLCompiledKernel(
        Context context,
        EntryPoint entryPoint,
        KernelInfo? info,
        uint[] spirv,
        BindingInfo[] bindings,
        int pushConstantCount)
        : base(context, entryPoint, info)
    {
        SpirvWords = spirv;
        Bindings = bindings;
        PushConstantCount = pushConstantCount;
    }

    /// <summary>
    /// The SPIR-V words for this kernel.
    /// </summary>
    public uint[] SpirvWords { get; }

    /// <summary>
    /// Descriptor-set bindings generated at compile-time.
    /// </summary>
    public BindingInfo[] Bindings { get; }

    /// <summary>
    /// Number of 32-bit integers used in the push-constant block.
    /// This typically contains all view lengths followed by scalar ints.
    /// </summary>
    public int PushConstantCount { get; }
}
