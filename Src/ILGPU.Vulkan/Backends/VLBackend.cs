// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLBackend.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using ILGPU.Runtime.Vulkan;
using System;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Minimal Vulkan backend scaffold. Emits SPIR-V for a restricted subset.
/// </summary>
[Obsolete("Not implemented")]
public sealed class VLBackend : Backend
{
    public VLBackend(
        Context context,
        VulkanCapabilityContext capabilities)
        : base(
              context,
              capabilities,
              BackendType.OpenCL, // placeholder enum until BackendType adds Vulkan
              new VLArgumentMapper(context))
    {
        // Kernel transformers will be added as the backend matures.
    }

    protected override CompiledKernel Compile(
        EntryPoint entryPoint,
        in BackendContext backendContext,
        in KernelSpecialization specialization)
    {
        ArgumentNullException.ThrowIfNull(entryPoint);
        if (!entryPoint.IsImplicitlyGrouped)
            throw new NotImplementedException("Only implicitly grouped kernels implemented");

        var words = VLCodeGenerator.Generate(entryPoint, backendContext);
        SpirvDebug.Dump("vl-last", words);
        return new VLCompiledKernel(Context, entryPoint, null, words);
    }
}
