// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLBackend.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.IR.Transformations;
using ILGPU.Backends.Vulkan.Transformations;
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
        // Initialize kernel transformers to match CL/PTX design.
        InitializeKernelTransformers(builder =>
        {
            var transformerBuilder = Transformer.CreateBuilder(
                TransformerConfiguration.Empty);
            transformerBuilder.AddBackendOptimizations<CodePlacement.GroupOperands>(
                new VLAcceleratorSpecializer(
                    PointerType,
                    context.Properties.EnableIOOperations),
                context.Properties.InliningMode,
                context.Properties.OptimizationLevel);
            // Validate unsupported IR patterns post-lowering
            transformerBuilder.Add(new Transformations.VLValidateUnsupportedTransformation());
            builder.Add(transformerBuilder.ToTransformer());
        });
    }

    protected override EntryPoint CreateEntryPoint(
        in EntryPointDescription entry,
        in BackendContext backendContext,
        in KernelSpecialization specialization) =>
        new SeparateViewEntryPoint(
            entry,
            backendContext.SharedMemorySpecification,
            specialization,
            Context.TypeContext,
            2);

    protected override CompiledKernel Compile(
        EntryPoint entryPoint,
        in BackendContext backendContext,
        in KernelSpecialization specialization)
    {
        ArgumentNullException.ThrowIfNull(entryPoint);
        if (!entryPoint.IsImplicitlyGrouped)
            throw new NotImplementedException("Only implicitly grouped kernels implemented");

            var result = VLCodeGenerator.Generate(entryPoint, backendContext);
            return new VLCompiledKernel(Context, entryPoint, null, result.Words, result.Bindings, result.PushConstantCount);
        }
}
