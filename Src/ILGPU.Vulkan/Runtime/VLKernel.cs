// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLKernel.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.Vulkan;
using Silk.NET.Vulkan;
using System;
using System.Reflection;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Minimal Vulkan runtime kernel placeholder.
/// </summary>
[Obsolete("Not implemented")]
public sealed class VLKernel : Kernel
{
    private readonly VLAccelerator _acc;
    private readonly VLCompiledKernel _compiled;
    private ShaderModule _module;
    private DescriptorSetLayout _dsl;
    private PipelineLayout _layout;
    private Pipeline _pipeline;

    public VLKernel(
        VLAccelerator accelerator,
        VLCompiledKernel compiledKernel,
        MethodInfo? launcher)
        : base(accelerator, compiledKernel, launcher)
    {
        _acc = accelerator;
        _compiled = compiledKernel;
        CreateShaderModule();
    }

    protected override unsafe void DisposeAcceleratorObject(bool disposing)
    {
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;
        if (_pipeline.Handle != 0) { vk.DestroyPipeline(dev, _pipeline, null); _pipeline = default; }
        if (_layout.Handle != 0) { vk.DestroyPipelineLayout(dev, _layout, null); _layout = default; }
        if (_dsl.Handle != 0) { vk.DestroyDescriptorSetLayout(dev, _dsl, null); _dsl = default; }
        if (_module.Handle != 0) { vk.DestroyShaderModule(dev, _module, null); _module = default; }
    }

    private unsafe void CreateShaderModule()
    {
        if (_compiled.SpirvWords is null || _compiled.SpirvWords.Length == 0)
            return;
        fixed (uint* pCode = _compiled.SpirvWords)
        {
            ShaderModuleCreateInfo smci = new()
            {
                SType = StructureType.ShaderModuleCreateInfo,
                CodeSize = (nuint)(_compiled.SpirvWords.Length * 4),
                PCode = pCode,
            };
            _acc.Vk.CreateShaderModule(_acc.LogicalDevice, in smci, null, out _module)
                .ThrowOnError();
        }
    }

    internal unsafe void EnsurePipeline(int bindingCount)
    {
        if (_pipeline.Handle != 0)
            return;

        // Descriptor set layout with bindingCount storage buffers
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

        var bindings = stackalloc DescriptorSetLayoutBinding[Math.Max(1, bindingCount)];
        for (uint i = 0; i < (uint)bindingCount; i++)
        {
            bindings[i] = new DescriptorSetLayoutBinding
            {
                Binding = i,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.ShaderStageComputeBit,
            };
        }
        DescriptorSetLayoutCreateInfo dslci = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = (uint)bindingCount,
            PBindings = bindings,
        };
        vk.CreateDescriptorSetLayout(dev, in dslci, null, out _dsl).ThrowOnError();

        // Pipeline layout
        var setLayouts = stackalloc DescriptorSetLayout[1];
        setLayouts[0] = _dsl;
        PipelineLayoutCreateInfo plci = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 1,
            PSetLayouts = setLayouts,
        };
        vk.CreatePipelineLayout(dev, in plci, null, out _layout).ThrowOnError();

        // Compute pipeline
        var entryName = "main";
        PipelineShaderStageCreateInfo ssc = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.ShaderStageComputeBit,
            Module = _module,
            PName = (byte*)Silk.NET.Core.Native.SilkMarshal.StringToPtr(entryName),
        };
        try
        {
            ComputePipelineCreateInfo cpci = new()
            {
                SType = StructureType.ComputePipelineCreateInfo,
                Stage = ssc,
                Layout = _layout,
            };
            vk.CreateComputePipelines(dev, default, 1, in cpci, null, out _pipeline).ThrowOnError();
        }
        finally
        {
            Silk.NET.Core.Native.SilkMarshal.Free((nint)ssc.PName);
        }
    }

    internal DescriptorSetLayout DescriptorSetLayout => _dsl;
    internal PipelineLayout PipelineLayout => _layout;
    internal Pipeline Pipeline => _pipeline;
}
