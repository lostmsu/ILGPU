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

        internal unsafe void EnsurePipeline()
        {
            if (_pipeline.Handle != 0)
                return;

        // Descriptor set layout with bindingCount storage buffers
        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;

            var bindingCount = _compiled.Bindings?.Length ?? 0;
            var bindings = stackalloc DescriptorSetLayoutBinding[Math.Max(1, bindingCount)];
            for (uint i = 0; i < (uint)bindingCount; i++)
            {
                bindings[i] = new DescriptorSetLayoutBinding
                {
                    Binding = i,
                    DescriptorType = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                    StageFlags = ShaderStageFlags.ComputeBit,
                };
            }
            DescriptorSetLayoutCreateInfo dslci = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)bindingCount,
                PBindings = bindings,
            };
        vk.CreateDescriptorSetLayout(dev, in dslci, null, out _dsl).ThrowOnError();

        // Pipeline layout (descriptor set + push constants for view lengths and scalars)
        var setLayouts = stackalloc DescriptorSetLayout[1];
        setLayouts[0] = _dsl;
        PushConstantRange pcr = default;
        var totalPcCount = _compiled.PushConstantCount;
        var usePush = totalPcCount > 0;
        if (usePush)
        {
            pcr = new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset = 0,
                Size = (uint)(totalPcCount * sizeof(int)),
            };
        }
        PipelineLayoutCreateInfo plci = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 1,
            PSetLayouts = setLayouts,
            PushConstantRangeCount = usePush ? 1u : 0u,
            PPushConstantRanges = usePush ? &pcr : null,
        };
        vk.CreatePipelineLayout(dev, in plci, null, out _layout).ThrowOnError();

        // Compute pipeline is created in EnsurePipeline(RuntimeKernelConfig) using
        // specialization constants for LocalSizeId.
    }

    internal DescriptorSetLayout DescriptorSetLayout => _dsl;
    internal PipelineLayout PipelineLayout => _layout;
    internal Pipeline Pipeline => _pipeline;
    internal int BindingCount => _compiled.Bindings?.Length ?? 0;
    internal int PushConstantCount => _compiled.PushConstantCount;

    internal unsafe void EnsurePipeline(RuntimeKernelConfig config)
    {
        if (_pipeline.Handle != 0)
            return;

        // Create base pipeline state (module, layouts) if needed
        EnsurePipeline();

        var vk = _acc.Vk;
        var dev = _acc.LogicalDevice;
        var entryName = "main";
        // Specialization constants for LocalSizeId with SpecId 0/1/2
        Silk.NET.Vulkan.SpecializationMapEntry* entries = stackalloc Silk.NET.Vulkan.SpecializationMapEntry[3];
        entries[0] = new Silk.NET.Vulkan.SpecializationMapEntry { ConstantID = 0, Offset = 0u, Size = (uint)sizeof(int) };
        entries[1] = new Silk.NET.Vulkan.SpecializationMapEntry { ConstantID = 1, Offset = (uint)sizeof(int), Size = (uint)sizeof(int) };
        entries[2] = new Silk.NET.Vulkan.SpecializationMapEntry { ConstantID = 2, Offset = (uint)(2 * sizeof(int)), Size = (uint)sizeof(int) };
        int* data = stackalloc int[3];
        data[0] = (int)config.GroupDim.X;
        data[1] = (int)config.GroupDim.Y;
        data[2] = (int)config.GroupDim.Z;
        Silk.NET.Vulkan.SpecializationInfo spec = new Silk.NET.Vulkan.SpecializationInfo
        {
            MapEntryCount = 3,
            PMapEntries = entries,
            DataSize = (nuint)(3 * sizeof(int)),
            PData = data,
        };

        var ssc = new PipelineShaderStageCreateInfo
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.ComputeBit,
            Module = _module,
            PName = (byte*)Silk.NET.Core.Native.SilkMarshal.StringToPtr(entryName),
            PSpecializationInfo = &spec,
        };
        try
        {
            var cpci = new ComputePipelineCreateInfo
            {
                SType = StructureType.ComputePipelineCreateInfo,
                Stage = ssc,
                Layout = _layout,
            };
            vk.CreateComputePipelines(dev, default, 1, in cpci, null, out _pipeline).ThrowOnError();
        }
        catch (AccessViolationException ave)
        {
            Console.Error.WriteLine($"[Vulkan] AccessViolation during CreateComputePipelines: {ave.Message}");
            throw;
        }
        finally
        {
            Silk.NET.Core.Native.SilkMarshal.Free((nint)ssc.PName);
        }
    }
}
