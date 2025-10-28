// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLCodeGenerator.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using System;

namespace ILGPU.Backends.Vulkan;

internal static class VLCodeGenerator
{
    public readonly struct Result
    {
        public Result(uint[] words, VLCompiledKernel.BindingInfo[] bindings, int pushConstantCount)
        {
            Words = words; Bindings = bindings; PushConstantCount = pushConstantCount;
        }
        public uint[] Words { get; }
        public VLCompiledKernel.BindingInfo[] Bindings { get; }
        public int PushConstantCount { get; }
    }

    public static Result Generate(EntryPoint entryPoint, in Backend.BackendContext backendContext)
    {
        var module = new SpirvModuleBuilder(entryPoint);
        var translator = new MinimalVlTranslator(module, entryPoint);
        translator.TranslateKernel(backendContext);
        var words = module.ToUIntArray();
        var spvPath = SpirvDebug.Dump("vl-last", words);
        if (spvPath != null && string.Equals(Environment.GetEnvironmentVariable("ILGPU_VULKAN_MAP"), "1", StringComparison.Ordinal))
        {
            var map = module.BuildDebugMap();
            SpirvDebug.WriteMapFor(spvPath, map);
        }
        return new Result(words, translator.GetBindings(), translator.GetPushConstantCount());
    }
}
