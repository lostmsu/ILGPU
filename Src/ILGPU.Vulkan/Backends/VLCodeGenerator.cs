// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLCodeGenerator.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;

namespace ILGPU.Backends.Vulkan;

internal static class VLCodeGenerator
{
    public static uint[] Generate(EntryPoint entryPoint, in Backend.BackendContext backendContext)
    {
        var module = new SpirvModuleBuilder(entryPoint);
        var translator = new MinimalVlTranslator(module, entryPoint);
        translator.TranslateKernel(backendContext);
        return module.ToUIntArray();
    }
}

