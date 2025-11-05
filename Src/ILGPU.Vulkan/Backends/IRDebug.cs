// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: IRDebug.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using System;
using System.IO;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Simple IR dump helper gated by environment variables.
///  - ILGPU_VULKAN_IR_DUMP=1 enables dumping to %TEMP%/ilgpu-vl-ir-*.txt
/// </summary>
internal static class IRDebug
{
    private static bool IsEnabled() => string.Equals(
        Environment.GetEnvironmentVariable("ILGPU_VULKAN_IR_DUMP"),
        "1",
        StringComparison.Ordinal);

    public static void DumpKernelIR(EntryPoint entryPoint, in Backend.BackendContext backendContext, string stage)
    {
        if (!IsEnabled())
            return;

        var m = backendContext.KernelMethod;
        var path = Path.Combine(
            Path.GetTempPath(),
            $"ilgpu-vl-ir-{DateTime.UtcNow:yyyyMMdd_HHmmss_fffffff}-{Sanitize(entryPoint.Name)}-{stage}.txt");
        using var sw = new StreamWriter(path);
        sw.WriteLine($"// EntryPoint: {entryPoint.Name}");
        sw.WriteLine($"// Stage: {stage}");
        sw.WriteLine();
        m.Dump(sw);
        VLTrace.Log($"IR dump written: {path}");
    }

    private static string Sanitize(string name)
    {
        foreach (var ch in Path.GetInvalidFileNameChars())
            name = name.Replace(ch, '_');
        return name;
    }
}

