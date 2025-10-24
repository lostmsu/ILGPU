// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: SpirvValidator.cs
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.IO;

namespace ILGPU.Backends.Vulkan;

internal static class SpirvValidator
{
    private static string? FindTool(string exe)
    {
        var fromEnv = Environment.GetEnvironmentVariable("ILGPU_VULKAN_SPIRV_TOOLS");
        if (!string.IsNullOrWhiteSpace(fromEnv))
        {
            var p = Path.Combine(fromEnv!, exe);
            if (File.Exists(p)) return p;
        }
        var sdkRoot = Path.Combine("C:\\VulkanSDK");
        if (Directory.Exists(sdkRoot))
        {
            foreach (var dir in Directory.GetDirectories(sdkRoot))
            {
                var bin = Path.Combine(dir, "Bin");
                var cand = Path.Combine(bin, exe);
                if (File.Exists(cand)) return cand;
            }
        }
        return null;
    }

    public static bool TryValidate(string spvPath, out string output)
    {
        output = string.Empty;
        var valExe = FindTool("spirv-val.exe");
        if (valExe is null) { output = "spirv-val not found"; return false; }
        return Run(valExe, spvPath, out output);
    }

    public static bool TryDisassemble(string spvPath, out string output)
    {
        output = string.Empty;
        var disExe = FindTool("spirv-dis.exe");
        if (disExe is null) { output = "spirv-dis not found"; return false; }
        return Run(disExe, spvPath, out output);
    }

    private static bool Run(string exe, string arg, out string output)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = '"' + arg + '"',
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        using var p = Process.Start(psi)!;
        var stdout = p.StandardOutput.ReadToEnd();
        var stderr = p.StandardError.ReadToEnd();
        p.WaitForExit();
        output = string.IsNullOrWhiteSpace(stderr) ? stdout : stderr;
        return p.ExitCode == 0;
    }
}

