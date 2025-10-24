// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLTrace.cs
// ---------------------------------------------------------------------------------------

using System;
using System.IO;

namespace ILGPU.Backends.Vulkan;

internal static class VLTrace
{
    private static bool? s_enabled;
    private static string? s_path;
    private static bool Enabled
    {
        get
        {
            if (s_enabled.HasValue)
                return s_enabled.Value;
            var env = Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE");
            s_enabled = string.Equals(env, "1", StringComparison.Ordinal);
            return s_enabled.Value;
        }
    }
    private static string PathForLog => s_path ??=
        (Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE_FILE") is string p && !string.IsNullOrWhiteSpace(p)
            ? p
            : System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ilgpu-vl-trace.txt"));

    public static void Log(string message)
    {
        if (!Enabled) return;
        var line = $"[VL][{DateTime.UtcNow:HH:mm:ss.fffffff}] {message}";
        Console.WriteLine(line);
        File.AppendAllText(PathForLog, line + Environment.NewLine);
    }
}
