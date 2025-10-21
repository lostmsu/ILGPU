// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: SpirvDebug.cs
// ---------------------------------------------------------------------------------------

using System;
using System.Buffers.Binary;
using System.IO;
using System.Diagnostics;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Helper for dumping SPIR-V modules for offline validation.
/// Controlled by env var ILGPU_VULKAN_DUMP ("1" to enable).
/// </summary>
internal static class SpirvDebug
{
    private static bool Enabled =>
        string.Equals(
            Environment.GetEnvironmentVariable("ILGPU_VULKAN_DUMP"),
            "1",
            StringComparison.Ordinal);

    /// <summary>
    /// Dumps the given SPIR-V words as a .spv file into the temp folder and
    /// records the actual path into a small .path file in the current directory.
    /// </summary>
    /// <param name="baseName">Base filename without extension.</param>
    /// <param name="words">SPIR-V words.</param>
    /// <returns>The full path of the written .spv file, or null if disabled.</returns>
    public static string? Dump(string baseName, ReadOnlySpan<uint> words)
    {
        if (!Enabled)
            return null;

        var bytes = new byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4, 4), words[i]);

        var temp = Path.GetTempPath();
        var name = $"{baseName}-{DateTime.UtcNow:yyyyMMdd_HHmmss_fffffff}.spv";
        var fullPath = Path.Combine(temp, name);

        // Let exceptions bubble to caller if something goes wrong.
        File.WriteAllBytes(fullPath, bytes);

        // Log the path in debug builds for discovery
        Debug.WriteLine($"[ILGPU.Vulkan] SPIR-V dump written: {fullPath}");

        return fullPath;
    }

}
