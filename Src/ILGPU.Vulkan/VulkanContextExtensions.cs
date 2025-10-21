// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VulkanContextExtensions.cs
// ---------------------------------------------------------------------------------------

using System;

namespace ILGPU.Runtime.Vulkan;

/// <summary>
/// Vulkan-specific context extensions.
/// </summary>
public static class VulkanContextExtensions
{
    /// <summary>
    /// Enables all compatible Vulkan devices.
    /// </summary>
    /// <param name="builder">The builder instance.</param>
    /// <returns>The updated builder instance.</returns>
    public static Context.Builder Vulkan(this Context.Builder builder)
    {
        if (builder is null)
            throw new ArgumentNullException(nameof(builder));
        return builder.Vulkan(_ => true);
    }

    /// <summary>
    /// Enables all Vulkan devices matching a predicate.
    /// </summary>
    /// <param name="builder">The builder instance.</param>
    /// <param name="predicate">Filter for devices to include.</param>
    /// <returns>The updated builder instance.</returns>
    public static Context.Builder Vulkan(
        this Context.Builder builder,
        Predicate<VulkanDevice> predicate)
    {
        if (builder is null)
            throw new ArgumentNullException(nameof(builder));
        if (predicate is null)
            throw new ArgumentNullException(nameof(predicate));
        VulkanDevice.GetDevices(predicate, builder.DeviceRegistry);
        return builder;
    }
}
