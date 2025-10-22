// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: TestContext.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Vulkan;
using System;

namespace ILGPU.Tests.Vulkan;

/// <summary>
/// An abstract test context for Vulkan accelerators.
/// </summary>
public abstract class VulkanTestContext : TestContext
{
    /// <summary>
    /// Creates a new test context instance.
    /// </summary>
    /// <param name="optimizationLevel">The optimization level to use.</param>
    /// <param name="enableAssertions">Enables use of assertions.</param>
    /// <param name="forceDebugConfig">Forces use of debug configuration in O1 and O2 builds.</param>
    /// <param name="prepareContext">The context preparation handler.</param>
    protected VulkanTestContext(
        OptimizationLevel optimizationLevel,
        bool enableAssertions,
        bool forceDebugConfig,
        Action<Context.Builder> prepareContext)
        : base(
              optimizationLevel,
              enableAssertions,
              forceDebugConfig,
              builder => prepareContext(builder.Vulkan()),
              context => context.CreateVulkanAccelerator(0))
    { }
}

