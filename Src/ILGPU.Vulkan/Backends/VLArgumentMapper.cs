// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLArgumentMapper.cs
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Argument mapper for Vulkan backend.
/// </summary>
internal sealed class VLArgumentMapper : ArgumentMapper
{
    public VLArgumentMapper(Context context) : base(context) { }

    protected override Type MapViewType(Type sourceViewType, Type targetType)
    {
        // Vulkan uses native ArrayView parameters at launch time; no remapping here.
        return sourceViewType;
    }

    protected override void MapViewInstance<TILEmitter, TSource, TTarget>(
        in TILEmitter emitter,
        Type targetType,
        in TSource source,
        in TTarget target)
    {
        // No-op: Vulkan descriptors are prepared at dispatch.
    }

    internal sealed record NonViewPcInfo(int ParamIndex, int StartSlot, int WordCount);

    internal sealed record Plan(
        IReadOnlyList<ILGPU.Backends.Vulkan.VLCompiledKernel.BindingInfo> Bindings,
        int PushConstantCount,
        int[] ViewParamIndices,
        int[] ScalarIntParamIndices,
        NonViewPcInfo[] NonViewPc)
    {
        public int ViewCount => ViewParamIndices.Length;
        public int ScalarIntCount => ScalarIntParamIndices.Length;
        public int TotalScalarWordCount => ScalarIntCount + (NonViewPc?.Sum(i => i.WordCount) ?? 0);
    }

    private static bool IsAnyArrayViewType(Type t)
    {
        if (!t.IsGenericType) return false;
        var g = t.GetGenericTypeDefinition();
        return g == typeof(ArrayView<>) ||
               g == typeof(ArrayView1D<,>) ||
               g == typeof(ArrayView2D<,>) ||
               g == typeof(ArrayView3D<,>);
    }

    /// <summary>
    /// Builds a mapping plan for the given entry point. Bindings are assigned
    /// sequentially to view parameters in declaration order. Push-constant slots
    /// include first all view lengths (1 per view) followed by scalar int parameters.
    /// </summary>
    public Plan BuildPlan(EntryPoint entryPoint)
    {
        if (entryPoint is null) throw new ArgumentNullException(nameof(entryPoint));

        var bindings = new List<ILGPU.Backends.Vulkan.VLCompiledKernel.BindingInfo>();
        var viewIdx = new List<int>();
        var scalarIdx = new List<int>();
        var nonViewPc = new List<NonViewPcInfo>();

        uint binding = 0;
        var parameters = entryPoint.Parameters;
        for (int i = 0; i < parameters.Count; i++)
        {
            var pt = parameters[i];
            if (IsAnyArrayViewType(pt))
            {
                bindings.Add(new ILGPU.Backends.Vulkan.VLCompiledKernel.BindingInfo(i, binding));
                viewIdx.Add(i);
                binding++;
            }
            else if (pt == typeof(int))
            {
                scalarIdx.Add(i);
            }
        }

        // After view lengths and scalar ints, non-view packing is handled at runtime.
        // The compiled kernel's PushConstantCount (from IR sizes) is used for allocation.
        var pushConstantCount = checked((int)binding) + scalarIdx.Count;
        return new Plan(bindings, pushConstantCount, viewIdx.ToArray(), scalarIdx.ToArray(), Array.Empty<NonViewPcInfo>());
    }
}
