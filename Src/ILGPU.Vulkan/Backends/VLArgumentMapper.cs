// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLArgumentMapper.cs
// ---------------------------------------------------------------------------------------

using System;
using ILGPU.Backends.EntryPoints;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Minimal argument mapper stub for Vulkan backend.
/// </summary>
[Obsolete("Not implemented")]
internal sealed class VLArgumentMapper : ArgumentMapper
{
    public VLArgumentMapper(Context context) : base(context) { }

    protected override Type MapViewType(Type sourceViewType, Type targetType)
    {
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
}
