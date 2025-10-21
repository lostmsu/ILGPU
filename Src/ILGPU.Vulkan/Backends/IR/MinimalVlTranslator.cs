// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: MinimalVlTranslator.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.IR;
using ILGPU.IR.Values;
using System;
using System.Collections.Generic;

namespace ILGPU.Backends.Vulkan;

internal sealed class MinimalVlTranslator
{
    private readonly SpirvModuleBuilder m;
    private readonly EntryPoint ep;
    private readonly Dictionary<Value, uint> idMap = [];
    private readonly Dictionary<int, (uint varId, bool unsigned)> paramBinding = [];

    public MinimalVlTranslator(SpirvModuleBuilder module, EntryPoint entryPoint)
    {
        m = module ?? throw new ArgumentNullException(nameof(module));
        ep = entryPoint ?? throw new ArgumentNullException(nameof(entryPoint));
    }

    public void TranslateKernel(in Backend.BackendContext ctx)
    {
        var method = ctx.KernelMethod;
        uint binding = 0;
        foreach (var p in method.Parameters)
        {
            if (!p.ParameterType.IsViewType)
                continue;
            bool unsigned = false;
            var (_, varId) = m.DeclareStorageBuffer(binding++, unsigned);
            paramBinding[p.Index] = (varId, unsigned);
        }

        foreach (var block in method.Blocks)
        {
            foreach (var entry in block)
            {
                var v = (Value)entry;
                switch (v.ValueKind)
                {
                    case ValueKind.GridIndex:
                    {
                        var g = (GridIndexValue)v;
                        idMap[v] = m.EmitLoadGlobalIndex(g.Dimension);
                        break;
                    }
                    case ValueKind.LoadElementAddress:
                    {
                        var lea = (LoadElementAddress)v;
                        var param = lea.Source.ResolveAs<Parameter>();
                        if (param is null)
                            throw new NotSupportedException("LEA source must be parameter");
                        if (!idMap.TryGetValue(lea.Offset, out var idx))
                        {
                            idx = m.EmitLoadGlobalIndex(DeviceConstantDimension3D.X);
                            idMap[lea.Offset.Resolve()] = idx;
                        }
                        var (varId, unsigned) = paramBinding[param.Index];
                        idMap[v] = m.AccessChainElement(varId, idx, unsigned);
                        break;
                    }
                    case ValueKind.Load:
                    {
                        var load = (Load)v;
                        var srcPtr = idMap[load.Source];
                        bool unsigned = false;
                        foreach (var kv in paramBinding.Values)
                        {
                            if (kv.varId == srcPtr) { unsigned = kv.unsigned; break; }
                        }
                        idMap[v] = m.EmitLoadScalar(srcPtr, unsigned);
                        break;
                    }
                    case ValueKind.BinaryArithmetic:
                    {
                        var bin = (BinaryArithmeticValue)v;
                        var a = idMap[bin.Left];
                        var b = idMap[bin.Right];
                        idMap[v] = m.EmitAdd(a, b, false);
                        break;
                    }
                    case ValueKind.Store:
                    {
                        var st = (Store)v;
                        m.EmitStore(idMap[st.Target], idMap[st.Value]);
                        break;
                    }
                    case ValueKind.Return:
                        break;
                    default:
                        break;
                }
            }
        }
    }
}

