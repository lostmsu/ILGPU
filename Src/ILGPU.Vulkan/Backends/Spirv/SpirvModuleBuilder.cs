// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: SpirvModuleBuilder.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.IR.Values;
using Silk.NET.SPIRV;
using System;
using System.Collections.Generic;

namespace ILGPU.Backends.Vulkan;

internal sealed class SpirvModuleBuilder
{
    private readonly SpirvWriter w = new();
    private uint nextId = 1;
    private readonly EntryPoint ep;

    private readonly uint idVoid, idFuncType;
    private readonly uint idUint, idInt, idV3u;
    private readonly uint idPtrInV3u, idPtrStorageUint, idPtrStorageInt;
    private readonly uint idConst0;
    private readonly uint idGlGlobalInvocationId;
    private readonly uint entryFuncId, labelId;
    private bool functionStarted;

    // Sectioned emission
    private readonly List<Action<SpirvWriter>> entryAndModes = [];
    private readonly List<Action<SpirvWriter>> annotations = [];
    private readonly List<Action<SpirvWriter>> typesGlobals = [];
    private readonly List<Action<SpirvWriter>> func = [];

    public SpirvModuleBuilder(EntryPoint entryPoint)
    {
        ep = entryPoint;
        w.Header(0x00010300u, 0u, 1u, 0u);
        w.Write(Op.Capability, (uint)Capability.Shader);
        // MemoryModel: GLSL450 (1)
        w.Write(Op.MemoryModel,
            (uint)AddressingModel.Logical,
            1u);

        // Prepare ids used by entry point and annotations
        idGlGlobalInvocationId = NewId();
        entryFuncId = NewId();
        labelId = NewId();

        // Entry point and execution mode (queued)
        entryAndModes.Add(sw => sw.OpEntryPointCompute(entryFuncId, "main", [idGlGlobalInvocationId]));
        entryAndModes.Add(sw => sw.OpExecutionModeLocalSize(entryFuncId, 64u, 1u, 1u));

        // Annotations (decorations) queued
        annotations.Add(sw => sw.Write(Op.Decorate, idGlGlobalInvocationId, (uint)Decoration.BuiltIn, (uint)BuiltIn.GlobalInvocationId));

        // Types (queued)
        idVoid = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeVoid, idVoid));
        idUint = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeInt, idUint, 32u, 0u));
        idInt = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeInt, idInt, 32u, 1u));
        idV3u = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeVector, idV3u, idUint, 3u));
        idPtrInV3u = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrInV3u, (uint)StorageClass.Input, idV3u));
        idPtrStorageUint = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrStorageUint, (uint)StorageClass.StorageBuffer, idUint));
        idPtrStorageInt = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrStorageInt, (uint)StorageClass.StorageBuffer, idInt));
        idFuncType = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeFunction, idFuncType, idVoid));

        // Globals (queued)
        typesGlobals.Add(sw => sw.Write(Op.Variable, idPtrInV3u, idGlGlobalInvocationId, (uint)StorageClass.Input));

        // Constants (queued)
        idConst0 = NewId(); typesGlobals.Add(sw => sw.Write(Op.Constant, idUint, idConst0, 0u));

        functionStarted = false;
    }

    public void BeginFunction()
    {
        if (functionStarted) return;
        func.Add(sw => sw.Write(Op.Function, idVoid, entryFuncId, 0u, idFuncType));
        func.Add(sw => sw.Write(Op.Label, labelId));
        functionStarted = true;
    }

    public uint NewId() => nextId++;

    public (uint structId, uint varId) DeclareStorageBuffer(uint binding, bool unsigned)
    {
        var elem = unsigned ? idUint : idInt;
        var idRuntimeArr = NewId();
        // Queue annotations for the runtime array and struct
        annotations.Add(sw => sw.Write(Op.Decorate, idRuntimeArr, (uint)Decoration.ArrayStride, 4u));
        var structId = NewId();
        annotations.Add(sw => sw.Write(Op.MemberDecorate, structId, 0u, (uint)Decoration.Offset, 0u));
        annotations.Add(sw => sw.Write(Op.Decorate, structId, (uint)Decoration.Block));
        // Queue types and globals
        typesGlobals.Add(sw => sw.Write(Op.TypeRuntimeArray, idRuntimeArr, elem));
        typesGlobals.Add(sw => sw.Write(Op.TypeStruct, structId, idRuntimeArr));
        var ptrStruct = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, ptrStruct, (uint)StorageClass.StorageBuffer, structId));
        var varId = NewId(); typesGlobals.Add(sw => sw.Write(Op.Variable, ptrStruct, varId, (uint)StorageClass.StorageBuffer));
        // Variable descriptor decorations (annotation section)
        annotations.Add(sw => sw.Write(Op.Decorate, varId, (uint)Decoration.DescriptorSet, 0u));
        annotations.Add(sw => sw.Write(Op.Decorate, varId, (uint)Decoration.Binding, binding));
        return (structId, varId);
    }

    public uint EmitLoadGlobalIndex(DeviceConstantDimension3D dim)
    {
        BeginFunction();
        var gid = NewId(); func.Add(sw => sw.Write(Op.Load, idV3u, gid, idGlGlobalInvocationId));
        var comp = dim switch { DeviceConstantDimension3D.Y => 1u, DeviceConstantDimension3D.Z => 2u, _ => 0u };
        var idx = NewId(); func.Add(sw => sw.Write(Op.CompositeExtract, idUint, idx, gid, comp));
        return idx;
    }

    public uint AccessChainElement(uint bufVarId, uint indexId, bool unsigned)
    {
        BeginFunction();
        var ptrType = unsigned ? idPtrStorageUint : idPtrStorageInt;
        var ptr = NewId();
        func.Add(sw => sw.Write(Op.AccessChain, ptrType, ptr, bufVarId, idConst0, indexId));
        return ptr;
    }

    public uint EmitLoadScalar(uint ptrId, bool unsigned)
    {
        BeginFunction();
        var elem = unsigned ? idUint : idInt;
        var val = NewId();
        func.Add(sw => sw.Write(Op.Load, elem, val, ptrId));
        return val;
    }

    public uint EmitAdd(uint aId, uint bId, bool unsigned)
    {
        BeginFunction();
        var elem = unsigned ? idUint : idInt;
        var res = NewId();
        func.Add(sw => sw.Write(Op.IAdd, elem, res, aId, bId));
        return res;
    }

    public void EmitStore(uint ptr, uint val) { BeginFunction(); func.Add(sw => sw.Write(Op.Store, ptr, val)); }

    public byte[] ToArray()
    {
        if (functionStarted)
        {
            func.Add(sw => sw.Write(Op.Return));
            func.Add(sw => sw.Write(Op.FunctionEnd));
        }
        // Emit sections in required order
        foreach (var e in entryAndModes) e(w);
        foreach (var a in annotations) a(w);
        foreach (var t in typesGlobals) t(w);
        foreach (var f in func) f(w);
        w.PatchBound(nextId);
        return w.ToArray();
    }

    public uint[] ToUIntArray()
    {
        if (functionStarted)
        {
            func.Add(sw => sw.Write(Op.Return));
            func.Add(sw => sw.Write(Op.FunctionEnd));
        }
        foreach (var e in entryAndModes) e(w);
        foreach (var a in annotations) a(w);
        foreach (var t in typesGlobals) t(w);
        foreach (var f in func) f(w);
        w.PatchBound(nextId);
        return w.ToUIntArray();
    }
}

