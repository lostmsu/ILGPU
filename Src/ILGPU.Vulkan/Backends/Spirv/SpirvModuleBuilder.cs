// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: SpirvModuleBuilder.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.IR;
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using Silk.NET.SPIRV;
using System;
using System.Collections.Generic;

namespace ILGPU.Backends.Vulkan;

internal sealed class SpirvModuleBuilder
{
    public sealed class Origin
    {
        public uint ResultId { get; init; }
        public string Section { get; init; } = "func";
        public int Ordinal { get; init; }
        public uint BlockLabelId { get; init; }
        public string? Method { get; init; }
        public string? ValueKind { get; init; }
        public string? Value { get; init; }
    }

    public sealed class DebugMap
    {
        public string EntryPoint { get; init; } = string.Empty;
        public Origin[] Origins { get; init; } = Array.Empty<Origin>();
    }
    private readonly SpirvWriter w = new();
    private uint nextId = 1;
    private readonly EntryPoint ep;

    private readonly uint idVoid, idFuncType;
    private readonly uint idBool, idUint, idInt, idV3u;
    private uint idLong, idULong;
    private readonly Dictionary<BasicValueType, uint> primitiveTypeIds = new();
    private readonly Dictionary<uint, uint> functionPtrTypeIds = new();
    private readonly Dictionary<int, uint> constInt32 = new();
    private readonly Dictionary<long, uint> constInt64 = new();
    private uint constTrueId;
    private uint constFalseId;
    private readonly uint idPtrInV3u, idPtrStorageUint, idPtrStorageInt;
    private uint idPtrPushInt;
    private readonly uint idConst0;
    private readonly uint idGlGlobalInvocationId;
    private readonly uint idGlNumWorkgroups;
    private readonly uint entryFuncId;
    private bool functionStarted;
    private readonly Dictionary<BasicBlock, uint> blockLabels = new();

    // Sectioned emission
    private readonly List<Action<SpirvWriter>> capabilities = [];
    private Action<SpirvWriter>? memoryModel;
    private readonly List<Action<SpirvWriter>> entryAndModes = [];
    private readonly List<uint> entryInterface = [];
    private readonly List<Action<SpirvWriter>> annotations = [];
    private readonly List<Action<SpirvWriter>> typesGlobals = [];
    private readonly List<Action<SpirvWriter>> func = [];
    private uint currentBlockLabelId;
    private readonly Dictionary<uint, Origin> origins = new();

    private string? currentMethod;
    private string? currentValueKind;
    private string? currentValueText;
    // Push constants for view lengths
    private uint pcStructId;
    private uint pcVarId;
    private uint pcMemberCount;
    // Track basic-block termination between labels
    private bool hasAnyLabel;
    private bool lastBlockHasTerminator = true;

    public SpirvModuleBuilder(EntryPoint entryPoint)
    {
        ep = entryPoint;
        w.Header(0x00010300u, 0u, 1u, 0u);
        // Queue baseline capability and memory model for ordered emission later
        capabilities.Add(sw => sw.Write(Op.Capability, (uint)Capability.Shader));
        // Enable 64-bit integers to support kernels with 64-bit element types
        capabilities.Add(sw => sw.Write(Op.Capability, (uint)Capability.Int64));
        memoryModel = sw => sw.Write(
            Op.MemoryModel,
            (uint)AddressingModel.Logical,
            1u);

        // Prepare ids used by entry point and annotations
        idGlGlobalInvocationId = NewId();
        idGlNumWorkgroups = NewId();
        entryFuncId = NewId();

        // Entry point and execution mode (queued)
        entryInterface.Add(idGlGlobalInvocationId);
        entryAndModes.Add(sw => sw.OpEntryPointCompute(entryFuncId, "main", entryInterface.ToArray()));
        entryAndModes.Add(sw => sw.OpExecutionModeLocalSize(entryFuncId, 64u, 1u, 1u));

        // Annotations (decorations) queued
        annotations.Add(sw => sw.Write(Op.Decorate, idGlGlobalInvocationId, (uint)Decoration.BuiltIn, (uint)BuiltIn.GlobalInvocationId));
        annotations.Add(sw => sw.Write(Op.Decorate, idGlNumWorkgroups, (uint)Decoration.BuiltIn, (uint)BuiltIn.NumWorkgroups));

        // Types (queued)
        idVoid = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeVoid, idVoid));
        idBool = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeBool, idBool));
        idUint = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeInt, idUint, 32u, 0u));
        idInt = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeInt, idInt, 32u, 1u));
        // 64-bit integer types are declared lazily on first use with Int64 capability
        idV3u = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeVector, idV3u, idUint, 3u));
        idPtrInV3u = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrInV3u, (uint)StorageClass.Input, idV3u));
        idPtrStorageUint = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrStorageUint, (uint)StorageClass.StorageBuffer, idUint));
        idPtrStorageInt = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrStorageInt, (uint)StorageClass.StorageBuffer, idInt));
        idPtrPushInt = 0; // deferred until DeclarePushConstants
        idFuncType = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeFunction, idFuncType, idVoid));

        // Globals (queued)
        typesGlobals.Add(sw => sw.Write(Op.Variable, idPtrInV3u, idGlGlobalInvocationId, (uint)StorageClass.Input));
        typesGlobals.Add(sw => sw.Write(Op.Variable, idPtrInV3u, idGlNumWorkgroups, (uint)StorageClass.Input));

        // Constants (queued)
        idConst0 = NewId(); typesGlobals.Add(sw => sw.Write(Op.Constant, idUint, idConst0, 0u));

        functionStarted = false;
        currentBlockLabelId = 0;
    }

    public void SetCurrentOrigin(string method, string valueKind, string value)
    {
        currentMethod = method;
        currentValueKind = valueKind;
        currentValueText = value;
    }


    public void DeclarePushConstants(uint lengthCount)
    {
        if (pcVarId != 0 || lengthCount == 0)
            return;
        pcMemberCount = lengthCount;
        var members = new List<uint>((int)lengthCount);
        for (uint i = 0; i < lengthCount; i++)
            members.Add(idInt);
        pcStructId = NewId();
        var ops = new List<uint>(1 + members.Count);
        ops.Add(pcStructId);
        ops.AddRange(members);
        typesGlobals.Add(sw => sw.Write(Op.TypeStruct, ops.ToArray()));
        // Decorate members with offsets
        for (uint i = 0; i < lengthCount; i++)
        {
            var offset = i * 4u;
            var member = i; // member index
            annotations.Add(sw => sw.Write(Op.MemberDecorate, pcStructId, member, (uint)Decoration.Offset, offset));
        }
        annotations.Add(sw => sw.Write(Op.Decorate, pcStructId, (uint)Decoration.Block));
        // Pointer type and variable
        idPtrPushInt = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, idPtrPushInt, (uint)StorageClass.PushConstant, idInt));
        var ptrStruct = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypePointer, ptrStruct, (uint)StorageClass.PushConstant, pcStructId));
        pcVarId = NewId(); typesGlobals.Add(sw => sw.Write(Op.Variable, ptrStruct, pcVarId, (uint)StorageClass.PushConstant));
        // Optional: debug names can be added here if writer supports string literals
    }

    public void BeginFunction()
    {
        if (functionStarted) return;
        func.Add(sw => sw.Write(Op.Function, idVoid, entryFuncId, 0u, idFuncType));
        functionStarted = true;
    }

    public uint NewId() => nextId++;

    public void DeclareBlocks(Method method)
    {
        foreach (var block in method.Blocks)
            blockLabels[block] = NewId();
    }

    public void BeginBlock(BasicBlock block)
    {
        BeginFunction();
        if (blockLabels.TryGetValue(block, out var lbl))
        {
            func.Add(sw => sw.Write(Op.Label, lbl));
            currentBlockLabelId = lbl;
            hasAnyLabel = true;
            lastBlockHasTerminator = false;
        }
        else
        {
            var newLbl = NewId();
            blockLabels[block] = newLbl;
            func.Add(sw => sw.Write(Op.Label, newLbl));
            currentBlockLabelId = newLbl;
            hasAnyLabel = true;
            lastBlockHasTerminator = false;
        }
    }

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

    public uint EmitLoadNumWorkgroups(DeviceConstantDimension3D dim)
    {
        BeginFunction();
        var ng = NewId(); func.Add(sw => sw.Write(Op.Load, idV3u, ng, idGlNumWorkgroups));
        var comp = dim switch { DeviceConstantDimension3D.Y => 1u, DeviceConstantDimension3D.Z => 2u, _ => 0u };
        var val = NewId(); func.Add(sw => sw.Write(Op.CompositeExtract, idUint, val, ng, comp));
        return val;
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

    private bool capInt64Added;
    private void EnsureInt64Capability()
    {
        if (!capInt64Added)
        {
            capInt64Added = true;
            capabilities.Add(sw => sw.Write(Op.Capability, (uint)Capability.Int64));
        }
    }

    private uint GetIntTypeId(int width, bool unsigned)
    {
        if (width == 64)
        {
            // Capability Int64 required when using 64-bit types
            EnsureInt64Capability();
            if (unsigned)
            {
                if (idULong == 0)
                {
                    idULong = NewId();
                    typesGlobals.Add(sw => sw.Write(Op.TypeInt, idULong, 64u, 0u));
                }
                return idULong;
            }
            else
            {
                if (idLong == 0)
                {
                    idLong = NewId();
                    typesGlobals.Add(sw => sw.Write(Op.TypeInt, idLong, 64u, 1u));
                }
                return idLong;
            }
        }
        return unsigned ? idUint : idInt;
    }

    public uint EmitAdd(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        func.Add(sw => sw.Write(Op.IAdd, elem, res, aId, bId));
        return res;
    }

    public uint EmitSub(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        func.Add(sw => sw.Write(Op.ISub, elem, res, aId, bId));
        return res;
    }

    public uint EmitMul(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        func.Add(sw => sw.Write(Op.IMul, elem, res, aId, bId));
        return res;
    }

    public uint EmitDiv(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        var op = unsigned ? Op.UDiv : Op.SDiv;
        func.Add(sw => sw.Write(op, elem, res, aId, bId));
        return res;
    }

    public uint EmitRem(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        var op = unsigned ? Op.UMod : Op.SRem;
        func.Add(sw => sw.Write(op, elem, res, aId, bId));
        return res;
    }

    public uint EmitShiftLeft(uint aId, uint bId, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, false);
        var res = NewId();
        func.Add(sw => sw.Write(Op.ShiftLeftLogical, elem, res, aId, bId));
        return res;
    }

    public uint EmitShiftRight(uint aId, uint bId, bool unsigned, int width = 32)
    {
        BeginFunction();
        var elem = GetIntTypeId(width, unsigned);
        var res = NewId();
        var op = unsigned ? Op.ShiftRightLogical : Op.ShiftRightArithmetic;
        func.Add(sw => sw.Write(op, elem, res, aId, bId));
        return res;
    }

    public uint EmitBitwiseAnd(uint aId, uint bId, int width = 32)
    {
        BeginFunction();
        var res = NewId();
        var ty = width == 64 ? GetIntTypeId(64, false) : idInt;
        func.Add(sw => sw.Write(Op.BitwiseAnd, ty, res, aId, bId));
        return res;
    }

    public uint EmitBitwiseOr(uint aId, uint bId, int width = 32)
    {
        BeginFunction();
        var res = NewId();
        var ty = width == 64 ? GetIntTypeId(64, false) : idInt;
        func.Add(sw => sw.Write(Op.BitwiseOr, ty, res, aId, bId));
        return res;
    }

    public uint EmitBitwiseXor(uint aId, uint bId, int width = 32)
    {
        BeginFunction();
        var res = NewId();
        var ty = width == 64 ? GetIntTypeId(64, false) : idInt;
        func.Add(sw => sw.Write(Op.BitwiseXor, ty, res, aId, bId));
        return res;
    }

    public uint EmitLogicalAnd(uint aBoolId, uint bBoolId)
    {
        BeginFunction();
        var res = NewId();
        func.Add(sw => sw.Write(Op.LogicalAnd, idBool, res, aBoolId, bBoolId));
        return res;
    }

    public uint EmitLogicalOr(uint aBoolId, uint bBoolId)
    {
        BeginFunction();
        var res = NewId();
        func.Add(sw => sw.Write(Op.LogicalOr, idBool, res, aBoolId, bBoolId));
        return res;
    }

    public uint EmitSelect(uint condBoolId, uint trueValId, uint falseValId, bool unsigned)
    {
        BeginFunction();
        var elem = unsigned ? idUint : idInt;
        var res = NewId();
        func.Add(sw => sw.Write(Op.Select, elem, res, condBoolId, trueValId, falseValId));
        return res;
    }

    public uint EmitMulUint(uint aId, uint bId)
    {
        BeginFunction();
        var res = NewId();
        func.Add(sw => sw.Write(Op.IMul, idUint, res, aId, bId));
        return res;
    }

    public void EmitStore(uint ptr, uint val) { BeginFunction(); func.Add(sw => sw.Write(Op.Store, ptr, val)); }

    public uint EmitIntBitcast(uint valueId, bool toUnsigned)
    {
        BeginFunction();
        var targetType = toUnsigned ? idUint : idInt;
        var res = NewId();
        func.Add(sw => sw.Write(Op.Bitcast, targetType, res, valueId));
        return res;
    }

    public uint EmitConvertI32ToI64(uint srcId, bool unsigned)
    {
        BeginFunction();
        // Ensure 64-bit integer type
        var dstTy = GetIntTypeId(64, unsigned);
        var res = NewId();
        var op = unsigned ? Op.UConvert : Op.SConvert;
        func.Add(sw => sw.Write(op, dstTy, res, srcId));
        return res;
    }

    public uint EmitLoadViewLength(uint index)
    {
        if (pcVarId == 0)
            throw new InvalidOperationException("Push constants not declared");
        BeginFunction();
        // Access member index
        var memberIndexConstId = EmitConstInt32((int)index);
        var ptr = NewId();
        func.Add(sw => sw.Write(Op.AccessChain, idPtrPushInt, ptr, pcVarId, memberIndexConstId));
        var val = NewId();
        func.Add(sw => sw.Write(Op.Load, idInt, val, ptr));
        return val;
    }

    private uint EnsurePrimitiveType(BasicValueType bvt)
    {
        if (primitiveTypeIds.TryGetValue(bvt, out var id))
            return id;
        uint newId = 0;
        switch (bvt)
        {
            case BasicValueType.Int1:
                newId = idBool; break;
            case BasicValueType.Int32:
                newId = idInt; break;
            case BasicValueType.Int64:
                // Enable Int64 capability and lazily declare 64-bit int type
                EnsureInt64Capability();
                if (idLong == 0)
                {
                    idLong = NewId();
                    typesGlobals.Add(sw => sw.Write(Op.TypeInt, idLong, 64u, 1u));
                }
                newId = idLong; break;
            case BasicValueType.Float32:
                newId = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeFloat, newId, 32u)); break;
            case BasicValueType.Float64:
                newId = NewId(); typesGlobals.Add(sw => sw.Write(Op.TypeFloat, newId, 64u)); break;
            default:
                // Fallback to 32-bit int
                newId = idInt; break;
        }
        primitiveTypeIds[bvt] = newId;
        return newId;
    }

    private uint EnsureFunctionPtrType(uint elemTypeId)
    {
        if (functionPtrTypeIds.TryGetValue(elemTypeId, out var ptrId))
            return ptrId;
        var newPtr = NewId();
        typesGlobals.Add(sw => sw.Write(Op.TypePointer, newPtr, (uint)StorageClass.Function, elemTypeId));
        functionPtrTypeIds[elemTypeId] = newPtr;
        return newPtr;
    }

    public uint DeclareFunctionVariable(TypeNode elemType)
    {
        BeginFunction();
        var prim = elemType as PrimitiveType;
        var elemId = EnsurePrimitiveType(prim?.BasicValueType ?? BasicValueType.Int32);
        var ptrType = EnsureFunctionPtrType(elemId);
        var varId = NewId();
        func.Add(sw => sw.Write(Op.Variable, ptrType, varId, (uint)StorageClass.Function));
        return varId;
    }

    public uint EmitConstInt32(int value)
    {
        if (constInt32.TryGetValue(value, out var id))
            return id;
        var newId = NewId();
        typesGlobals.Add(sw => sw.Write(Op.Constant, idInt, newId, unchecked((uint)value)));
        constInt32[value] = newId;
        origins[newId] = new Origin
        {
            ResultId = newId,
            Section = "types",
            Ordinal = typesGlobals.Count - 1,
            BlockLabelId = 0,
            Method = currentMethod,
            ValueKind = currentValueKind,
            Value = currentValueText,
        };
        return newId;
    }

    public uint EmitConstInt64(long value)
    {
        if (constInt64.TryGetValue(value, out var id))
            return id;
        // Ensure capability and type
        EnsureInt64Capability();
        if (idLong == 0)
        {
            idLong = NewId();
            typesGlobals.Add(sw => sw.Write(Op.TypeInt, idLong, 64u, 1u));
        }
        var newId = NewId();
        // Constants take their bit-pattern split into 32-bit words (low, high)
        unchecked
        {
            var low = (uint)(value & 0xFFFFFFFFL);
            var high = (uint)((ulong)value >> 32);
            typesGlobals.Add(sw => sw.Write(Op.Constant, idLong, newId, low, high));
        }
        constInt64[value] = newId;
        return newId;
    }

    public uint EmitConstBool(bool value)
    {
        if (value)
        {
            if (constTrueId != 0) return constTrueId;
            constTrueId = NewId();
            typesGlobals.Add(sw => sw.Write(Op.ConstantTrue, idBool, constTrueId));
            return constTrueId;
        }
        else
        {
            if (constFalseId != 0) return constFalseId;
            constFalseId = NewId();
            typesGlobals.Add(sw => sw.Write(Op.ConstantFalse, idBool, constFalseId));
            return constFalseId;
        }
    }

    public uint EmitConvertI64ToI32(uint srcId, bool unsigned)
    {
        BeginFunction();
        var res = NewId();
        var op = unsigned ? Op.UConvert : Op.SConvert;
        func.Add(sw => sw.Write(op, idInt, res, srcId));
        return res;
    }

    public void EmitReturn()
    {
        BeginFunction();
        func.Add(sw => sw.Write(Op.Return));
        lastBlockHasTerminator = true;
    }

    public void EmitBranch(BasicBlock target)
    {
        BeginFunction();
        var targetId = blockLabels[target];
        func.Add(sw => sw.Write(Op.Branch, targetId));
        lastBlockHasTerminator = true;
    }

    public void EmitBranchConditional(uint condId, BasicBlock trueTarget, BasicBlock falseTarget)
    {
        BeginFunction();
        var t = blockLabels[trueTarget];
        var f = blockLabels[falseTarget];
        func.Add(sw => sw.Write(Op.BranchConditional, condId, t, f));
        lastBlockHasTerminator = true;
    }

    public void EmitSelectionMerge(BasicBlock merge)
    {
        BeginFunction();
        var m = blockLabels[merge];
        func.Add(sw => sw.Write(Op.SelectionMerge, m, 0u));
    }

    public uint EmitPhi(TypeNode type, ReadOnlySpan<(BasicBlock Pred, uint ValueId)> incomings)
    {
        BeginFunction();
        var prim = type as PrimitiveType;
        var typeId = EnsurePrimitiveType(prim?.BasicValueType ?? BasicValueType.Int32);
        var resultId = NewId();
        // Build operands: [resultType, resultId, (value, label)*]
        var ops = new List<uint>(2 + incomings.Length * 2) { typeId, resultId };
        foreach (var inc in incomings)
        {
            ops.Add(inc.ValueId);
            ops.Add(blockLabels[inc.Pred]);
        }
        func.Add(sw =>
        {
            // Emit raw with dynamic operand array
            sw.Write(Op.Phi, ops.ToArray());
        });
        return resultId;
    }

    public uint EmitCompareInt(uint leftId, uint rightId, CompareKind kind, bool unsigned)
    {
        BeginFunction();
        var res = NewId();
        Op op = kind switch
        {
            CompareKind.Equal => Op.IEqual,
            CompareKind.NotEqual => Op.INotEqual,
            CompareKind.LessThan => unsigned ? Op.ULessThan : Op.SLessThan,
            CompareKind.LessEqual => unsigned ? Op.ULessThanEqual : Op.SLessThanEqual,
            CompareKind.GreaterThan => unsigned ? Op.UGreaterThan : Op.SGreaterThan,
            CompareKind.GreaterEqual => unsigned ? Op.UGreaterThanEqual : Op.SGreaterThanEqual,
            _ => Op.IEqual,
        };
        func.Add(sw => sw.Write(op, idBool, res, leftId, rightId));
        return res;
    }

    public void SetOrigin(uint resultId, string? method, string? valueKind, string? value)
    {
        var ordinal = func.Count;
        origins[resultId] = new Origin
        {
            ResultId = resultId,
            Section = "func",
            Ordinal = ordinal,
            BlockLabelId = currentBlockLabelId,
            Method = method,
            ValueKind = valueKind,
            Value = value,
        };
    }

    public DebugMap BuildDebugMap()
    {
        var arr = new List<Origin>(origins.Values).ToArray();
        return new DebugMap
        {
            EntryPoint = ep.MethodInfo.Name,
            Origins = arr,
        };
    }

    public byte[] ToArray()
    {
        if (functionStarted)
        {
            if (!lastBlockHasTerminator)
                func.Add(sw => sw.Write(Op.Return));
            func.Add(sw => sw.Write(Op.FunctionEnd));
        }
        // Emit sections in required order
        foreach (var c in capabilities) c(w);
        memoryModel?.Invoke(w);
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
            if (!lastBlockHasTerminator)
                func.Add(sw => sw.Write(Op.Return));
            func.Add(sw => sw.Write(Op.FunctionEnd));
        }
        foreach (var c in capabilities) c(w);
        memoryModel?.Invoke(w);
        foreach (var e in entryAndModes) e(w);
        foreach (var a in annotations) a(w);
        foreach (var t in typesGlobals) t(w);
        foreach (var f in func) f(w);
        w.PatchBound(nextId);
        return w.ToUIntArray();
    }
}
