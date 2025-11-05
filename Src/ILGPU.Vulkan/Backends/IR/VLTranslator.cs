// NOTE: This translator mirrors CL/CUDA structure and consumes lowered IR.
// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLTranslator.cs
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.EntryPoints;
using ILGPU.IR;
using ILGPU.IR.Analyses;
using ILGPU.IR.Analyses.ControlFlowDirection;
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using System;
using System.Collections.Generic;

namespace ILGPU.Backends.Vulkan;

// Clean Vulkan translator modeled after CL/CUDA backends.
// Only supports lowered IR. For unsupported constructs, throws NotImplementedException.
internal sealed class VLTranslator
{
    private readonly SpirvModuleBuilder m;
    private readonly EntryPoint ep;
    private readonly Dictionary<Value, uint> ids = new();
    private readonly List<VLCompiledKernel.BindingInfo> bindings = new();
    private readonly Dictionary<int, (uint varId, uint binding, int elemSize)> viewParams = new();
    private readonly Dictionary<uint, int> bindingElemSize = new();
    private readonly Dictionary<uint, (uint binding, int elemSize)> varIdToBinding = new();
    private readonly Dictionary<int, uint> scalarPcIndex = new();
    private readonly Dictionary<int, (uint start, int words, uint[] wordIds)> pcParamWords = new();
    private int viewBindingCount;
    private Dominators<Backwards>? postDom;
    private readonly Dictionary<LoadElementAddress, (uint binding, uint totalBytes, int bindingElemSize, uint varId)> leaInfo = new();
    private readonly Dictionary<uint, (uint binding, uint totalBytes, int bindingElemSize, uint varId)> leaPtrInfo = new();
    private readonly Dictionary<AlignTo, (uint binding, uint varId, int elemSize, uint deltaBytesId, uint capElemId, uint mainLenId)> alignToInfo = new();
    private readonly Dictionary<Value, (uint binding, uint varId, int elemSize, uint addBytesId, uint overrideLenId)> derivedViews = new();

    public VLTranslator(SpirvModuleBuilder module, EntryPoint entryPoint)
    {
        m = module; ep = entryPoint;
    }

    public VLCompiledKernel.BindingInfo[] GetBindings() => bindings.ToArray();
    public int GetPushConstantCount() => viewBindingCount * 2 + scalarPcIndex.Count;

    public void TranslateKernel(in Backend.BackendContext backendContext)
    {
        var method = backendContext.KernelMethod;
        // Map parameters
        uint binding = 0;
        for (int i = 0; i < ep.Parameters.Count; i++)
        {
            var pt = ep.Parameters[i];
            VLTrace.Log($"EP.Param[{i}] type={pt}");
            if (IsArrayView(pt))
            {
                int es = GetElemSize(pt);
                var dec = m.DeclareStorageBuffer(binding, es, unsigned: false);
                viewParams[ep.KernelIndexParameterOffset + i] = (dec.varId, binding, es);
                bindingElemSize[binding] = es;
                varIdToBinding[dec.varId] = (binding, es);
                bindings.Add(new VLCompiledKernel.BindingInfo(i, binding));
                binding++;
            }
        }
        viewBindingCount = (int)binding;
        VLTrace.Log($"Mapped viewParams keys: [{string.Join(",", viewParams.Keys)}]; viewBindingCount={viewBindingCount}");
        // Scalar ints after 2N view slots
        uint scalarCount = 0;
        for (int i = 0; i < ep.Parameters.Count; i++)
        {
            var pt = ep.Parameters[i];
            if (pt == typeof(int))
            {
                int irParamIndex = ep.KernelIndexParameterOffset + i;
                scalarPcIndex[irParamIndex] = (uint)(viewBindingCount * 2 + scalarCount);
                scalarCount++;
            }
        }
        int totalPc = (int)(viewBindingCount * 2 + scalarCount);
        if (totalPc > 0) m.DeclarePushConstants((uint)totalPc);

        // Compute post-dominators for structured control flow
        postDom = method.Blocks.CreatePostDominators();

        // Entry block: materialize scalar push constants
        m.DeclareBlocks(method);
        var entry = method.EntryBlock;
        m.BeginBlock(entry);
        foreach (var p in method.Parameters)
        {
            if (scalarPcIndex.TryGetValue(p.Index, out var slot))
            {
                var w0 = m.EmitLoadPushConstant(slot);
                pcParamWords[p.Index] = (slot, 1, new uint[] { w0 });
                // Also expose directly in ids so uses via GetId work without an explicit Visit(Parameter)
                ids[p] = w0;
            }
            // Do not map view parameters here — they must be accessed via GetField
        }

        // Walk blocks and emit
        var visitor = new Visitor(this);
        foreach (var block in method.Blocks)
        {
            if (!ReferenceEquals(block, entry))
                m.BeginBlock(block);
            foreach (var value in block)
                value.Accept(visitor);
            // Emit terminator at the end of the block
            block.Terminator?.Accept(visitor);
        }
    }
    private static bool IsArrayView(Type t) => t.FullName?.StartsWith("ILGPU.Runtime.ArrayView") == true;
    private static int GetElemSize(Type t)
    {
        if (t.IsGenericType)
        {
            var g = t.GetGenericArguments()[0];
            if (g == typeof(int) || g == typeof(uint) || g == typeof(float)) return 4;
            if (g == typeof(long) || g == typeof(ulong) || g == typeof(double)) return 8;
        }
        return 4;
    }

    private static int GetElemSize(TypeNode t)
    {
        if (t is PrimitiveType pt)
        {
            return pt.BasicValueType switch
            {
                BasicValueType.Int64 or BasicValueType.Float64 => 8,
                BasicValueType.Int32 or BasicValueType.Float32 => 4,
                BasicValueType.Int16 or BasicValueType.Float16 => 2,
                BasicValueType.Int8 or BasicValueType.Int1 => 1,
                _ => throw new NotImplementedException($"GetElemSize({pt.BasicValueType})"),
            };
        }
        throw new NotImplementedException("What is the semantics of element size for non-primitive types?");
    }

    private bool TryResolveViewParam(Value v, out uint binding, out int elemSize)
    {
        binding = 0; elemSize = 4;
        var r = v.Resolve();
        if (r is Parameter p && viewParams.TryGetValue(p.Index, out var info))
        { binding = info.binding; elemSize = info.elemSize; return true; }
        return false;
    }

    private uint GetId(Value v)
    {
        if (ids.TryGetValue(v, out var id)) return id;
        var r = v.Resolve();
        if (r is Parameter p)
        {
            if (scalarPcIndex.TryGetValue(p.Index, out var slot))
            {
                var pc = m.EmitLoadPushConstant(slot);
                ids[p] = pc; return pc;
            }
            if (viewParams.TryGetValue(p.Index, out var _))
                throw new NotImplementedException("Attempted to use a view parameter as a scalar value; expected lowered field access.");
            // Assume kernel index parameter (use X component)
            var gid = m.EmitLoadGlobalIndex(DeviceConstantDimension3D.X);
            ids[p] = gid; return gid;
        }
        throw new NotImplementedException($"Unsupported value kind: {v.ValueKind}");
    }

    private readonly struct Visitor : IValueVisitor
    {
        private readonly VLTranslator t;
        public Visitor(VLTranslator t) { this.t = t; }

        private bool TryGetStorageVarId(Value v, out uint varId)
        {
            varId = 0;
            var r = v.Resolve();
            // unwrap trivial shells
            while (true)
            {
                switch (r.ValueKind)
                {
                    case ValueKind.ViewCast:
                        r = ((ViewCast)r).Value.Resolve(); continue;
                    case ValueKind.AddressSpaceCast:
                        r = ((AddressSpaceCast)r).Value.Resolve(); continue;
                    case ValueKind.Convert:
                        r = ((ConvertValue)r).Value.Resolve(); continue;
                    case ValueKind.AsAligned:
                        r = ((AsAligned)r).Source.Resolve(); continue;
                }
                break;
            }
            if (r is GetField gf && gf.FieldSpan.Index == 0)
            {
                var obj = gf.ObjectValue.Resolve();
                if (obj is Parameter p && t.viewParams.TryGetValue(p.Index, out var info))
                {
                    varId = info.varId; return true;
                }
            }
            return false;
        }

        private bool TryBuildPointerBytes(Value v, out uint varId, out uint binding, out uint byteOffset, out int elemSize)
        {
            // Start with zero byte offset
            byteOffset = t.m.EmitConstInt32(0);
            var cur = v.Resolve();
            // Pending pointer AlignTo alignments to apply once binding/base are known
            System.Collections.Generic.List<uint>? pendingPtrAlignments = null;
            System.Text.StringBuilder? sbTrace = null;
            if (Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE") == "1")
            {
                sbTrace = new System.Text.StringBuilder();
                sbTrace.Append("LEA source chain: ");
                sbTrace.Append(cur.ValueKind);
            }
            // Unwrap simple shells
            while (true)
            {
                switch (cur.ValueKind)
                {
                    case ValueKind.GetField:
                    {
                        var gf = (GetField)cur;
                        if (sbTrace != null) { sbTrace.Append("[field="); sbTrace.Append(gf.FieldSpan.Index); sbTrace.Append("]"); }
                        // If field 0 (pointer field), try to resolve via parameter mapping first
                        if (gf.FieldSpan.Index == 0)
                        {
                            var obj = gf.ObjectValue.Resolve();
                            if (obj is Parameter p0 && t.viewParams.TryGetValue(p0.Index, out var info0))
                            {
                                varId = info0.varId; binding = info0.binding; elemSize = info0.elemSize; return true;
                            }
                            // Otherwise, if the GetField result was already materialized to a var id, try reverse map
                            if (t.ids.TryGetValue(gf, out var gfVarId) && t.varIdToBinding.TryGetValue(gfVarId, out var bindInfo))
                            {
                                varId = gfVarId; binding = bindInfo.binding; elemSize = bindInfo.elemSize; return true;
                            }
                        }
                        // Otherwise, continue unwrapping the object
                        cur = gf.ObjectValue.Resolve();
                        if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); }
                        continue;
                    }
                    case ValueKind.LoadFieldAddress: cur = ((LoadFieldAddress)cur).Source.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.ViewCast:
                    {
                        var vc = (ViewCast)cur;
                        elemSize = VLTranslator.GetElemSize(vc.TargetElementType);
                        cur = vc.Value.Resolve();
                        if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); }
                        continue;
                    }
                    case ValueKind.AddressSpaceCast: cur = ((AddressSpaceCast)cur).Value.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.PointerCast: cur = ((PointerCast)cur).Value.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.IntAsPointerCast: cur = ((IntAsPointerCast)cur).Value.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.ArrayToViewCast: cur = ((ArrayToViewCast)cur).Value.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.Convert: cur = ((ConvertValue)cur).Value.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); } continue;
                    case ValueKind.AsAligned: cur = ((AsAligned)cur).Source.Resolve(); if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); sbTrace.Append("[asAligned]"); } continue;
                    case ValueKind.AlignTo:
                    {
                        var at = (AlignTo)cur;
                        if (at.IsPointerOperation)
                        {
                            // Record pointer alignment to apply once binding/base are known
                            pendingPtrAlignments ??= new System.Collections.Generic.List<uint>(1);
                            pendingPtrAlignments.Add(t.GetId(at.AlignmentInBytes.Resolve()));
                            cur = at.Source.Resolve();
                            if (sbTrace != null) { sbTrace.Append(" -> "); sbTrace.Append(cur.ValueKind); sbTrace.Append("[ptr-align]"); }
                            continue;
                        }
                        // For views, leave AlignTo to GetField handling
                        cur = at; // break out to base resolution
                        break;
                    }
                    case ValueKind.NewView:
                    case ValueKind.SubView:
                        // These should have been lowered by the view/pointer lowering passes
                        throw new NotImplementedException("Unexpected view operation in translator. Expected lowering to have removed NewView/SubView.");
                }
                break;
            }
            // Fold nested LEA offsets
            if (cur is LoadElementAddress lea)
            {
                if (!TryBuildPointerBytes(lea.Source.Resolve(), out var vId2, out var b2, out var baseBytes, out var es2))
                { if (sbTrace != null) VLTrace.Log(sbTrace.ToString()); varId = 0; binding = 0; elemSize = 4; return false; }
                var idx = t.GetId(lea.Offset.Resolve());
                var idxBytes = t.m.EmitMul(idx, t.m.EmitConstInt32(es2), false);
                byteOffset = t.m.EmitAdd(baseBytes, idxBytes, false);
                varId = vId2; binding = b2; elemSize = es2; return true;
            }
            // Base: derived view (e.g., AlignTo field)
            if (t.derivedViews.TryGetValue(cur, out var dv))
            {
                varId = dv.varId; binding = dv.binding; elemSize = dv.elemSize; byteOffset = dv.addBytesId;
                // Apply any pending pointer alignment now that binding/base are known
                if (pendingPtrAlignments is { Count: > 0 })
                {
                    var baseOffSlot = (uint)(t.viewBindingCount + (int)binding);
                    var baseOffS = t.m.EmitLoadPushConstant(baseOffSlot);
                    foreach (var alignId in pendingPtrAlignments)
                    {
                        var sum = t.m.EmitAdd(baseOffS, byteOffset, true);
                        var rem = t.m.EmitUMod(sum, alignId);
                        var diff = t.m.EmitSub(alignId, rem, true);
                        var delta = t.m.EmitUMod(diff, alignId);
                        byteOffset = t.m.EmitAdd(byteOffset, delta, true);
                    }
                }
                return true;
            }
            // Base: parameter-bound view
            if (cur is Parameter p)
            {
                if (t.viewParams.TryGetValue(p.Index, out var info))
                {
                    varId = info.varId; binding = info.binding; elemSize = info.elemSize;
                    // Apply any pending pointer alignment now that binding/base are known
                    if (pendingPtrAlignments is { Count: > 0 })
                    {
                        var baseOffSlot = (uint)(t.viewBindingCount + (int)binding);
                        var baseOffS = t.m.EmitLoadPushConstant(baseOffSlot);
                        foreach (var alignId in pendingPtrAlignments)
                        {
                            var sum = t.m.EmitAdd(baseOffS, byteOffset, true);
                            var rem = t.m.EmitUMod(sum, alignId);
                            var diff = t.m.EmitSub(alignId, rem, true);
                            var delta = t.m.EmitUMod(diff, alignId);
                            byteOffset = t.m.EmitAdd(byteOffset, delta, true);
                        }
                    }
                    return true;
                }
                if (sbTrace != null) VLTrace.Log(sbTrace.ToString() + $" | Param idx={p.Index} not mapped");
            }
            else
            {
                if (sbTrace != null) VLTrace.Log(sbTrace.ToString());
            }
            varId = 0; binding = 0; elemSize = 4; return false;
        }

        public void Visit(Parameter value)
        {
            if (t.pcParamWords.TryGetValue(value.Index, out var info))
                t.ids[value] = info.wordIds[0];
            else if (t.viewParams.TryGetValue(value.Index, out var infoView))
                t.ids[value] = infoView.varId;
            else
                throw new NotImplementedException("Unsupported parameter");
        }

        public void Visit(GridIndexValue value)
        { var id = t.m.EmitLoadGlobalIndex(value.Dimension); t.ids[value] = id; }

        public void Visit(GetViewLength value)
        {
            var vv = value.View.Resolve();
            // Derived views (e.g., from AlignTo fields) can provide an override length
            if (t.derivedViews.TryGetValue(vv, out var dv))
            {
                if (dv.overrideLenId != 0)
                {
                    var id = dv.overrideLenId;
                    if (value.Type is PrimitiveType pvt && pvt.BasicValueType == BasicValueType.Int64)
                        id = t.m.EmitConvertI32ToI64(id, false);
                    t.ids[value] = id; return;
                }
            }
            if (vv is SubViewValue sv)
            {
                var id = t.GetId(sv.Length.Resolve());
                if (value.Type is PrimitiveType p && p.BasicValueType == BasicValueType.Int64)
                    id = t.m.EmitConvertI32ToI64(id, false);
                t.ids[value] = id; return;
            }
            if (vv is NewView nv)
            {
                var id = t.GetId(nv.Length.Resolve());
                if (value.Type is PrimitiveType p && p.BasicValueType == BasicValueType.Int64)
                    id = t.m.EmitConvertI32ToI64(id, false);
                t.ids[value] = id; return;
            }
            if (t.TryResolveViewParam(vv, out var binding, out _))
            {
                var id = t.m.EmitLoadViewLength(binding);
                if (value.Type is PrimitiveType p && p.BasicValueType == BasicValueType.Int64)
                    id = t.m.EmitConvertI32ToI64(id, false);
                t.ids[value] = id; return;
            }
            throw new NotImplementedException("GetViewLength unsupported view");
        }

        public void Visit(LoadElementAddress value)
        {
            var src = value.Source.Resolve();
            if (!TryBuildPointerBytes(src, out var varId, out var binding, out var byteOff, out var es))
                throw new NotImplementedException("LEA expects lowered pointer to parameter-bound view");
            if (Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE") == "1")
            {
                var offRes = value.Offset.Resolve();
                if (offRes is PrimitiveValue pv && pv.BasicValueType == BasicValueType.Int32)
                    VLTrace.Log($"LEA: src={src.ValueKind} binding={binding} es={es} idxConst={pv.Int32Value}");
                else
                    VLTrace.Log($"LEA: src={src.ValueKind} binding={binding} es={es} idxKind={offRes.ValueKind}");
            }
            var baseOffSlot = (uint)(t.viewBindingCount + (int)binding);
            var baseOffS = t.m.EmitLoadPushConstant(baseOffSlot);
            // Include this LEA's index contribution using the LEA element size (pointer target type)
            int leaElemSize = value.Type is PointerType pt
                ? VLTranslator.GetElemSize(pt.ElementType)
                : es;
            var idxBytes = t.m.EmitMul(t.GetId(value.Offset.Resolve()), t.m.EmitConstInt32(leaElemSize), false);
            var totalBytes = t.m.EmitAdd(t.m.EmitAdd(baseOffS, byteOff, false), idxBytes, false);
            var totalU = t.m.EmitIntBitcast(totalBytes, true);
            int bes = t.bindingElemSize[binding];
            int sh = bes == 8 ? 3 : bes == 4 ? 2 : bes == 2 ? 1 : 0;
            var elemIndex = sh > 0
                ? t.m.EmitIntBitcast(t.m.EmitShiftRight(totalU, t.m.EmitIntBitcast(t.m.EmitConstInt32(sh), true), true), false)
                : t.m.EmitDiv(totalBytes, t.m.EmitConstInt32(bes), false);
            var ptr = t.m.AccessChainElement(varId, elemIndex, false, bes);
            t.ids[value] = ptr;
            // Cache info for subsequent Load on this LEA
            var cache = ((uint)binding, totalBytes, bes, varId);
            t.leaInfo[value] = cache;
            t.leaPtrInfo[ptr] = cache;
        }

        public void Visit(Load value)
        {
            // If source derives from a LEA (possibly wrapped in casts), recover LEA info
            var src = value.Source.Resolve();
            LoadElementAddress? leaSrc = null;
            Value cur = src;
            for (int i = 0; i < 8; i++) // bounded unwrap to avoid cycles
            {
                if (cur is LoadElementAddress lea) { leaSrc = lea; break; }
                switch (cur.ValueKind)
                {
                    case ValueKind.PointerCast: cur = ((PointerCast)cur).Value.Resolve(); continue;
                    case ValueKind.AddressSpaceCast: cur = ((AddressSpaceCast)cur).Value.Resolve(); continue;
                    case ValueKind.IntAsPointerCast: cur = ((IntAsPointerCast)cur).Value.Resolve(); continue;
                    case ValueKind.Convert: cur = ((ConvertValue)cur).Value.Resolve(); continue;
                    case ValueKind.LoadFieldAddress: cur = ((LoadFieldAddress)cur).Source.Resolve(); continue;
                }
                break;
            }

            // Prefer matching by pointer id since pointer casts collapse to underlying pointer id
            var ptrFromSrc = t.GetId(src);
            if (t.leaPtrInfo.TryGetValue(ptrFromSrc, out var info) || (leaSrc != null && t.leaInfo.TryGetValue(leaSrc, out info)))
            {
                var esDesired = (value.Type is PrimitiveType p && p.BasicValueType == BasicValueType.Int64) ? 8 : 4;
                var ptr = ptrFromSrc;
                if (Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE") == "1")
                    VLTrace.Log($"Load via LEA: binding={info.binding} bes={info.bindingElemSize} esDesired={esDesired}");
                if (esDesired == info.bindingElemSize)
                {
                    t.ids[value] = t.m.EmitLoadScalar(ptr, false, esDesired);
                    return;
                }
                // 64->32 split based on final byte offset remainder captured from LEA
                if (info.bindingElemSize == 8 && esDesired == 4)
                {
                    var v64 = t.m.EmitLoadScalar(ptr, false, 8);
                    var v64u = t.m.EmitBitcastI64ToU64(v64);
                    var rem = t.m.EmitUMod(info.totalBytes, t.m.EmitConstInt32(8));
                    if (Environment.GetEnvironmentVariable("ILGPU_VULKAN_TRACE") == "1") VLTrace.Log("Load split 64->32 using rem%8");
                    var isHigh = t.m.EmitCompareInt(rem, t.m.EmitConstInt32(4), CompareKind.Equal, true);
                    var hi64 = t.m.EmitShiftRight(v64u, t.m.EmitConstUInt64(32), true, 64);
                    var hi32 = t.m.EmitConvertU64ToI32(hi64);
                    var lo64 = t.m.EmitBitwiseAnd(v64u, t.m.EmitConstUInt64(0x00000000FFFFFFFFUL), 64);
                    var lo32 = t.m.EmitConvertU64ToI32(lo64);
                    t.ids[value] = t.m.EmitSelectInt(isHigh, hi32, lo32);
                    return;
                }
            }

            // Fallback: direct scalar load
            var ptrFallback = t.GetId(src);
            var esFallback = (value.Type is PrimitiveType p2 && p2.BasicValueType == BasicValueType.Int64) ? 8 : 4;
            t.ids[value] = t.m.EmitLoadScalar(ptrFallback, false, esFallback);
        }

        public void Visit(Store value)
        { t.m.EmitStore(t.GetId(value.Target.Resolve()), t.GetId(value.Value.Resolve())); }

        public void Visit(CompareValue value)
        { t.ids[value] = t.m.EmitCompareInt(t.GetId(value.Left.Resolve()), t.GetId(value.Right.Resolve()), value.Kind, (value.Flags & CompareFlags.UnsignedOrUnordered) != 0); }
        public void Visit(ReturnTerminator value) => t.m.EmitReturn();

        public void Visit(BinaryArithmeticValue value)
        {
            var a = t.GetId(value.Left.Resolve());
            var b = t.GetId(value.Right.Resolve());
            var u = value.IsUnsigned;
            int width = 32;
            if (value.Type is PrimitiveType pt)
            {
                width = pt.BasicValueType switch
                {
                    BasicValueType.Int32 => 32,
                    BasicValueType.Int64 => 64,
                    _ => 32,
                };
            }
            // Guard against using storage buffer variables as SSA values.
            if (TryGetStorageVarId(value.Left, out _) || TryGetStorageVarId(value.Right, out _))
                throw new NotImplementedException($"BinaryArithmetic on storage vars unsupported: kind={value.Kind} width={width}");
            uint id = value.Kind switch
            {
                BinaryArithmeticKind.Add => t.m.EmitAdd(a, b, u, width),
                BinaryArithmeticKind.Sub => t.m.EmitSub(a, b, u, width),
                BinaryArithmeticKind.Mul => t.m.EmitMul(a, b, u, width),
                BinaryArithmeticKind.Div => t.m.EmitDiv(a, b, u, width),
                BinaryArithmeticKind.Rem => t.m.EmitRem(a, b, u, width),
                BinaryArithmeticKind.Shl => t.m.EmitShiftLeft(a, b, width),
                BinaryArithmeticKind.Shr => t.m.EmitShiftRight(a, b, u, width),
                BinaryArithmeticKind.And => t.m.EmitBitwiseAnd(a, b, width),
                BinaryArithmeticKind.Or  => t.m.EmitBitwiseOr(a, b, width),
                BinaryArithmeticKind.Xor => t.m.EmitBitwiseXor(a, b, width),
                BinaryArithmeticKind.Min =>
                    // (a < b) ? a : b  (signed/unsigned per 'u')
                    t.m.EmitSelectWidth(
                        t.m.EmitCompareInt(a, b, CompareKind.LessThan, u),
                        a,
                        b,
                        width,
                        u),
                BinaryArithmeticKind.Max =>
                    t.m.EmitSelectWidth(
                        t.m.EmitCompareInt(a, b, CompareKind.GreaterThan, u),
                        a,
                        b,
                        width,
                        u),
                _ => throw new NotImplementedException()
            };
            t.ids[value] = id;
        }

        public void Visit(UnconditionalBranch value) => t.m.EmitBranch(value.Target);
        public void Visit(IfBranch value)
        {
            var merge = t.postDom!.GetImmediateCommonDominator(value.TrueTarget, value.FalseTarget);
            t.m.EmitSelectionMerge(merge);
            t.m.EmitBranchConditional(t.GetId(value.Condition.Resolve()), value.TrueTarget, value.FalseTarget);
        }

        public void Visit(PrimitiveValue value)
        {
            uint id = value.BasicValueType switch
            {
                BasicValueType.Int32 => t.m.EmitConstInt32(value.Int32Value),
                BasicValueType.Int64 => t.m.EmitConstInt64(value.Int64Value),
                BasicValueType.Int1  => t.m.EmitConstInt32(value.Int1Value ? 1 : 0),
                _ => throw new NotImplementedException()
            };
            t.ids[value] = id;
        }

        public void Visit(GetField value)
        {
            var obj = value.ObjectValue.Resolve();
            // AlignTo field selection: materialize derived views with adjusted base/length
            if (obj is AlignTo at)
            {
                if (!t.alignToInfo.TryGetValue(at, out var info))
                {
                    // Build info now if not yet present
                    if (!t.TryResolveViewParam(at.Source.Resolve(), out var bind, out var es))
                        throw new NotImplementedException("AlignTo on non-parameter view is unsupported");
                    var varId0 = t.viewParams[(at.Source.Resolve() as Parameter)!.Index].varId;
                    var baseOffSlot = (uint)(t.viewBindingCount + (int)bind);
                    var baseOffS = t.m.EmitLoadPushConstant(baseOffSlot);
                    var alignId = t.GetId(at.AlignmentInBytes.Resolve());
                    var rem = t.m.EmitUMod(baseOffS, alignId);
                    var diff = t.m.EmitSub(alignId, rem, true);
                    var deltaBytes = t.m.EmitUMod(diff, alignId);
                    var capElem = t.m.EmitDiv(deltaBytes, t.m.EmitConstInt32(es), true);
                    var lenId = t.m.EmitLoadViewLength(bind);
                    var mainLen = t.m.EmitSub(lenId, capElem, true);
                    var built = ((uint)bind, varId0, es, deltaBytes, capElem, mainLen);
                    t.alignToInfo[at] = built;
                    info = built;
                }
                int field = value.FieldSpan.Index;
                if (field == 0)
                {
                    // prefix: no base add; override length to capElem
                    t.derivedViews[value] = (info.binding, info.varId, info.elemSize, t.m.EmitConstInt32(0), info.capElemId);
                    t.ids[value] = info.varId; return;
                }
                if (field == 1)
                {
                    // main: base add deltaBytes; override length to mainLen
                    t.derivedViews[value] = (info.binding, info.varId, info.elemSize, info.deltaBytesId, info.mainLenId);
                    t.ids[value] = info.varId; return;
                }
                throw new NotImplementedException("AlignTo has only 2 fields");
            }
            // View struct fields: for parameter-bound views, map length to PC
            if (obj is Parameter p && t.viewParams.TryGetValue(p.Index, out var vinfo))
            {
                int field = value.FieldSpan.Index;
                if (field == 0)
                {
                    t.ids[value] = vinfo.varId; return;
                }
                if (field == 1 || field == 2)
                {
                    var len = t.m.EmitLoadViewLength(vinfo.binding);
                    t.ids[value] = len; return;
                }
            }
            // Kernel index parameter fields (Index1D/2D/3D): map to gl_GlobalInvocationID components
            if (obj is Parameter kp && !t.viewParams.ContainsKey(kp.Index))
            {
                var comp = value.FieldSpan.Index;
                var gid = t.m.EmitLoadGlobalIndex(comp switch { 1 => DeviceConstantDimension3D.Y, 2 => DeviceConstantDimension3D.Z, _ => DeviceConstantDimension3D.X });
                t.ids[value] = gid; return;
            }
            // Default: forward
            t.ids[value] = t.GetId(obj);
        }
        public void Visit(FloatAsIntCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(IntAsFloatCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(Predicate value) { t.ids[value] = t.GetId(value.Condition.Resolve()); }
        public void Visit(AtomicCAS value) => throw new NotImplementedException();
        public void Visit(UndefinedValue value) => throw new NotImplementedException();
        public void Visit(HandleValue value) => throw new NotImplementedException();

        // Unused/unsupported in this subset
        public void Visit(Alloca value) => throw new NotImplementedException();
        public void Visit(UnaryArithmeticValue value) => throw new NotImplementedException();
        public void Visit(TernaryArithmeticValue value) => throw new NotImplementedException();
        public void Visit(ConvertValue value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(IntAsPointerCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(PointerAsIntCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(PointerCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(AddressSpaceCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(ViewCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(ArrayToViewCast value) { t.ids[value] = t.GetId(value.Value.Resolve()); }
        public void Visit(StringValue value) { }
        public void Visit(MethodCall value) => throw new NotImplementedException();
        public void Visit(PhiValue value) => throw new NotImplementedException();
        public void Visit(GenericAtomic value) => throw new NotImplementedException();
        public void Visit(MemoryBarrier value) { }
        public void Visit(SubViewValue value) => throw new NotImplementedException();
        public void Visit(LoadArrayElementAddress value) => throw new NotImplementedException();
        public void Visit(LoadFieldAddress value) { t.ids[value] = t.GetId(value.Source.Resolve()); }
        public void Visit(NewView value) => throw new NotImplementedException();
        public void Visit(AlignTo value)
        {
            // Defer materialization to GetField; record nothing here.
            // Ensure source is a view we can resolve at field selection time.
            if (!value.IsViewOperation)
            {
                // Pointer AlignTo: treat as a no-op for now and forward the source
                t.ids[value] = t.GetId(value.Source.Resolve());
                return;
            }
        }
        public void Visit(AsAligned value) { t.ids[value] = t.GetId(value.Source.Resolve()); }
        public void Visit(NewArray value) => throw new NotImplementedException();
        public void Visit(GetArrayLength value) => throw new NotImplementedException();
        public void Visit(NullValue value) => throw new NotImplementedException();
        public void Visit(StructureValue value) => throw new NotImplementedException();
        public void Visit(SetField value) => throw new NotImplementedException();
        public void Visit(AcceleratorTypeValue value) => throw new NotImplementedException();
        public void Visit(GroupIndexValue value) => throw new NotImplementedException();
        public void Visit(GridDimensionValue value) => throw new NotImplementedException();
        public void Visit(GroupDimensionValue value) => throw new NotImplementedException();
        public void Visit(WarpSizeValue value) => throw new NotImplementedException();
        public void Visit(LaneIdxValue value) => throw new NotImplementedException();
        public void Visit(DynamicMemoryLengthValue value) => throw new NotImplementedException();
        public void Visit(PredicateBarrier value) => throw new NotImplementedException();
        public void Visit(Barrier value) => throw new NotImplementedException();
        public void Visit(Broadcast value) => throw new NotImplementedException();
        public void Visit(WarpShuffle value) => throw new NotImplementedException();
        public void Visit(SubWarpShuffle value) => throw new NotImplementedException();
        public void Visit(DebugAssertOperation value) { }
        public void Visit(WriteToOutput value) { }
        // duplicate handled above
        public void Visit(SwitchBranch value) => throw new NotImplementedException();
        public void Visit(LanguageEmitValue value) { }
    }
}
