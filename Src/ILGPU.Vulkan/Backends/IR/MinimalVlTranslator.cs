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
using ILGPU.IR.Analyses;
using ILGPU.IR.Analyses.ControlFlowDirection;
using ILGPU.IR.Types;
using System;
using System.Collections.Generic;
using ILGPU.Runtime;

namespace ILGPU.Backends.Vulkan;

internal sealed class MinimalVlTranslator
{
    private readonly SpirvModuleBuilder m;
    private readonly EntryPoint ep;
    private readonly Dictionary<Value, uint> idMap = [];
    private readonly Dictionary<int, (uint varId, bool unsigned, uint binding)> paramBinding = [];
    private readonly List<VLCompiledKernel.BindingInfo> bindings = [];
    // Element size per descriptor binding (bytes of the view's T)
    private readonly Dictionary<uint, int> bindingElemSize = new();
    private Dominators<Backwards>? postDom;
    private readonly Stack<BasicBlock> selectionMergeStack = new();

    private static bool IsAnyArrayViewType(Type t)
    {
        if (!t.IsGenericType) return false;
        var g = t.GetGenericTypeDefinition();
        return g == typeof(ArrayView<>) ||
               g == typeof(ArrayView1D<,>) ||
               g == typeof(ArrayView2D<,>) ||
               g == typeof(ArrayView3D<,>);
    }

    private static string DescribeValue(Value v, int depth = 0)
    {
        if (depth > 8) return "...";
        if (v is Parameter p) return $"Parameter[{p.Index}]";
        switch (v.ValueKind)
        {
            case ValueKind.GetField:
                return $"GetField({DescribeValue(((GetField)v).ObjectValue.Resolve(), depth+1)})";
            case ValueKind.LoadFieldAddress:
                return $"LoadFieldAddress({DescribeValue(((LoadFieldAddress)v).Source.Resolve(), depth+1)})";
            case ValueKind.NewView:
                return $"NewView.ptr({DescribeValue(((NewView)v).Pointer.Resolve(), depth+1)})";
            case ValueKind.AddressSpaceCast:
                return $"AddressSpaceCast({DescribeValue(((AddressSpaceCast)v).Value.Resolve(), depth+1)})";
            case ValueKind.ViewCast:
                return $"ViewCast({DescribeValue(((ViewCast)v).Value.Resolve(), depth+1)})";
            case ValueKind.SubView:
                return $"SubView.src({DescribeValue(((SubViewValue)v).Source.Resolve(), depth+1)})";
            case ValueKind.Structure:
                return "Structure";
            default:
                return v.ValueKind.ToString();
        }
    }

    public MinimalVlTranslator(SpirvModuleBuilder module, EntryPoint entryPoint)
    {
        m = module ?? throw new ArgumentNullException(nameof(module));
        ep = entryPoint ?? throw new ArgumentNullException(nameof(entryPoint));
    }

    public void TranslateKernel(in Backend.BackendContext ctx)
    {
        var method = ctx.KernelMethod;
        VLTrace.Log($"TranslateKernel: IR param count={method.NumParameters}");
        foreach (var p in method.Parameters)
        {
            VLTrace.Log($"IR Param idx={p.Index} type={p.ParameterType}");
        }
        postDom = method.Blocks.CreatePostDominators();
        // Predeclare block labels for CFG emission
        m.DeclareBlocks(method);
        uint binding = 0;
        // Build bindings based on entry-point parameter types; map to IR parameter indices
        for (int i = 0, e = ep.Parameters.Count; i < e; ++i)
        {
            var pt = ep.Parameters[i];
            if (!IsAnyArrayViewType(pt))
                continue;
            bool unsigned = false;
            var (_, varId) = m.DeclareStorageBuffer(binding, unsigned);
            int irParamIndex = ep.KernelIndexParameterOffset + i;
            paramBinding[irParamIndex] = (varId, unsigned, binding);
            bindings.Add(new VLCompiledKernel.BindingInfo(i, binding));
            bindingElemSize[binding] = GetElementSizeBytes(pt);
            binding++;
        }
        // Declare push constants for view lengths, in same order
        if (binding > 0)
            m.DeclarePushConstants(binding);

        foreach (var block in method.Blocks)
        {
            // Pop selection-merge scope if we reached the merge block
            if (selectionMergeStack.Count > 0 && ReferenceEquals(selectionMergeStack.Peek(), block))
                selectionMergeStack.Pop();
            m.BeginBlock(block);
            var visitor = new Visitor(this);
            foreach (var entry in block)
                ((Value)entry).Accept(visitor);
            // Visit the block terminator to emit structured control flow
            block.Terminator?.Accept(visitor);
        }
    }

    // Generic helpers for view lowering
    private static int GetElementSizeBytes(Type viewType)
    {
        // Extract element type argument from ArrayView<>, ArrayView{1D,2D,3D}
        Type? elem = null;
        if (viewType.IsGenericType)
        {
            var g = viewType.GetGenericTypeDefinition();
            var args = viewType.GetGenericArguments();
            if (g == typeof(ArrayView<>)) elem = args[0];
            else if (g == typeof(ArrayView1D<,>) || g == typeof(ArrayView2D<,>) || g == typeof(ArrayView3D<,>)) elem = args[0];
        }
        elem ??= typeof(int);
        if (elem == typeof(byte) || elem == typeof(sbyte)) return 1;
        if (elem == typeof(short) || elem == typeof(ushort)) return 2;
        if (elem == typeof(int) || elem == typeof(uint) || elem == typeof(float)) return 4;
        if (elem == typeof(long) || elem == typeof(ulong) || elem == typeof(double)) return 8;
        return 4;
    }

    private static int GetElementSizeBytes(ILGPU.IR.Types.TypeNode irType)
    {
        if (irType is ILGPU.IR.Types.PrimitiveType pt)
        {
            switch (pt.BasicValueType)
            {
                case BasicValueType.Int1: return 1; // treat as 1 byte for offsets
                case BasicValueType.Int8: return 1;
                case BasicValueType.Int16: return 2;
                case BasicValueType.Int32: return 4;
                case BasicValueType.Int64: return 8;
                case BasicValueType.Float32: return 4;
                case BasicValueType.Float64: return 8;
                default: return 4;
            }
        }
        return 4;
    }

    private uint MulByConst(uint valueId, int factor)
    {
        var c = m.EmitConstInt32(factor);
        return m.EmitMul(valueId, c, unsigned: false);
    }

    private struct ViewExpr
    {
        public uint VarId;
        public uint Binding;
        public uint ByteId;     // 32-bit signed byte offset
        public int ElemSize;    // element size in bytes
        public bool Unsigned;
    }

    // Build a byte-offset-based pointer for NewView pointers
    private bool TryBuildPtrExpr(Value v, out uint binding, out uint byteId)
    {
        // Reduce wrappers
        while (true)
        {
            switch (v.ValueKind)
            {
                case ValueKind.ViewCast:
                    v = ((ViewCast)v).Value.Resolve(); continue;
                case ValueKind.AddressSpaceCast:
                    v = ((AddressSpaceCast)v).Value.Resolve(); continue;
                case ValueKind.LoadFieldAddress:
                    v = ((LoadFieldAddress)v).Source.Resolve(); continue;
                case ValueKind.AlignTo:
                    v = ((AlignTo)v).Source.Resolve(); continue;
                case ValueKind.AsAligned:
                    v = ((AsAligned)v).Source.Resolve(); continue;
                case ValueKind.ArrayToViewCast:
                    v = ((ArrayToViewCast)v).Value.Resolve(); continue;
                case ValueKind.PointerCast:
                    v = ((PointerCast)v).Value.Resolve(); continue;
                case ValueKind.IntAsPointerCast:
                    v = ((IntAsPointerCast)v).Value.Resolve(); continue;
                default:
                    break;
            }
            break;
        }
        if (v is LoadElementAddress lea)
        {
            // pointer = lea(view, idx) -> byteOffset(view) + idx * sizeof(view.T)
            var viewVal = lea.Source.Resolve();
            if (!TryBuildViewExpr(viewVal, out var ve)) { binding = 0; byteId = 0; return false; }
            var idx = GetOrBuild(lea.Offset.Resolve());
            var idxBytes = MulByConst(idx, ve.ElemSize);
            binding = ve.Binding;
            byteId = m.EmitAdd(ve.ByteId, idxBytes, unsigned: false);
            return true;
        }
        binding = 0; byteId = 0; return false;
    }

    private bool TryBuildViewExpr(Value v, out ViewExpr expr)
    {
        // Parameter-backed view
        if (v is Parameter p)
        {
            if (paramBinding.TryGetValue(p.Index, out var pinf))
            {
                expr = new ViewExpr { VarId = pinf.varId, Binding = pinf.binding, ByteId = m.EmitConstInt32(0), ElemSize = bindingElemSize.TryGetValue(pinf.binding, out var s0) ? s0 : 4, Unsigned = pinf.unsigned };
                return true;
            }
            // Compute ordinal relative to KernelIndexParameterOffset and derive binding generically
            int j = p.Index - ep.KernelIndexParameterOffset;
            if (j >= 0 && j < ep.Parameters.Count && IsAnyArrayViewType(ep.Parameters[j]))
            {
                uint? bind = null;
                foreach (var b in bindings)
                {
                    if (b.ParamIndex == j) { bind = b.Binding; break; }
                }
                if (bind.HasValue)
                {
                    uint varId = 0; bool u = false;
                    foreach (var kv in paramBinding)
                    {
                        if (kv.Value.binding == bind.Value) { varId = kv.Value.varId; u = kv.Value.unsigned; break; }
                    }
                    if (varId != 0)
                    {
                        int es = bindingElemSize.TryGetValue(bind.Value, out var s1) ? s1 : 4;
                        expr = new ViewExpr { VarId = varId, Binding = bind.Value, ByteId = m.EmitConstInt32(0), ElemSize = es, Unsigned = u };
                        return true;
                    }
                }
            }
        }
        switch (v.ValueKind)
        {
            case ValueKind.GetField:
                return TryBuildViewExpr(((GetField)v).ObjectValue.Resolve(), out expr);
            case ValueKind.AlignTo:
                return TryBuildViewExpr(((AlignTo)v).Source.Resolve(), out expr);
            case ValueKind.AsAligned:
                return TryBuildViewExpr(((AsAligned)v).Source.Resolve(), out expr);
            case ValueKind.ArrayToViewCast:
                return TryBuildViewExpr(((ArrayToViewCast)v).Value.Resolve(), out expr);
            case ValueKind.PointerCast:
                return TryBuildViewExpr(((PointerCast)v).Value.Resolve(), out expr);
            case ValueKind.IntAsPointerCast:
                return TryBuildViewExpr(((IntAsPointerCast)v).Value.Resolve(), out expr);
            case ValueKind.Convert:
                return TryBuildViewExpr(((ConvertValue)v).Value.Resolve(), out expr);
            case ValueKind.Structure:
            {
                bool has = false; ViewExpr found = default;
                foreach (var n in v.Nodes)
                {
                    if (TryBuildViewExpr(n.Resolve(), out var inner))
                    {
                        if (!has) { has = true; found = inner; }
                        else if (found.Binding != inner.Binding)
                        { expr = default; return false; }
                    }
                }
                if (has) { expr = found; return true; }
                expr = default; return false;
            }
            case ValueKind.ViewCast:
            {
                var vc = (ViewCast)v;
                if (!TryBuildViewExpr(vc.Value.Resolve(), out expr)) return false;
                expr.ElemSize = GetElementSizeBytes(vc.TargetType);
                return true;
            }
            case ValueKind.AddressSpaceCast:
                return TryBuildViewExpr(((AddressSpaceCast)v).Value.Resolve(), out expr);
            case ValueKind.SubView:
            {
                var sv = (SubViewValue)v;
                if (!TryBuildViewExpr(sv.Source.Resolve(), out expr)) return false;
                var off = GetOrBuild(sv.Offset.Resolve());
                expr.ByteId = m.EmitAdd(expr.ByteId, off, unsigned: false);
                return true;
            }
            case ValueKind.NewView:
            {
                var nv = (NewView)v;
                if (!TryBuildPtrExpr(nv.Pointer.Resolve(), out var bind, out var baseBytes)) { expr = default; return false; }
                // Map binding -> varId
                uint varId = 0; bool uns = false;
                foreach (var kv in paramBinding)
                {
                    if (kv.Value.binding == bind) { varId = kv.Value.varId; uns = kv.Value.unsigned; break; }
                }
                if (varId == 0) { expr = default; return false; }
                expr = new ViewExpr { VarId = varId, Binding = bind, ByteId = baseBytes, ElemSize = GetElementSizeBytes(nv.Type), Unsigned = uns };
                return true;
            }
        }
        expr = default; return false;
    }

    private readonly struct Visitor : IValueVisitor
    {
        private readonly MinimalVlTranslator t;
        public Visitor(MinimalVlTranslator t) { this.t = t; }

        public void Visit(Alloca alloca) { var id = t.m.DeclareFunctionVariable(alloca.AllocaType); t.idMap[alloca] = id; }
        public void Visit(UnconditionalBranch value)
        {
            // If inside a selection, ensure arms end at the declared merge
            if (t.selectionMergeStack.Count > 0)
            {
                var merge = t.selectionMergeStack.Peek();
                t.m.EmitBranch(merge);
                return;
            }
            t.m.EmitBranch(value.Target);
        }
        public void Visit(IfBranch value)
        {
            var cond = t.GetOrBuild(value.Condition.Resolve());
            var merge = t.postDom!.GetImmediateCommonDominator(value.TrueTarget, value.FalseTarget);
            t.m.EmitSelectionMerge(merge);
            // Enter selection scope until we reach merge label
            t.selectionMergeStack.Push(merge);
            t.m.EmitBranchConditional(cond, value.TrueTarget, value.FalseTarget);
        }
        public void Visit(PhiValue value)
        {
            var srcs = value.Sources;
            var incoming = new (BasicBlock, uint)[srcs.Length];
            var phiPrim = value.PhiType as PrimitiveType;
            var phiBvt = phiPrim?.BasicValueType ?? BasicValueType.Int32;
            for (int i = 0; i < srcs.Length; i++)
            {
                var sv = value.GetValue(srcs[i])!.Resolve();
                var idv = t.GetOrBuild(sv);
                // Normalize incoming type to match phi result type
                if (phiBvt == BasicValueType.Int1)
                {
                    // Convert non-bool to bool via compare != 0
                    if (!(sv.Type.IsPrimitiveType && sv.Type.BasicValueType == BasicValueType.Int1))
                    {
                        var zero = t.m.EmitConstInt32(0);
                        idv = t.m.EmitCompareInt(idv, zero, CompareKind.NotEqual, false);
                    }
                }
                incoming[i] = (srcs[i], idv);
            }
            var id = t.m.EmitPhi(value.PhiType, incoming);
            t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(PhiValue), value.ToString());
            t.idMap[value] = id;
        }
        public void Visit(ReturnTerminator value) => t.m.EmitReturn();
        public void Visit(CompareValue value)
        {
            Ensure32Bit(value.Left.Resolve());
            Ensure32Bit(value.Right.Resolve());
            var l = t.GetOrBuild(value.Left.Resolve());
            var r = t.GetOrBuild(value.Right.Resolve());
            bool u = (value.Flags & CompareFlags.UnsignedOrUnordered) == CompareFlags.UnsignedOrUnordered;
            // Normalize both operands to the same signedness using bitcasts
            var lN = t.m.EmitIntBitcast(l, toUnsigned: u);
            var rN = t.m.EmitIntBitcast(r, toUnsigned: u);
            var id = t.m.EmitCompareInt(lN, rN, value.Kind, u);
            t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(CompareValue), value.ToString());
            t.idMap[value] = id;
        }
        
        private static void Ensure32Bit(Value v) { /* temporarily allow 64-bit */ }
        public void Visit(ConvertValue value)
        {
            var inner = value.Value.Resolve();
            // Ignore view shells (handled by LEA/GetViewLength)
            if (t.TryBuildViewExpr(inner, out _))
                return;

            var srcId = t.GetOrBuild(inner);
            var srcBvt = inner.Type.BasicValueType;
            var dstBvt = value.Type.BasicValueType;

            uint resId;
            if (dstBvt == BasicValueType.Int1)
            {
                // Normalize any integer to bool via (val != 0)
                uint zero;
                if (srcBvt == BasicValueType.Int64)
                    zero = t.m.EmitConstInt64(0);
                else
                    zero = t.m.EmitConstInt32(0);
                resId = t.m.EmitCompareInt(srcId, zero, CompareKind.NotEqual, unsigned: value.IsSourceUnsigned);
            }
            else if (srcBvt == BasicValueType.Int1 && (dstBvt == BasicValueType.Int32 || dstBvt == BasicValueType.Int64))
            {
                // Bool to integer: select(cond, 1, 0)
                uint one = dstBvt == BasicValueType.Int64 ? t.m.EmitConstInt64(1) : t.m.EmitConstInt32(1);
                uint zero = dstBvt == BasicValueType.Int64 ? t.m.EmitConstInt64(0) : t.m.EmitConstInt32(0);
                // EmitSelect uses integer type based on 'unsigned' flag; pick from TargetUnsigned
                resId = t.m.EmitSelect(srcId, one, zero, unsigned: value.IsResultUnsigned);
            }
            else
            {
                // Same width int conversions: bitcast between signed/unsigned
                resId = t.m.EmitIntBitcast(srcId, toUnsigned: value.IsResultUnsigned);
            }
            t.m.SetOrigin(resId, t.ep.MethodInfo.Name, nameof(ConvertValue), value.ToString());
            t.idMap[value] = resId;
        }
        public void Visit(GetField value){ var dim = value.FieldSpan.Index switch {1=>DeviceConstantDimension3D.Y,2=>DeviceConstantDimension3D.Z,_=>DeviceConstantDimension3D.X}; var id=t.m.EmitLoadGlobalIndex(dim); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GetField), value.ToString()); t.idMap[value]=id; }
        public void Visit(GridIndexValue value){ var id=t.m.EmitLoadGlobalIndex(value.Dimension); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GridIndexValue), value.ToString()); t.idMap[value]=id; }
        public void Visit(LoadElementAddress value)
        {
            var src = value.Source.Resolve();
            if (!t.TryBuildViewExpr(src, out var ve))
                throw new NotImplementedException($"LEA: cannot build view expr; chain={DescribeValue(src)}");
            // idxBytes = idx * elemSize
            var idx = t.GetOrBuild(value.Offset.Resolve());
            var idxBytes = t.MulByConst(idx, ve.ElemSize);
            var totalBytes = t.m.EmitAdd(ve.ByteId, idxBytes, unsigned: false);
            // To storage element index (4 bytes): >> 2
            var totalU = t.m.EmitIntBitcast(totalBytes, toUnsigned: true);
            var two = t.m.EmitConstInt32(2);
            var twoU = t.m.EmitIntBitcast(two, toUnsigned: true);
            var elemIndexU = t.m.EmitShiftRight(totalU, twoU, unsigned: true);
            var elemIndex = t.m.EmitIntBitcast(elemIndexU, toUnsigned: false);
            var id = t.m.AccessChainElement(ve.VarId, elemIndex, ve.Unsigned);
            t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(LoadElementAddress), value.ToString());
            t.idMap[value] = id;
        }
        public void Visit(Load value)
        {
            var src = value.Source.Resolve();
            // If this load is from a view-like expression, compute element index from bytes
            if (t.TryBuildViewExpr(src, out var ve))
            {
                var offU = t.m.EmitIntBitcast(ve.ByteId, toUnsigned: true);
                var sh2 = t.m.EmitConstInt32(2);
                var sh2U = t.m.EmitIntBitcast(sh2, toUnsigned: true);
                var elemIndexU = t.m.EmitShiftRight(offU, sh2U, unsigned: true);
                var elemIndex = t.m.EmitIntBitcast(elemIndexU, toUnsigned: false);
                var ptr = t.m.AccessChainElement(ve.VarId, elemIndex, ve.Unsigned);
                var id = t.m.EmitLoadScalar(ptr, ve.Unsigned);
                t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(Load), value.ToString());
                t.idMap[value] = id;
                return;
            }
            // Fallback: load via precomputed pointer id
            var ptrId = t.GetOrBuild(src);
            var valId = t.m.EmitLoadScalar(ptrId, unsigned: false);
            t.m.SetOrigin(valId, t.ep.MethodInfo.Name, nameof(Load), value.ToString());
            t.idMap[value] = valId;
        }
        public void Visit(GetViewLength value)
        {
            var vv = value.View.Resolve();
            // Generic path: compute base length from binding and subtract consumed elements from byte offset
            if (!t.TryBuildViewExpr(vv, out var ve))
                throw new NotImplementedException($"GetViewLength: cannot build view expr; chain={DescribeValue(vv)}");
            var baseLen = t.m.EmitLoadViewLength(ve.Binding);
            // subElems = byteOffset / elemSize (use shifts for power-of-two sizes)
            uint subElems = t.m.EmitConstInt32(0);
            if (ve.ByteId != t.m.EmitConstInt32(0))
            {
                int shift = ve.ElemSize switch { 8 => 3, 4 => 2, 2 => 1, 1 => 0, _ => 2 };
                if (shift > 0)
                {
                    var offU = t.m.EmitIntBitcast(ve.ByteId, toUnsigned: true);
                    var sh2 = t.m.EmitConstInt32(shift);
                    var sh2U = t.m.EmitIntBitcast(sh2, toUnsigned: true);
                    var elemsU = t.m.EmitShiftRight(offU, sh2U, unsigned: true);
                    subElems = t.m.EmitIntBitcast(elemsU, toUnsigned: false);
                }
            }
            var baseLenS = t.m.EmitIntBitcast(baseLen, toUnsigned: false);
            var len = t.m.EmitSub(baseLenS, subElems, unsigned: false);
            t.m.SetOrigin(len, t.ep.MethodInfo.Name, nameof(GetViewLength), value.ToString());
            t.idMap[value] = len;
        }
        public void Visit(BinaryArithmeticValue value)
        {
            var left = value.Left.Resolve();
            var right = value.Right.Resolve();
            // Handle boolean logical operations
            if (value.PrimitiveType.BasicValueType == BasicValueType.Int1)
            {
                var a = t.GetOrBuild(left);
                var b = t.GetOrBuild(right);
                uint id = value.Kind switch
                {
                    BinaryArithmeticKind.And => t.m.EmitLogicalAnd(a, b),
                    BinaryArithmeticKind.Or => t.m.EmitLogicalOr(a, b),
                    BinaryArithmeticKind.Xor => throw new NotImplementedException("Bool XOR not implemented"),
                    _ => throw new NotImplementedException($"Unsupported boolean op: {value.Kind}")
                };
                t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(BinaryArithmeticValue), value.ToString());
                t.idMap[value] = id;
                return;
            }

            // Integer 32-bit arithmetic
            Ensure32Bit(left); Ensure32Bit(right);
            var ua = value.IsUnsigned;
            var a32 = t.GetOrBuild(left);
            var b32 = t.GetOrBuild(right);
            uint rid = value.Kind switch
            {
                BinaryArithmeticKind.Add => t.m.EmitAdd(a32, b32, ua),
                BinaryArithmeticKind.Sub => t.m.EmitSub(a32, b32, ua),
                BinaryArithmeticKind.Mul => t.m.EmitMul(a32, b32, ua),
                BinaryArithmeticKind.And => t.m.EmitBitwiseAnd(a32, b32),
                BinaryArithmeticKind.Or  => t.m.EmitBitwiseOr(a32, b32),
                BinaryArithmeticKind.Xor => t.m.EmitBitwiseXor(a32, b32),
                _ => throw new NotImplementedException($"Unsupported int op: {value.Kind}")
            };
            t.m.SetOrigin(rid, t.ep.MethodInfo.Name, nameof(BinaryArithmeticValue), value.ToString());
            t.idMap[value] = rid;
        }
        public void Visit(Store value){ t.m.EmitStore(t.GetOrBuild(value.Target.Resolve()), t.GetOrBuild(value.Value.Resolve())); }
        public void Visit(PrimitiveValue value){ var id=t.GetOrBuild(value); t.idMap[value]=id; }
        public void Visit(StringValue value){ /* ignore */ }
        public void Visit(AddressSpaceCast value)
        {
            var inner = value.Value.Resolve();
            // Treat as a no-op for view-like values; handled by generic view lowering
            if (t.TryBuildViewExpr(inner, out _))
                return;
            var id = t.GetOrBuild(inner);
            t.idMap[value] = id;
        }
        public void Visit(ViewCast value)
        {
            var inner = value.Value.Resolve();
            // View casts are handled in TryBuildViewExpr when consumed
            if (t.TryBuildViewExpr(inner, out _))
                return;
            var id = t.GetOrBuild(inner);
            t.idMap[value] = id;
        }
        public void Visit(Parameter value) { }
        public void Visit(AtomicCAS value) { }
        public void Visit(GroupIndexValue value) { }
        public void Visit(GridDimensionValue value) { }
        public void Visit(GroupDimensionValue value) { }
        public void Visit(WarpSizeValue value) { }
        public void Visit(LaneIdxValue value) { }
        public void Visit(DynamicMemoryLengthValue value) { }
        public void Visit(PredicateBarrier value) { }
        public void Visit(Barrier value) { }
        public void Visit(Broadcast value) { }
        public void Visit(WarpShuffle value) { }
        public void Visit(SubWarpShuffle value) { }
        public void Visit(MethodCall value) { throw new NotImplementedException("Method calls not supported"); }
        public void Visit(UnaryArithmeticValue value) { throw new NotImplementedException(); }
        public void Visit(TernaryArithmeticValue value) { throw new NotImplementedException(); }
        public void Visit(IntAsPointerCast value) { throw new NotImplementedException(); }
        public void Visit(PointerAsIntCast value) { throw new NotImplementedException(); }
        public void Visit(PointerCast value) { throw new NotImplementedException(); }
        public void Visit(ArrayToViewCast value) { throw new NotImplementedException(); }
        public void Visit(FloatAsIntCast value) { throw new NotImplementedException(); }
        public void Visit(IntAsFloatCast value) { throw new NotImplementedException(); }
        public void Visit(Predicate value) { throw new NotImplementedException(); }
        public void Visit(GenericAtomic value) { throw new NotImplementedException(); }
        public void Visit(MemoryBarrier value) { }
        public void Visit(SubViewValue value) { /* shell view: handled by LEA/GetViewLength */ }
        public void Visit(LoadArrayElementAddress value) { throw new NotImplementedException(); }
        public void Visit(LoadFieldAddress value) { throw new NotImplementedException(); }
        public void Visit(NewView value) { }
        public void Visit(AlignTo value) { throw new NotImplementedException(); }
        public void Visit(AsAligned value) { throw new NotImplementedException(); }
        public void Visit(NewArray value) { throw new NotImplementedException(); }
        public void Visit(GetArrayLength value) { throw new NotImplementedException(); }
        public void Visit(NullValue value) { }
        public void Visit(StructureValue value) { throw new NotImplementedException(); }
        public void Visit(SetField value) { throw new NotImplementedException(); }
        public void Visit(AcceleratorTypeValue value) { throw new NotImplementedException(); }
        public void Visit(UndefinedValue value) { throw new NotImplementedException(); }
        public void Visit(HandleValue value) { throw new NotImplementedException(); }
        public void Visit(DebugAssertOperation value) { }
        public void Visit(WriteToOutput value) { }
        public void Visit(SwitchBranch value) { throw new NotImplementedException(); }
        public void Visit(LanguageEmitValue value) { }
    }
    private bool TryResolveViewBinding(Value v, out (uint varId, bool unsigned, uint binding) info)
    {
        VLTrace.Log($"TryResolveViewBinding: start kind={v.ValueKind}; v={v}");
        if (v is Parameter p && paramBinding.TryGetValue(p.Index, out var pinf))
        {
            VLTrace.Log($"TryResolveViewBinding: matched Parameter index={p.Index} to binding={pinf.binding}");
            info = pinf; return true;
        }
        switch (v.ValueKind)
        {
            case ValueKind.GetField:
                VLTrace.Log("TryResolveViewBinding: unwrap GetField.ObjectValue");
                return TryResolveViewBinding(((GetField)v).ObjectValue.Resolve(), out info);
            case ValueKind.LoadFieldAddress:
                VLTrace.Log("TryResolveViewBinding: unwrap LoadFieldAddress.Source");
                return TryResolveViewBinding(((LoadFieldAddress)v).Source.Resolve(), out info);
            case ValueKind.AlignTo:
                VLTrace.Log("TryResolveViewBinding: unwrap AlignTo.Source");
                return TryResolveViewBinding(((AlignTo)v).Source.Resolve(), out info);
            case ValueKind.AsAligned:
                VLTrace.Log("TryResolveViewBinding: unwrap AsAligned.Source");
                return TryResolveViewBinding(((AsAligned)v).Source.Resolve(), out info);
            case ValueKind.NewView:
                VLTrace.Log("TryResolveViewBinding: unwrap NewView.Pointer");
                return TryResolveViewBinding(((NewView)v).Pointer.Resolve(), out info);
            case ValueKind.AddressSpaceCast:
                VLTrace.Log("TryResolveViewBinding: unwrap AddressSpaceCast.Value");
                return TryResolveViewBinding(((AddressSpaceCast)v).Value.Resolve(), out info);
            case ValueKind.ArrayToViewCast:
                VLTrace.Log("TryResolveViewBinding: unwrap ArrayToViewCast.Value");
                return TryResolveViewBinding(((ArrayToViewCast)v).Value.Resolve(), out info);
            case ValueKind.ViewCast:
                VLTrace.Log("TryResolveViewBinding: unwrap ViewCast.Value");
                return TryResolveViewBinding(((ViewCast)v).Value.Resolve(), out info);
            case ValueKind.SubView:
                VLTrace.Log("TryResolveViewBinding: unwrap SubView.Source");
                return TryResolveViewBinding(((SubViewValue)v).Source.Resolve(), out info);
            case ValueKind.Structure:
            {
                (uint varId, bool unsigned, uint binding) found = default;
                bool has = false;
                foreach (var node in v.Nodes)
                {
                    var inner = node.Resolve();
                    VLTrace.Log($"TryResolveViewBinding: scan Structure node kind={inner.ValueKind}; node={inner}");
                    if (TryResolveViewBinding(inner, out var sub))
                    {
                        if (!has)
                        {
                            found = sub; has = true;
                        }
                        else if (found.binding != sub.binding)
                        {
                            info = default; return false;
                        }
                    }
                }
                if (has) { info = found; return true; }
                info = default; VLTrace.Log("TryResolveViewBinding: Structure had no bound sub-nodes"); return false;
            }
        }
        info = default; VLTrace.Log($"TryResolveViewBinding: failed for kind={v.ValueKind}"); return false;
    }
    private uint GetOrBuild(Value v)
    {
        if (idMap.TryGetValue(v, out var id))
            return id;
        switch (v.ValueKind)
        {
            case ValueKind.AddressSpaceCast:
            {
                var asc = (AddressSpaceCast)v;
                return GetOrBuild(asc.Value.Resolve());
            }
            case ValueKind.ViewCast:
            {
                var vc = (ViewCast)v;
                return GetOrBuild(vc.Value.Resolve());
            }
            case ValueKind.GetViewLength:
            {
                var gl = (GetViewLength)v;
                var viewVal = gl.View.Resolve();
                VLTrace.Log($"GOB GetViewLength for {viewVal}");
                // Shell-aware: direct IR-provided lengths
                if (viewVal is SubViewValue sv)
                {
                    var lenId = GetOrBuild(sv.Length.Resolve());
                    idMap[v] = lenId;
                    return lenId;
                }
                if (viewVal is NewView nv)
                {
                    var lenId = GetOrBuild(nv.Length.Resolve());
                    idMap[v] = lenId;
                    return lenId;
                }
                if (!TryResolveViewBinding(viewVal, out var info))
                {
                    VLTrace.Log("GOB GetViewLength failed to resolve view binding");
                    throw new NotImplementedException("View length for non-bound parameter");
                }
                VLTrace.Log($"GOB GetViewLength resolved binding {info.binding}");
                var pcLenId = m.EmitLoadViewLength(info.binding);
                idMap[v] = pcLenId;
                return pcLenId;
            }
            case ValueKind.Convert:
            {
                var conv = (ConvertValue)v;
                var src = GetOrBuild(conv.Value.Resolve());
                var rid = m.EmitIntBitcast(src, conv.IsResultUnsigned);
                idMap[v] = rid;
                return rid;
            }
            case ValueKind.Compare:
            {
                var cmp = (CompareValue)v;
                var left = GetOrBuild(cmp.Left.Resolve());
                var right = GetOrBuild(cmp.Right.Resolve());
                var unsigned = (cmp.Flags & CompareFlags.UnsignedOrUnordered) == CompareFlags.UnsignedOrUnordered;
                var rid = m.EmitCompareInt(left, right, cmp.Kind, unsigned);
                idMap[v] = rid;
                return rid;
            }
            case ValueKind.BinaryArithmetic:
            {
                var bin = (BinaryArithmeticValue)v;
                var a = GetOrBuild(bin.Left.Resolve());
                var b = GetOrBuild(bin.Right.Resolve());
                var rid = m.EmitAdd(a, b, false);
                idMap[v] = rid;
                return rid;
            }
            case ValueKind.LoadElementAddress:
            {
                var lea = (LoadElementAddress)v;
                var src = lea.Source.Resolve();
                if (!TryResolveViewBinding(src, out var info))
                    throw new NotImplementedException("LEA source not a supported view parameter");
                var idx = GetOrBuild(lea.Offset.Resolve());
                var ptr = m.AccessChainElement(info.varId, idx, info.unsigned);
                idMap[v] = ptr;
                return ptr;
            }
            case ValueKind.Load:
            {
                var load = (Load)v;
                var srcPtr = GetOrBuild(load.Source.Resolve());
                bool unsigned = false;
                foreach (var kv in paramBinding.Values)
                {
                    if (kv.varId == srcPtr) { unsigned = kv.unsigned; break; }
                }
                var val = m.EmitLoadScalar(srcPtr, unsigned);
                idMap[v] = val;
                return val;
            }
            case ValueKind.GetField:
            {
                var gf = (GetField)v;
                var objParam = gf.ObjectValue.ResolveAs<Parameter>();
                if (objParam != null && objParam.Index == 0)
                {
                    var dim = gf.FieldSpan.Index switch
                    {
                        1 => DeviceConstantDimension3D.Y,
                        2 => DeviceConstantDimension3D.Z,
                        _ => DeviceConstantDimension3D.X,
                    };
                    var gid = m.EmitLoadGlobalIndex(dim);
                    idMap[v] = gid;
                    return gid;
                }
                // Fallback: treat as grid-index-like access
                {
                    var dim = gf.FieldSpan.Index switch
                    {
                        1 => DeviceConstantDimension3D.Y,
                        2 => DeviceConstantDimension3D.Z,
                        _ => DeviceConstantDimension3D.X,
                    };
                    var gid = m.EmitLoadGlobalIndex(dim);
                    idMap[v] = gid;
                    return gid;
                }
            }
            case ValueKind.Alloca:
            {
                var alloca = (Alloca)v;
                var varId = m.DeclareFunctionVariable(alloca.AllocaType);
                idMap[v] = varId;
                return varId;
            }
            default:
                break;
        }
        if (v is PrimitiveValue pv)
        {
            if (pv.BasicValueType == BasicValueType.Int1)
            {
                var b = pv.Int32Value != 0;
                var boolConstId = m.EmitConstBool(b);
                idMap[v] = boolConstId;
                return boolConstId;
            }
            if (pv.BasicValueType == BasicValueType.Int64)
            {
                long lv = pv.Int64Value;
                if (lv < int.MinValue || lv > int.MaxValue)
                    throw new NotImplementedException("64-bit integer constants are not supported in Vulkan backend yet");
                var c32 = m.EmitConstInt32((int)lv);
                idMap[v] = c32;
                return c32;
            }
            // Default to 32-bit int for now
            var intConstId = m.EmitConstInt32((int)pv.Int32Value);
            idMap[v] = intConstId;
            return intConstId;
        }
        if (v is Parameter p)
        {
            if (p.Index == 0)
            {
                var gid = m.EmitLoadGlobalIndex(DeviceConstantDimension3D.X);
                idMap[v] = gid;
                return gid;
            }
            if (p.Index == ILGPU.Runtime.Kernel.KernelParamDimensionIdx)
            {
                // Compute global extent X = NumWorkgroups.x * LocalSize.x (64)
                var ng = m.EmitLoadNumWorkgroups(DeviceConstantDimension3D.X);
                var ls = m.EmitConstInt32(64);
                var lsu = m.EmitIntBitcast(ls, true);
                var total = m.EmitMulUint(ng, lsu);
                var totalS32 = m.EmitIntBitcast(total, false);
                idMap[v] = totalS32;
                return totalS32;
            }
        }
        // Fallback: try existing mapping or fail with detailed chain for diagnostics
        throw new NotImplementedException($"Unsupported value kind: {v}; chain={DescribeValue(v)}");
    }
    public VLCompiledKernel.BindingInfo[] GetBindings() => bindings.ToArray();
    public int GetPushConstantCount() => bindings.Count; // one int per view length
}
