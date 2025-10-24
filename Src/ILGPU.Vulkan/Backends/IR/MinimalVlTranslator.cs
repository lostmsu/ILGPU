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
    private Dominators<Backwards>? postDom;

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
            binding++;
        }
        // Declare push constants for view lengths, in same order
        if (binding > 0)
            m.DeclarePushConstants(binding);

        foreach (var block in method.Blocks)
        {
            m.BeginBlock(block);
            var visitor = new Visitor(this);
            foreach (var entry in block)
                ((Value)entry).Accept(visitor);
        }
    }

    private readonly struct Visitor : IValueVisitor
    {
        private readonly MinimalVlTranslator t;
        public Visitor(MinimalVlTranslator t) { this.t = t; }

        public void Visit(Alloca alloca) { var id = t.m.DeclareFunctionVariable(alloca.AllocaType); t.idMap[alloca] = id; }
        public void Visit(UnconditionalBranch value) => t.m.EmitBranch(value.Target);
        public void Visit(IfBranch value) { var cond = t.GetOrBuild(value.Condition.Resolve()); var merge = t.postDom!.GetImmediateCommonDominator(value.TrueTarget, value.FalseTarget); t.m.EmitSelectionMerge(merge); t.m.EmitBranchConditional(cond, value.TrueTarget, value.FalseTarget); }
        public void Visit(PhiValue value) { var srcs = value.Sources; var incoming = new (BasicBlock, uint)[srcs.Length]; for (int i=0;i<srcs.Length;i++){ var sv = value.GetValue(srcs[i])!.Resolve(); incoming[i]=(srcs[i], t.GetOrBuild(sv)); } var id = t.m.EmitPhi(value.PhiType, incoming); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(PhiValue), value.ToString()); t.idMap[value]=id; }
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
        
        private static void Ensure32Bit(Value v)
        {
            var pt = v.Type as ILGPU.IR.Types.PrimitiveType;
            if (pt != null && pt.BasicValueType == BasicValueType.Int64)
                throw new NotImplementedException("64-bit integers are not supported by Vulkan backend yet");
        }
        public void Visit(ConvertValue value){ Ensure32Bit(value.Value.Resolve()); var src=t.GetOrBuild(value.Value.Resolve()); var res=t.m.EmitIntBitcast(src, value.IsResultUnsigned); t.m.SetOrigin(res, t.ep.MethodInfo.Name, nameof(ConvertValue), value.ToString()); t.idMap[value]=res; }
        public void Visit(GetField value){ var dim = value.FieldSpan.Index switch {1=>DeviceConstantDimension3D.Y,2=>DeviceConstantDimension3D.Z,_=>DeviceConstantDimension3D.X}; var id=t.m.EmitLoadGlobalIndex(dim); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GetField), value.ToString()); t.idMap[value]=id; }
        public void Visit(GridIndexValue value){ var id=t.m.EmitLoadGlobalIndex(value.Dimension); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GridIndexValue), value.ToString()); t.idMap[value]=id; }
        public void Visit(LoadElementAddress value){ var src=value.Source.Resolve(); if(!t.TryResolveViewBinding(src,out var info)) throw new NotImplementedException("LEA source not a supported view parameter"); Ensure32Bit(value.Offset.Resolve()); var idx=t.GetOrBuild(value.Offset.Resolve()); var id=t.m.AccessChainElement(info.varId, idx, info.unsigned); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(LoadElementAddress), value.ToString()); t.idMap[value]=id; }
        public void Visit(Load value){ var ptr=t.GetOrBuild(value.Source.Resolve()); bool u=false; foreach(var kv in t.paramBinding.Values){ if(kv.varId==ptr){ u=kv.unsigned; break; } } var id=t.m.EmitLoadScalar(ptr,u); t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(Load), value.ToString()); t.idMap[value]=id; }
        public void Visit(GetViewLength value)
        {
            var vv = value.View.Resolve();
            if (vv is SubViewValue sv)
            {
                Ensure32Bit(sv.Length.Resolve());
                var len = t.GetOrBuild(sv.Length.Resolve());
                t.m.SetOrigin(len, t.ep.MethodInfo.Name, nameof(GetViewLength), value.ToString());
                t.idMap[value] = len;
                return;
            }
            if (vv is NewView nv)
            {
                Ensure32Bit(nv.Length.Resolve());
                var len = t.GetOrBuild(nv.Length.Resolve());
                t.m.SetOrigin(len, t.ep.MethodInfo.Name, nameof(GetViewLength), value.ToString());
                t.idMap[value] = len;
                return;
            }
            if (!t.TryResolveViewBinding(vv, out var info))
            {
                var chain = DescribeValue(vv);
                VLTrace.Log($"Visit(GetViewLength): non-bound view; chain={chain}");
                if (t.paramBinding.Count == 1)
                {
                    // Single-view kernel fallback: use the only binding's length
                    foreach (var kv in t.paramBinding.Values)
                    {
                        var id = t.m.EmitLoadViewLength(kv.binding);
                        t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GetViewLength), value.ToString());
                        t.idMap[value] = id;
                        return;
                    }
                }
                throw new NotImplementedException($"View length for non-bound parameter; chain={chain}");
            }
            {
                var id = t.m.EmitLoadViewLength(info.binding);
                t.m.SetOrigin(id, t.ep.MethodInfo.Name, nameof(GetViewLength), value.ToString());
                t.idMap[value] = id;
            }
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
        public void Visit(AddressSpaceCast value){ var id=t.GetOrBuild(value.Value.Resolve()); t.idMap[value]=id; }
        public void Visit(ViewCast value){ var id=t.GetOrBuild(value.Value.Resolve()); t.idMap[value]=id; }
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
        public void Visit(SubViewValue value) { throw new NotImplementedException(); }
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
            var cid = m.EmitConstInt32((int)pv.Int32Value);
            idMap[v] = cid;
            return cid;
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
        // Fallback: try existing mapping or fail
        throw new NotImplementedException($"Unsupported value kind: {v}");
    }
    public VLCompiledKernel.BindingInfo[] GetBindings() => bindings.ToArray();
}

