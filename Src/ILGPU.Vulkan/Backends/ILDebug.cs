// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ILDebug.cs
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;

namespace ILGPU.Backends.Vulkan;

/// <summary>
/// Minimal IL dumper/validator to help diagnose InvalidProgramException in
/// generated launcher IL. Controlled by env vars:
///  - ILGPU_VULKAN_IL_DUMP=1 writes IL to %TEMP%/ilgpu-vl-il-*.txt
///  - ILGPU_VULKAN_IL_VALIDATE=1 runs a basic stack validator and logs issues
/// </summary>
internal static class ILDebug
{
    private static readonly Dictionary<short, OpCode> OpCodeMap = BuildOpCodeMap();

    private static Dictionary<short, OpCode> BuildOpCodeMap()
    {
        var map = new Dictionary<short, OpCode>();
        foreach (var f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.GetValue(null) is OpCode op)
                map[op.Value] = op;
        }
        return map;
    }

    public static void DumpIL(MethodInfo method)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("ILGPU_VULKAN_IL_DUMP"), "1", StringComparison.Ordinal))
            return;

        var body = method.GetMethodBody();
        if (body == null)
            return;
        var il = body.GetILAsByteArray();
        var sw = new StringWriter();
        sw.WriteLine($"Method: {method.DeclaringType?.FullName}::{method.Name}");
        int pos = 0;
        while (pos < il.Length)
        {
            int start = pos;
            OpCode op;
            byte code = il[pos++];
            if (code == 0xFE)
            {
                short v = (short)(0xFE00 | il[pos++]);
                op = OpCodeMap.TryGetValue(v, out var o) ? o : default;
            }
            else
            {
                op = OpCodeMap.TryGetValue(code, out var o) ? o : default;
            }
            sw.Write($"{start:X4}: {op.Name}");
            // Read operand (display token/inline values; no deep resolve here)
            switch (op.OperandType)
            {
                case OperandType.InlineNone:
                    break;
                case OperandType.ShortInlineI:
                    sw.Write($" {il[pos++]}");
                    break;
                case OperandType.InlineI:
                    sw.Write($" {BitConverter.ToInt32(il, pos)}"); pos += 4; break;
                case OperandType.InlineI8:
                    sw.Write($" {BitConverter.ToInt64(il, pos)}"); pos += 8; break;
                case OperandType.ShortInlineBrTarget:
                    sw.Write($" {il[pos++]}"); break;
                case OperandType.InlineBrTarget:
                case OperandType.InlineField:
                case OperandType.InlineMethod:
                case OperandType.InlineSig:
                case OperandType.InlineString:
                case OperandType.InlineTok:
                case OperandType.InlineType:
                case OperandType.ShortInlineR:
                case OperandType.InlineR:
                    sw.Write($" 0x{BitConverter.ToInt32(il, pos):X8}"); pos += 4; break;
                case OperandType.ShortInlineVar:
                    sw.Write($" {il[pos++]}"); break;
                case OperandType.InlineVar:
                    sw.Write($" {BitConverter.ToUInt16(il, pos)}"); pos += 2; break;
                default:
                    break;
            }
            sw.WriteLine();
        }
        var path = Path.Combine(Path.GetTempPath(), $"ilgpu-vl-il-{DateTime.UtcNow:yyyyMMdd_HHmmss_fffffff}.txt");
        File.WriteAllText(path, sw.ToString());
        VLTrace.Log($"IL dump written: {path}");
    }

    public static void ValidateStack(MethodInfo method)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("ILGPU_VULKAN_IL_VALIDATE"), "1", StringComparison.Ordinal))
            return;
        var body = method.GetMethodBody();
        if (body == null)
            return;
        var il = body.GetILAsByteArray();
        int pos = 0;
        int depth = 0;
        try
        {
            while (pos < il.Length)
            {
                OpCode op;
                byte code = il[pos++];
                if (code == 0xFE)
                {
                    short v = (short)(0xFE00 | il[pos++]);
                    op = OpCodeMap.TryGetValue(v, out var o) ? o : default;
                }
                else
                {
                    op = OpCodeMap.TryGetValue(code, out var o) ? o : default;
                }
                // Stack effect for common ops we emit
                switch (op.Value)
                {
                    // ldc.i4
                    case 0x20: case 0x1F: case 0x1E: case 0x1D: case 0x1C: case 0x1B: case 0x1A: case 0x16: depth++; break;
                    // ldarg, ldloc
                    case 0x02: case 0x03: case 0x04: case 0x06: case 0x0E: case 0x0C: case 0x0D: depth++; break;
                    // stloc
                    case 0x0A: case 0x0B: case 0x0D + 0x100: // inline var variant
                        depth--; break;
                    // newarr: pop len, push ref
                    case 0x8D: depth = depth - 1 + 1; break;
                    // box, castclass: pop, push
                    case 0x8C: case 0x74: break;
                    // stelem.i4 / stelem.ref: pop arr,index,value
                    case 0x9A: case 0xA2: depth -= 3; break;
                    // call
                    case 0x28:
                    case 0x6F: // callvirt
                        {
                            int token = BitConverter.ToInt32(il, pos); pos += 4;
                            var m = (MethodBase)method.Module.ResolveMethod(token);
                            int pops = m.GetParameters().Length + (m.IsStatic ? 0 : 1);
                            int pushes = (m is MethodInfo mi && mi.ReturnType != typeof(void)) ? 1 : 0;
                            depth = depth - pops + pushes;
                            break;
                        }
                    default:
                        // Skip operand advances for unhandled ops
                        switch (op.OperandType)
                        {
                            case OperandType.InlineI8: pos += 8; break;
                            case OperandType.InlineI:
                            case OperandType.InlineBrTarget:
                            case OperandType.InlineField:
                            case OperandType.InlineMethod:
                            case OperandType.InlineSig:
                            case OperandType.InlineString:
                            case OperandType.InlineTok:
                            case OperandType.InlineType:
                            case OperandType.ShortInlineR:
                            case OperandType.InlineR:
                                pos += 4; break;
                            case OperandType.InlineVar: pos += 2; break;
                            case OperandType.ShortInlineI:
                            case OperandType.ShortInlineBrTarget:
                            case OperandType.ShortInlineVar:
                                pos += 1; break;
                        }
                        break;
                }
                if (depth < 0)
                    throw new InvalidOperationException($"IL stack underflow at {pos:X4} ({op.Name}) in {method.Name}");
            }
        }
        catch (Exception ex)
        {
            VLTrace.Log($"IL validate error in {method.Name}: {ex.Message}");
        }
    }

    public static void ValidateAndDump(MethodInfo method)
    {
        ValidateStack(method);
        DumpIL(method);
    }
}

