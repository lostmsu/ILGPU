// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: VLValidateUnsupportedTransformation.cs
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Transformations;
using Method = ILGPU.IR.Method;
using ILGPU.IR.Values;
using ILGPU.IR.Types;
using System;
using System.Collections.Generic;
using System.Text;

namespace ILGPU.Backends.Vulkan.Transformations;

/// <summary>
/// Vulkan backend validation pass that checks for IR constructs not yet supported
/// by the SPIR-V/Vulkan codegen and fails with a combined message per method.
///
/// Intended to run late in the lowering pipeline (post view/pointer lowering).
/// </summary>
public sealed class VLValidateUnsupportedTransformation : UnorderedTransformation
{
    protected override bool PerformTransformation(Method.Builder builder)
    {
        var issues = new List<string>();

        foreach (var block in builder.Method.Blocks)
        {
            foreach (var entry in block)
            {
                var value = entry.Value;
                switch (value.ValueKind)
                {
                    case ILGPU.IR.ValueKind.ViewCast:
                        {
                            var vc = (ViewCast)value;
                            int srcSize = GetElementSize(vc.SourceElementType);
                            int dstSize = GetElementSize(vc.TargetElementType);
                            if (srcSize != dstSize)
                                issues.Add($"ViewCast {srcSize}->{dstSize}: {value}");
                            break;
                        }
                    case ILGPU.IR.ValueKind.NewView:
                        issues.Add($"NewView: {value}");
                        break;
                    case ILGPU.IR.ValueKind.SubView:
                        issues.Add($"SubView: {value}");
                        break;
                    default:
                        break;
                }
            }
        }

        if (issues.Count > 0)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Vulkan backend does not support the following IR in method '{builder.Method.Name}':");
            foreach (var i in issues)
                sb.AppendLine(" - " + i);
            sb.AppendLine("These should be lowered away (e.g., reinterpretation or subviews), or handled via value-level paths.");
            throw new PlatformNotSupportedException(sb.ToString());
        }

        // No changes performed
        return false;
    }

    private static int GetElementSize(TypeNode t) => t switch
    {
        PrimitiveType pt => pt.BasicValueType switch
        {
            BasicValueType.Int64 or BasicValueType.Float64 => 8,
            BasicValueType.Int32 or BasicValueType.Float32 => 4,
            BasicValueType.Int16 => 2,
            BasicValueType.Int8 or BasicValueType.Int1 => 1,
            _ => 4,
        },
        _ => 4,
    };
}
