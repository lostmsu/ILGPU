// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: SpirvWriter.cs
// ---------------------------------------------------------------------------------------

using Silk.NET.SPIRV;
using System;
using System.Buffers.Binary;
using System.Collections.Generic;

namespace ILGPU.Backends.Vulkan;

internal sealed class SpirvWriter
{
    private readonly List<uint> _words = [];
    private int _headerIndex = -1;

    public void Header(uint version, uint generator, uint bound, uint schema)
    {
        _headerIndex = _words.Count;
        _words.Add(0x07230203u); // Magic
        _words.Add(version);
        _words.Add(generator);
        _words.Add(bound);   // will patch later
        _words.Add(schema);
    }

    public void PatchBound(uint maxId)
    {
        if (_headerIndex < 0) throw new System.InvalidOperationException("Header not written");
        _words[_headerIndex + 3] = maxId + 1; // bound is max id + 1
    }

    public void Write(Op op, params uint[] operands)
    {
        uint wc = (uint)(1 + (operands?.Length ?? 0));
        _words.Add(((uint)op) | (wc << 16));
        if (operands != null)
            _words.AddRange(operands);
    }

    public void OpEntryPointCompute(uint funcId, string name, uint[] interfaceIds)
    {
        var raw = new List<uint> { (uint)ExecutionModel.GLCompute, funcId };
        AppendString(raw, name);
        if (interfaceIds is { Length: > 0 }) raw.AddRange(interfaceIds);
        EmitRaw(Op.EntryPoint, raw);
    }

    public void OpExecutionModeLocalSize(uint funcId, uint x, uint y, uint z)
    {
        Write(Op.ExecutionMode, funcId, (uint)ExecutionMode.LocalSize, x, y, z);
    }

    public byte[] ToArray()
    {
        var bytes = new byte[_words.Count * 4];
        for (int i = 0; i < _words.Count; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4, 4), _words[i]);
        return bytes;
    }

    public uint[] ToUIntArray() => [.. _words];

    private void EmitRaw(Op op, List<uint> operands)
    {
        uint wc = (uint)(1 + operands.Count);
        _words.Add(((uint)op) | (wc << 16));
        _words.AddRange(operands);
    }

    private static void AppendString(List<uint> list, string s)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
        for (int i = 0; i < bytes.Length; i += 4)
        {
            uint w = 0;
            if (i + 0 < bytes.Length) w |= bytes[i + 0];
            if (i + 1 < bytes.Length) w |= (uint)bytes[i + 1] << 8;
            if (i + 2 < bytes.Length) w |= (uint)bytes[i + 2] << 16;
            if (i + 3 < bytes.Length) w |= (uint)bytes[i + 3] << 24;
            list.Add(w);
        }
    }
}
