namespace ILGPU;

public readonly struct E8M0(byte @byte)
{
    public byte Byte => @byte;
}

public readonly struct BF16(ushort u16)
{
    public ushort U16 => u16;
}

public readonly struct MXFP4Block(
    E2M1x4 v0_7, E2M1x4 v8_15,
    E2M1x4 v16_23, E2M1x4 v24_31,
    E8M0 scale)
{
    public const int Elements = 8;

    public E2M1x4 V0_7 => v0_7;
    public E2M1x4 V8_15 => v8_15;
    public E2M1x4 V16_23 => v16_23;
    public E2M1x4 V24_31 => v24_31;

    public E8M0 Scale => scale;
}

public readonly struct E2M1x4(uint u32)
{
    public uint U32 => u32;
}
