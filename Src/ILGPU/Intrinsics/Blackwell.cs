using ILGPU.Runtime;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using static ILGPU.Intrinsics.Cuda.MatrixLayout;
using static ILGPU.Intrinsics.Cuda.PTX;
using static ILGPU.Intrinsics.Cuda.TensorRole;

namespace ILGPU.Intrinsics.Cuda;

public static class Blackwell
{
    public static bool IsSupported => PTX.IsSupported && PTX.Current.Major >= 10;
    public static bool IsHardwareAccelerated => PTX.IsHardwareAccelerated && PTX.Current.Major >= 10;

    public abstract class WGMMA : Hopper.WGMMA
    {
        public static bool IsSupported => Blackwell.IsSupported;

        [Flags]
        public enum Options : uint
        {
            None = 0,
            SaturateFinite = 1 << 0,
        }

        public static void MmaAsync<TShape, TA, TB, TC>(
            ref Tensor<Accumulator, TC, RowMajor> d,
            in Tensor<MatrixA, TA, RowMajor> a,
            in Tensor<MatrixB, TB, ColMajor> b,
            TShape shape = default,
            bool precondition = true,
            Options options = default)
            where TShape : struct, IWgmmaShape
            where TA : unmanaged
            where TB : unmanaged
            where TC : unmanaged => ThrowPlatformNotSupported();
    }

    public abstract class LdMatrix
    {
        public static bool IsSupported => Blackwell.IsSupported;

        public static Tensor<TRole, TElement, TLayout> Load_M8N8X4<TRole, TElement, TLayout>(
            ArrayView<TElement> sharedMem,
            int elementOffset,
            bool transpose = false)
            where TRole : TensorRole, new()
            where TElement : unmanaged
            where TLayout : MatrixLayout, new()
        {
            ThrowPlatformNotSupported();
            return default;
        }

        LdMatrix() { }
    }

    public abstract class TMA
    {
        public static bool IsSupported => Blackwell.IsSupported;

        public readonly struct Descriptor { }

        public readonly struct Barrier { }

        public static Descriptor CreateDescriptor<T>(ArrayView1D<T, Stride1D.Dense> globalView, int dim0, int dim1, int dim2, int dim3)
            where T : unmanaged
        {
            ThrowPlatformNotSupported();
            return default;
        }

        public static void Load2D<T>(in Descriptor descriptor, ArrayView<T> sharedMem, int coord0, int coord1, ref Barrier barrier)
            where T : unmanaged => ThrowPlatformNotSupported();

        public static void BarrierInit(ref Barrier barrier, int expectedTransactions) => ThrowPlatformNotSupported();

        public static void BarrierArrive(ref Barrier barrier, int transactionCount) => ThrowPlatformNotSupported();

        public static void BarrierWait(in Barrier barrier) => ThrowPlatformNotSupported();
    }

    public static class Dpx
    {
        public static bool IsSupported => Blackwell.IsSupported;

        public static void Add<T>(ref AccumulatorFragment<T> dst, in AccumulatorFragment<T> a, in AccumulatorFragment<T> b)
            where T : unmanaged => ThrowPlatformNotSupported();

        public static void ReluFma<T>(ref AccumulatorFragment<T> dst, in AccumulatorFragment<T> a, in AccumulatorFragment<T> b, in AccumulatorFragment<T> c)
            where T : unmanaged => ThrowPlatformNotSupported();
    }
}
