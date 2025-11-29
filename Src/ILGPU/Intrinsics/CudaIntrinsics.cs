using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics.CodeAnalysis;

namespace ILGPU.Intrinsics.Cuda
{
    /// <summary>
    /// Base PTX intrinsics information.
    /// </summary>
    public static partial class PTX
    {
        /// <summary>
        /// True when PTX intrinsics are available.
        /// </summary>
        public static bool IsSupported => false;
        /// <summary>
        /// Gets a value indicating whether hardware acceleration is available for the current environment.
        /// </summary>
        public static bool IsHardwareAccelerated => false;

        public static CudaArchitecture Current => new(0, 0);

        [DoesNotReturn]
        internal static void ThrowPlatformNotSupported()
        => throw new PlatformNotSupportedException("PTX intrinsics are only available when lowered by the CUDA backend.");
    }

    #region Fragments, shapes, layouts

    public interface IWgmmaShape { int M { get; } int N { get; } int K { get; } }

    public readonly struct M64N128K256 : IWgmmaShape { public int M => 64; public int N => 128; public int K => 256; }
    public readonly struct M128N128K256 : IWgmmaShape { public int M => 128; public int N => 128; public int K => 256; }
    public readonly struct M128N256K64 : IWgmmaShape { public int M => 128; public int N => 256; public int K => 64; }

    public abstract class MatrixLayout
    {
        internal MatrixLayout() { }
        public class RowMajor : MatrixLayout { }
        public class ColMajor : MatrixLayout { }
    }

    public abstract class TensorRole
    {
        internal TensorRole() { }

        public class MatrixA : TensorRole { }
        public class MatrixB : TensorRole { }
        public class Accumulator : TensorRole { }
    }

    public readonly struct Tensor<TRole, TElement, TLayout>
        where TRole : TensorRole, new()
        where TElement : unmanaged
        where TLayout : MatrixLayout, new()
    {
        public TElement Element { get; }
    }

    public readonly struct AccumulatorFragment<TElement>
        where TElement : unmanaged
    {
        // Opaque representation of an accumulator fragment.
    }

    #endregion
}
