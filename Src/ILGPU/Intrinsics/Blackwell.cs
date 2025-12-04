using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
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

    public unsafe abstract class TMA
    {
        public static bool IsSupported => Blackwell.IsSupported;

        public unsafe struct TensorMap
        {
            fixed ulong opaque[16];


            public enum DataType
            {
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_UINT8
                /// </summary>
                U8 = 0,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_UINT16
                /// </summary>
                U16 = 1,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_UINT32
                /// </summary>
                U32 = 2,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_INT32
                /// </summary>
                I32 = 3,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_UINT64
                /// </summary>
                U64 = 4,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_INT64
                /// </summary>
                I64 = 5,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                /// </summary>
                F16 = 6,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                /// </summary>
                F32 = 7,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_FLOAT64
                /// </summary>
                F64 = 8,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                /// </summary>
                BF16 = 9,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ
                /// </summary>
                F32_FTZ = 10,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_TFLOAT32
                /// </summary>
                TF32 = 11,
                /// <summary>
                /// CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ
                /// </summary>
                TF32_FTZ = 12,
                /// <summary>
                /// '16 x U4' packed values to memory aligned as 8 bytes. There are no gaps between packed values.
                /// CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
                /// </summary>
                X16U4_Align8B,    // 4 bits
                /// <summary>
                /// '16 x U4' packed values to memory aligned as 16 bytes. There are 8 byte gaps between every 8 byte chunk of packed values.
                /// CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                /// </summary>
                X16U4_Align16B,   // 4 bits
                /// <summary>
                /// '16 x U6' packed values to memory aligned as 16 bytes. There are 4 byte gaps between every 12 byte chunk of packed values.
                /// CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
                /// </summary>
                X16U6_Align16B,   // 6 bits
            }

            public enum Interleave
            {
                /// <summary>
                /// CU_TENSOR_MAP_INTERLEAVE_NONE
                /// </summary>
                NONE = 0,
                /// <summary>
                /// CU_TENSOR_MAP_INTERLEAVE_16B
                /// </summary>
                By16B = 1,
                /// <summary>
                /// CU_TENSOR_MAP_INTERLEAVE_32B
                /// </summary>
                By32B = 2,
            }

            /// <summary>
            /// L2 fetch size which indicates the
            /// byte granularity at which requests are filled from DRAM
            /// </summary>
            public enum L2Promotion
            {
                /// <summary>
                /// CU_TENSOR_MAP_L2_PROMOTION_NONE
                /// </summary>
                NONE = 0,
                /// <summary>
                /// CU_TENSOR_MAP_L2_PROMOTION_64B
                /// </summary>
                L2_64B,
                /// <summary>
                /// CU_TENSOR_MAP_L2_PROMOTION_128B
                /// </summary>
                L2_128B,
                /// <summary>
                /// CU_TENSOR_MAP_L2_PROMOTION_256B
                /// </summary>
                L2_256B,
            }

            /// <summary>
            /// Shared memory bank swizzling pattern.
            /// </summary>
            /// <remarks>
            /// Data are organized in a specific order in global memory;
            /// however, this may not match the order in which the application
            /// accesses data in shared memory.
            /// This difference in data organization may cause bank conflicts
            /// when shared memory is accessed. In order to avoid this problem, data can be loaded
            /// to shared memory with shuffling across shared memory banks.
            /// </remarks>
            public enum Swizzle
            {
                /// <summary>
                /// CU_TENSOR_MAP_SWIZZLE_NONE
                /// </summary>
                NONE = 0,
                /// <summary>
                /// CU_TENSOR_MAP_SWIZZLE_32B
                /// </summary>
                Span32B_Chunk16B = 1,
                /// <summary>
                /// CU_TENSOR_MAP_SWIZZLE_64B
                /// </summary>
                Span64B_Chunk16B = 2,
                /// <summary>
                /// CU_TENSOR_MAP_SWIZZLE_128B
                /// </summary>
                Span128B_Chunk16B = 3,
                /// <summary>
                /// Swizzle 32B chunks within 128B span
                /// CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
                /// </summary>
                Span128B_Chunk32B,
                /// <summary>
                /// Swizzle 32B chunks within 128B span, additionally swap lower 8B with upper 8B within each 16B for every alternate row
                /// CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
                /// </summary>
                Span128B_Chunk32B_Flip8B,
                /// <summary>
                /// Swizzle 64B chunks within 128B span
                /// CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
                /// </summary>
                Span128B_Chunk64B,
            }

            /// <summary>
            /// The value to use to fill out-of-bounds accesses
            /// </summary>
            public enum FloatOOBfill
            {
                /// <summary>
                /// CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                /// </summary>
                Zero = 0,
                /// <summary>
                /// CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                /// </summary>
                NaN,
            }

            public struct TensorRank
            {
                public TensorRank(uint rank)
                {
                    ArgumentOutOfRangeException.ThrowIfZero(rank);
                    // limitation as of CUDA 13
                    ArgumentOutOfRangeException.ThrowIfGreaterThan(rank, 5u);

                    Rank = rank;
                }
                public uint Rank { get; }
            }

            public struct GlobalAddress
            {
                public nint Address { get; }

                public GlobalAddress(nint address)
                {
                    if ((address & 0xF) != 0)
                        throw new ArgumentException("Global address must be 16-byte aligned", nameof(address));

                    Address = address;
                }
            }
        }

        // CUresult cuTensorMapEncodeTiled ( CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
        // Create a tensor map descriptor object representing tiled memory region.
        static delegate*<out TensorMap, TensorMap.DataType, TensorMap.TensorRank,
            // void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
            TensorMap.GlobalAddress, ulong*, ulong*,
            // const cuuint32_t* boxDim, const cuuint32_t* elementStrides,
            uint*, uint*,
            TensorMap.Interleave, TensorMap.Swizzle, TensorMap.L2Promotion, TensorMap.FloatOOBfill,
            CudaError> cuTensorMapEncodeTiled;

        static CudaError EncodeTiled(out TensorMap map,
            TensorMap.DataType dataType,
            TensorMap.TensorRank rank,
            TensorMap.GlobalAddress globalAddress,
            ulong* globalDim,
            ulong* globalStrides,
            uint* boxDim,
            uint* elementStrides,
            TensorMap.Interleave interleave,
            TensorMap.Swizzle swizzle,
            TensorMap.L2Promotion l2Promotion,
            TensorMap.FloatOOBfill floatOOBfill)
        {
            if (cuTensorMapEncodeTiled == null)
            {
                ThrowPlatformNotSupported();
            }
            ArgumentNullException.ThrowIfNull(globalDim);
            ArgumentNullException.ThrowIfNull(globalStrides);
            ArgumentNullException.ThrowIfNull(boxDim);
            ArgumentNullException.ThrowIfNull(elementStrides);

            if (interleave != TensorMap.Interleave.NONE)
            {
                if (rank.Rank < 3)
                    throw new ArgumentOutOfRangeException(nameof(rank), "Interleaving requires rank of at least 3.");
            }

            int elementBytes = Bytes(dataType);

            bool requiresAlign32 = interleave is TensorMap.Interleave.By32B
                || dataType is TensorMap.DataType.X16U6_Align16B
                            or TensorMap.DataType.X16U4_Align16B;
            if (requiresAlign32 && (globalAddress.Address & 0x1F) != 0)
                throw new ArgumentOutOfRangeException(nameof(globalAddress), "Global address must be 32-byte aligned");

            for (int dim = 0; dim < rank.Rank; dim++)
            {
                string dimName = $"{nameof(globalDim)}[{dim}]";
                if (globalDim[dim] == 0)
                    throw new ArgumentOutOfRangeException(dimName, "Global dimension must be non-zero.");
                if (globalDim[dim] > 0x1_0000_0000ul)
                    throw new ArgumentOutOfRangeException(dimName, "Global dimension exceeds maximum supported size.");

                string strideName = $"{nameof(globalStrides)}[{dim}]";
                if (dim < rank.Rank - 1)
                {
                    if ((globalStrides[dim] & 0xF) != 0)
                        throw new ArgumentOutOfRangeException(strideName, $"{strideName} must be a multiple of 16.");
                    if (globalStrides[dim] >= 1ul << 40)
                        throw new ArgumentOutOfRangeException(strideName, $"{strideName} exceeds maximum supported size.");
                    if (requiresAlign32 && (globalStrides[dim] & 0x1F) != 0)
                        throw new ArgumentOutOfRangeException(strideName, $"{strideName} must be a multiple of 32.");
                }

                string boxDimName = $"{nameof(boxDim)}[{dim}]";
                ArgumentOutOfRangeException.ThrowIfZero(boxDim[dim], boxDimName);
                ArgumentOutOfRangeException.ThrowIfGreaterThan(boxDim[dim], 256u, boxDimName);

                if (dim == 0)
                {
                    bool requiresMultiply128 = dataType is TensorMap.DataType.X16U6_Align16B
                        or TensorMap.DataType.X16U4_Align16B;
                    if (requiresMultiply128 && (globalDim[dim] & 0x7F) != 0)
                    {
                        throw new ArgumentOutOfRangeException(dimName, $"{dimName} must be a multiple of 128.");
                    }

                    bool requiresMultiply2 = dataType is TensorMap.DataType.X16U4_Align8B;
                    if (requiresMultiply2 && (globalDim[dim] & 0x1) != 0)
                    {
                        throw new ArgumentOutOfRangeException(dimName, $"{dimName} must be a multiple of 2.");
                    }

                    if (interleave == TensorMap.Interleave.NONE
                        && (boxDim[dim] * elementBytes) % 16 != 0)
                    {
                        throw new ArgumentOutOfRangeException(boxDimName, $"{boxDimName} must be a multiple of 16.");
                    }

                    if (requiresMultiply128 && boxDim[dim] != 128)
                    {
                        throw new ArgumentOutOfRangeException(boxDimName, $"{boxDimName} must be 128.");
                    }
                }
            }

            return cuTensorMapEncodeTiled(
                out map,
                dataType,
                rank,
                globalAddress,
                globalDim,
                globalStrides,
                boxDim,
                elementStrides,
                interleave,
                swizzle,
                l2Promotion,
                floatOOBfill);
        }

        static int Bytes(TensorMap.DataType type) => type switch
        {
            TensorMap.DataType.BF16
                or TensorMap.DataType.F16 => 2,
            TensorMap.DataType.F32 => 4,
            TensorMap.DataType.F64
                or TensorMap.DataType.I64
                or TensorMap.DataType.U64 => 8,
            _ => throw new NotImplementedException(type.ToString()),
        };

        public readonly struct Barrier { }

        public static TensorMap CreateDescriptor<T>(ArrayView1D<T, Stride1D.Dense> globalView, int dim0, int dim1, int dim2, int dim3)
            where T : unmanaged
        {
            ThrowPlatformNotSupported();
            return default;
        }

        /// <summary>
        /// Represents wm
        /// </summary>
        public static void Load2D<T>(in TensorMap descriptor, ArrayView<T> sharedMem, int coord0, int coord1, ref Barrier barrier)
            where T : unmanaged => ThrowPlatformNotSupported();

        public static void BarrierInit(ref Barrier barrier, int expectedTransactions) => ThrowPlatformNotSupported();

        public static void BarrierArrive(ref Barrier barrier, int transactionCount) => ThrowPlatformNotSupported();

        public static void BarrierWait(in Barrier barrier) => ThrowPlatformNotSupported();

        static TMA()
        {
            if (CudaAPI.CurrentAPI.GetProcAddress(
                    nameof(cuTensorMapEncodeTiled),
                    cudaVersion: 12000,
                    DriverProcAddressFlags.CU_GET_PROC_ADDRESS_DEFAULT,
                    out nint procAddress) is CudaError.CUDA_SUCCESS)
            {
                cuTensorMapEncodeTiled = (delegate*<out TensorMap, TensorMap.DataType, TensorMap.TensorRank,
                    TensorMap.GlobalAddress, ulong*, ulong*,
                    uint*, uint*,
                    TensorMap.Interleave, TensorMap.Swizzle, TensorMap.L2Promotion, TensorMap.FloatOOBfill,
                    CudaError>)procAddress;
            }
        }
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
