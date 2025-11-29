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
                CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
                CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
                CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
                CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
                CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
                CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
                CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
                CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
            }

            public enum Interleave
            {
                CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
                CU_TENSOR_MAP_INTERLEAVE_16B = 1,
                CU_TENSOR_MAP_INTERLEAVE_32B = 2,
            }

            public enum L2Promotion
            {
                CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
                CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            }

            public enum Swizzle
            {
                CU_TENSOR_MAP_SWIZZLE_NONE = 0,
                CU_TENSOR_MAP_SWIZZLE_32B,
                CU_TENSOR_MAP_SWIZZLE_64B,
                CU_TENSOR_MAP_SWIZZLE_128B,
            }

            public enum FloatOOBfill
            {
                CU_TENSOR_MAP_FLOAT_OOB_FILL_DEFAULT = 0,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
            }

            public struct TensorRank
            {
                public uint Rank { get; init; }
            }

            public struct GlobalAddress
            {
                public nint Address { get; init; }
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
                    "cuTensorMapEncodeTiled",
                    cudaVersion: 12000,
                    DriverProcAddressFlags.CU_GET_PROC_ADDRESS_DEFAULT,
                    out nint procAddress) == CudaError.CUDA_SUCCESS)
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
