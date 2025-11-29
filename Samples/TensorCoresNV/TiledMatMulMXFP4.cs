namespace ILGPU;

using ILGPU.Intrinsics.Cuda;
using static ILGPU.Intrinsics.Cuda.Blackwell.WGMMA;
using static ILGPU.Intrinsics.Cuda.MatrixLayout;
using static ILGPU.Intrinsics.Cuda.TensorRole;

class Kernels
{
#warning Shapes specific to Blackwell. For others must use constant folding or TypeNum
    public const int BlockM = 128;
    public const int BlockN = 128;
    public const int BlockK = 256;

    /// <param name="tile">(tileM, tileN)</param>
    /// <param name="A">[M, K_packed] row-major</param>
    /// <param name="B">[K, N_packed] mxfp4, col-major or packed layout</param>
    /// <param name="C">[M, N], row-major</param>
    /// <param name="M"></param>
    /// <param name="N"></param>
    /// <param name="K">Contraction dimension. logical K (elements), not packed-count</param>
    /// <param name="tmaA"></param>
    /// <param name="tmaB"></param>
    public static void Gemm(
        Index2D tile,
        ArrayView2D<MXFP4Block, Stride2D.DenseX> A,
        ArrayView2D<MXFP4Block, Stride2D.DenseX> B,
        ArrayView2D<BF16, Stride2D.DenseX> C,
        Blackwell.TMA.Descriptor tmaA,
        Blackwell.TMA.Descriptor tmaB)
    {
        if (!Blackwell.IsSupported || !Blackwell.WGMMA.IsSupported)
            return;

        int M = C.IntExtent.X;
        int N = C.IntExtent.Y;
        int KBlocks = A.IntExtent.Y;
        int K = KBlocks * MXFP4Block.Elements;

        // This block's tile origin in C
        int tileM = tile.X * BlockM;
        int tileN = tile.Y * BlockN;

        if (tileM >= M || tileN >= N)
            return;

        //
        // Shared memory: one K-slice of A (128×256) and B (256×128) in MXFP4 packed.
        // here we pretend 1 "element" in shared = 8 logical FP4s.
        //
        int elemsA = BlockM * BlockK; // in logical FP4s; packing is handled in descriptor
        int elemsB = BlockK * BlockN;

        var shared = SharedMemory.Allocate<MXFP4Block>((elemsA + elemsB) / MXFP4Block.Elements);

        var sharedA = shared.SubView(0, (elemsA / MXFP4Block.Elements));
        var sharedB = shared.SubView((elemsA / MXFP4Block.Elements), (elemsB / MXFP4Block.Elements));

        // TMA barrier
        Blackwell.TMA.Barrier barrier = default;
        Blackwell.TMA.BarrierInit(ref barrier, expectedTransactions: 2);

        // Accumulator fragment for this 128×128 C tile (in bf16)
        Tensor<Accumulator, AccumulatorFragment, RowMajor> acc = default;

        int numKTiles = (K + BlockK - 1) / BlockK;

        for (int kTile = 0; kTile < numKTiles; ++kTile)
        {
            int kBase = kTile * BlockK;
            if (kBase >= K)
                break;

            //
            // 1. TMA: global → shared for current tile of A and B in MXFP4.
            //    (Descriptors encode packing & strides.)
            //

            Blackwell.TMA.Load2D(
                in tmaA,
                sharedA,
                tileM,   // row in A
                kBase,   // col in A (K)
                ref barrier);

            Blackwell.TMA.Load2D(
                in tmaB,
                sharedB,
                kBase,   // row in B (K)
                tileN,   // col in B (N)
                ref barrier);

            Blackwell.TMA.BarrierArrive(ref barrier, transactionCount: 2);
            Blackwell.TMA.BarrierWait(in barrier);

            //
            // 2. LdMatrix: shared → register fragments (packed mxfp4).
            //    In practice you’d compute elementOffset from warp-group/lane.
            //

            var fragA = Blackwell.LdMatrix.Load_M8N8X4<MatrixA, MXFP4Block, RowMajor>(
                sharedA,
                elementOffset: 0,
                transpose: false);

            var fragB = Blackwell.LdMatrix.Load_M8N8X4<MatrixB, MXFP4Block, ColMajor>(
                sharedB,
                elementOffset: 0,
                transpose: false);

            //
            // 3. WGMMA: m128n128k256.bf16.f4e2m1x8.f4e2m1x8
            //    Native mxfp4 × mxfp4 → bf16 MMA on Blackwell Tensor Cores.
            //
            Blackwell.WGMMA.MmaAsync(
                ref acc,
                in fragA,
                in fragB,
                new M128N128K256());

            // Manage async group; here we just have 1 op per slice for clarity.
            Blackwell.WGMMA.CommitGroup();
            Blackwell.WGMMA.WaitGroup(maxPending: 0);
            Blackwell.WGMMA.Fence();
        }

        //
        // 4. Store bf16 accumulator tile back to C (row-major).
        //

        var cTile = C.SubView(
            new(tileM, tileN),
            new(Math.Min(BlockM, M - tileM),
                Math.Min(BlockN, N - tileN)));

        StoreTile_M128N128(
            in acc,
            cTile,
            leadingDim: (int)C.Extent.Y);
    }

    // Called from your Gemm kernel
    static void StoreTile_M128N128(
        in Tensor<Accumulator, AccumulatorFragment, RowMajor> acc,
        ArrayView2D<BF16, Stride2D.DenseX> cTile,
        int leadingDim)
    {
        // Flattened thread index within the block
        // (ILGPU: Group.Dimension.X * Group.IdxY + Group.IdxX, etc.)
        int lane = Group.LinearIndex; // 0..(Group.Dimension.Size-1)

        // We'll assume a 128-thread warpgroup (4 warps of 32)
        // and map each lane to a 2×2 patch:
        //
        //   rowPatch = lane / 16  → 0..7
        //   colPatch = lane % 16  → 0..15
        //
        // So the tile is covered by 8×16 patches, each 2×2 → 16×32 patches = 128×128
        int rowPatch = lane / 16;
        int colPatch = lane % 16;

        int rowBase = rowPatch * 2;
        int colBase = colPatch * 2;

        // Actual tile extents (handle ragged edges)
        int maxRows = (int)cTile.Extent.X;
        int maxCols = (int)cTile.Extent.Y;

        // Write the 2×2 fragment, guarding edges
        if (rowBase < maxRows && colBase < maxCols)
            cTile[rowBase, colBase] = acc.Element.C00;

        if (rowBase < maxRows && colBase + 1 < maxCols)
            cTile[rowBase, colBase + 1] = acc.Element.C01;

        if (rowBase + 1 < maxRows && colBase < maxCols)
            cTile[rowBase + 1, colBase] = acc.Element.C10;

        if (rowBase + 1 < maxRows && colBase + 1 < maxCols)
            cTile[rowBase + 1, colBase + 1] = acc.Element.C11;
    }

    // Accumulator fragment: each thread owns a small 2×2 patch
    // of the full 128×128 tile. Real HW layout is more complex,
    // but this is a sane software model and lets you write a
    // concrete StoreTile.
    struct AccumulatorFragment
    {
        public BF16 C00;
        public BF16 C01;
        public BF16 C10;
        public BF16 C11;

        public void Clear()
        {
            C00 = default;
            C01 = default;
            C10 = default;
            C11 = default;
        }
    }
}
