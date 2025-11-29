using Borg.Threading;
using ILGPU;
using ILGPU.Intrinsics.Cuda;
using ILGPU.Runtime.Cuda;
using System.Diagnostics;

using static ILGPU.Kernels;

const int TILE_SIZE = 16;
const int SIZE = 2048;
const int MULTIPLICATIONS_PER_ITERATION = 100;

using var context = Context.Create(builder => builder.Cuda());
var device = context.GetCudaDevice(0);
using var accelerator = device.CreateCudaAccelerator(context);

var err = CudaAPI.CurrentAPI.GetProcAddress("cuTensorMapEncodeTiled",
    cudaVersion: 12000,
    DriverProcAddressFlags.CU_GET_PROC_ADDRESS_DEFAULT,
    out nint cuTensorMapEncodeTiled);

var sw = Stopwatch.StartNew();

/// <summary>
/// Multiplies two dense matrices and returns the resultant matrix (using tiling).
/// </summary>
/// <param name="accelerator">The Accelerator to run the multiplication on</param>
/// <param name="a">A dense MxK matrix</param>
/// <param name="b">A dense KxN matrix</param>
/// <returns>A dense MxN matrix</returns>
static async Task<double> MatrixMultiply(CudaAccelerator accelerator, float[,] a, float[,] b,
                                         TimeSpan iterationTimeout,
                                         CancellationToken cancel)
{
    int m = a.GetLength(0);
    int ka = a.GetLength(1);
    int kb = b.GetLength(0);
    int n = b.GetLength(1);

    if (ka != kb)
        throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix",
                                    nameof(b));

    var kernel = accelerator.LoadStreamKernel<
        Index2D,
        ArrayView2D<MXFP4Block, Stride2D.DenseX>,
        ArrayView2D<MXFP4Block, Stride2D.DenseX>,
        ArrayView2D<BF16, Stride2D.DenseX>,
        Blackwell.TMA.TensorMap,
        Blackwell.TMA.TensorMap>(
        Gemm);
    cancel.ThrowIfCancellationRequested();

    var groupSize = new Index2D(TILE_SIZE, TILE_SIZE);
    var numGroups =
        new Index2D((m + BlockM - 1) / BlockM, (n + BlockN - 1) / BlockN);

    using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
    cancel.ThrowIfCancellationRequested();
    using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(ka, n));
    cancel.ThrowIfCancellationRequested();
    using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
    cancel.ThrowIfCancellationRequested();
    aBuffer.CopyFromCPU(a);
    cancel.ThrowIfCancellationRequested();
    bBuffer.CopyFromCPU(b);
    cancel.ThrowIfCancellationRequested();

    await accelerator.DefaultStream.SynchronizeAsync().ConfigureAwait(false);
    cancel.ThrowIfCancellationRequested();

    // Warmup
    await RunIteration(3).ConfigureAwait(false);

    var time = Stopwatch.StartNew();
    await RunIteration(1).ConfigureAwait(false);
    if (time.Elapsed * MULTIPLICATIONS_PER_ITERATION > iterationTimeout)
        return 1 / time.Elapsed.TotalSeconds;

    time.Restart();
    int iterations = 0;
    while (time.ElapsedMilliseconds < 5000)
    {
        await RunIteration(MULTIPLICATIONS_PER_ITERATION).ConfigureAwait(false);

        iterations += MULTIPLICATIONS_PER_ITERATION;
    }

    time.Stop();
    return iterations / time.Elapsed.TotalSeconds;

    async Task RunIteration(int multiplications)
    {
        using var timeout = iterationTimeout.ToCancellation().Link(cancel);
        var cancelToken = timeout.Token;

        for (int i = 0; i < multiplications; i++)
        {
            kernel((numGroups, groupSize),
                   aBuffer, bBuffer, cBuffer,
                   tmaA, tmaB);
            cancelToken.ThrowIfCancellationRequested();
        }

    done:
        var syncTime = Stopwatch.StartNew();
        await accelerator.DefaultStream.SynchronizeAsync()
                         .WaitAsync(cancelToken).ConfigureAwait(false);
        Console.WriteLine("Synchronized in {syncTime.Elapsed.TotalMilliseconds}ms");
    }
}
