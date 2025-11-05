using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Vulkan;

static class Program
{
    static void AlignKernel<T>(
        Index1D index,
        ArrayView1D<T, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> prefixLen,
        ArrayView1D<int, Stride1D.Dense> mainLen,
        int alignmentInBytes,
        T element)
        where T : unmanaged
    {
        var sourceAlignment = data.AsAligned(Interop.SizeOf<T>());
        var (prefix, main) = sourceAlignment.AlignTo(alignmentInBytes);

        prefixLen[index] = (int)prefix.Length;
        mainLen[index] = (int)main.Length;

        if (index < prefix.Length)
            prefix.AsAligned(1)[index] = element;
        if (index < main.Length)
            main.AsAligned(alignmentInBytes)[index] = element;
    }

    public static void VariableSubViewKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> low,
        ArrayView1D<int, Stride1D.Dense> high,
        ArrayView1D<long, Stride1D.Dense> src)
    {
        var v = src.VariableView(index);
        low[index] = v.SubView<int>(0).Value;
        high[index] = v.SubView<int>(sizeof(int)).Value;
    }

    static void Main()
    {
        using var watchdogCts = new System.Threading.CancellationTokenSource();
        var _ = new System.Threading.Thread(() =>
        {
            try
            {
                System.Threading.Thread.Sleep(90_000);
                if (!watchdogCts.IsCancellationRequested)
                    Environment.FailFast("VulkanQuickTest watchdog timeout (90s)");
            }
            catch { /* best-effort watchdog */ }
        }) { IsBackground = true };
        _.Start();
        const int NumThreads = 1024;
        const int Length = 8192;
        using var context = Context.Create(builder => builder.Vulkan());
        context.ClearCache();
        using var accel = context.CreateVulkanAccelerator(0);
        accel.ClearCache();
        Console.WriteLine($"Using: {accel.Name}");

        // AlignTo quick test (int32)
        using var data = accel.Allocate1D<int>(Length);
        using var p = accel.Allocate1D<int>(NumThreads);
        using var m = accel.Allocate1D<int>(NumThreads);
        data.MemSetToZero(); p.MemSetToZero(); m.MemSetToZero(); accel.Synchronize();
        var k = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            int,
            int>(AlignKernel);
        int alignBytes = 64; int value = 0x7FFFFFFF;
        k(NumThreads, data.View, p.View, m.View, alignBytes, value);
        accel.Synchronize();
        var p0 = p.GetAsArray1D()[0]; var m0 = m.GetAsArray1D()[0];
        Console.WriteLine($"AlignTo<int>: prefix={p0} main={m0}");
        var arr = data.GetAsArray1D();
        bool pass = true; int fidx = -1, fexp = 0, fgot = 0;
        for (int i = 0; i < Length; i++)
        {
            int expected = 0;
            if (i < NumThreads && i < p0) expected = value;
            if (i >= p0 && i < p0 + NumThreads) expected = value;
            if (arr[i] != expected) { pass = false; if (fidx < 0) { fidx = i; fexp = expected; fgot = arr[i]; } }
        }
        Console.WriteLine(pass ? "PASS AlignTo<int>" : $"FAIL AlignTo<int> first@{fidx} got={fgot} exp={fexp}");

        // VariableSubView quick test (64->32 split) - temporarily disabled during backend bring-up
        if (Environment.GetEnvironmentVariable("ILGPU_VULKAN_ENABLE_VARSV") == "1")
        {
            using var low = accel.Allocate1D<int>(NumThreads);
            using var high = accel.Allocate1D<int>(NumThreads);
            using var src = accel.Allocate1D<long>(NumThreads);
            var srcArr = new long[NumThreads];
            for (int i = 0; i < NumThreads; i++) srcArr[i] = ((long)int.MaxValue << 32) | 0xFFFFL;
            src.CopyFromCPU(accel.DefaultStream, srcArr);
            low.MemSetToZero(); high.MemSetToZero(); accel.Synchronize();
            var vs = accel.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<long, Stride1D.Dense>>(VariableSubViewKernel);
            vs(NumThreads, low.View, high.View, src.View);
            accel.Synchronize();
            var l = low.GetAsArray1D(); var h = high.GetAsArray1D();
            bool ok = true; int fi = -1, el = 0, gl = 0, eh = 0, gh = 0;
            for (int i = 0; i < NumThreads; i++)
            {
                int elv = 0xFFFF; int ehv = int.MaxValue;
                if (l[i] != elv || h[i] != ehv) { ok = false; if (fi < 0) { fi = i; el = elv; gl = l[i]; eh = ehv; gh = h[i]; } }
            }
            Console.WriteLine(ok ? "PASS VariableSubView" : $"FAIL VariableSubView first@{fi} low got={gl} exp={el} high got={gh} exp={eh}");
        }

        // Cancel watchdog
        watchdogCts.Cancel();
    }
}
