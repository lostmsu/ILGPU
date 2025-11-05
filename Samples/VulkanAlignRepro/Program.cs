using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Vulkan;

static class Program
{
    // Alignment kernel: main branch only
    static void LenGuard(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> data,
        ArrayView1D<long, Stride1D.Dense> prefixLen,
        ArrayView1D<long, Stride1D.Dense> mainLen,
        int alignmentInBytes,
        int value)
    {
        mainLen[index] = data.Length;

        if (index < data.Length)
        {
            data[index] = value;
        }
    }

    // SubView debug kernel: mimics failing test shape "implicit subview length"
    static void SubViewImplicitLenKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> data,
        int subViewOffset,
        ArrayView1D<int, Stride1D.Dense> debugCond,
        ArrayView1D<int, Stride1D.Dense> debugLen,
        int value)
    {
        ArrayView<int> raw = data;
        var sv = raw.SubView(subViewOffset);
        // Record the length observed by this thread
        debugLen[index] = (int)sv.Length;
        var cond = index < sv.Length;
        debugCond[index] = cond ? 1 : 0;
        if (cond)
            sv[index] = value;
    }

    static void Main()
    {
        const int NumThreads = 1024;
        const int Length = 8192;
        using var context = Context.Create(builder => builder.Vulkan());
        // Ensure fresh compilation so IR dumps are generated
        context.ClearCache(ClearCacheMode.Everything);
        using var accel = context.CreateVulkanAccelerator(0);
        Console.WriteLine($"Using: {accel.Name}");
        // Force recompilation to exercise IR dumps and avoid kernel cache
        accel.ClearCache(ClearCacheMode.Everything);

        using var data = accel.Allocate1D<int>(Length);
        data.MemSetToZero();
        accel.Synchronize();

        // Now run alignment repro
        using var prefixBuf = accel.Allocate1D<long>(NumThreads);
        using var mainBuf = accel.Allocate1D<long>(NumThreads);
        prefixBuf.MemSetToZero();
        mainBuf.MemSetToZero();
        data.MemSetToZero();
        accel.Synchronize();

        int alignBytes = 32; // pick a failing case
        int value = int.MaxValue;

        // Helper to validate contents
        void Validate(string label, long pLen, bool prefix, bool main)
        {
            var arr = data.GetAsArray1D();
            int mm = 0; int fidx = -1; int fexp = 0; int fgot = 0;
            for (int i = 0; i < Length; i++)
            {
                int expected = 0;
                if (prefix)
                {
                    if (i < NumThreads && i < pLen) expected = value;
                }
                if (main)
                {
                    if (i >= pLen && i < pLen + NumThreads) expected = value;
                }
                if (arr[i] != expected)
                {
                    mm++;
                    if (fidx < 0) { fidx = i; fexp = expected; fgot = arr[i]; }
                }
            }
            Console.WriteLine(mm == 0
                ? $"PASS {label}: writes correct"
                : $"FAIL {label}: mismatches={mm} first@0x{fidx:X}:{fgot:X8} expected={fexp:X8}");
        }

        // Run main-only
        var mainKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<long, Stride1D.Dense>,
            ArrayView1D<long, Stride1D.Dense>,
            int,
            int>(LenGuard);
        mainKernel(NumThreads, data.View, prefixBuf.View, mainBuf.View, alignBytes, value);
        accel.Synchronize();
        var pLens = prefixBuf.GetAsArray1D();
        var mLens = mainBuf.GetAsArray1D();
        long p0 = pLens[0];
        long m0 = mLens[0];
        Console.WriteLine($"main-only: prefixLen[0]={p0} mainLen[0]={m0}");
        Validate("main-only", p0, prefix: false, main: true);

        // SubView implicit length debug: probe offsets seen in failing tests
        using var dbgCond = accel.Allocate1D<int>(NumThreads);
        using var dbgLen = accel.Allocate1D<int>(NumThreads);
        var subKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            int,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            int>(SubViewImplicitLenKernel);

        foreach (var off in new[] { 0, 1024, 723, 319 })
        {
            data.MemSetToZero(); dbgCond.MemSetToZero(); dbgLen.MemSetToZero(); accel.Synchronize();
            Console.WriteLine($"\nSubViewImplicitLen offset={off}");
            subKernel(NumThreads, data.View, off, dbgCond.View, dbgLen.View, value);
            accel.Synchronize();
            var cond = dbgCond.GetAsArray1D();
            var lens = dbgLen.GetAsArray1D();
            // Print a compact window around the suspected boundary (0..127)
            Console.Write("cond[0..127]: ");
            for (int i = 0; i < Math.Min(128, cond.Length); i++)
            {
                if (i > 0 && i % 32 == 0) Console.Write(" ");
                Console.Write(cond[i] != 0 ? '1' : '0');
            }
            Console.WriteLine();
            Console.Write("len[0..7]:   ");
            for (int i = 0; i < Math.Min(8, lens.Length); i++)
                Console.Write($"{lens[i]} ");
            Console.WriteLine();

            // Also validate written data window 0..(off+NumThreads)
            var arr = data.GetAsArray1D();
            int mismatches = 0, fidx = -1, fexp = 0, fgot = 0;
            for (int i = 0; i < Math.Min(Length, off + NumThreads + 4); i++)
            {
                int expected = (i >= off && i < off + NumThreads) ? value : 0;
                if (arr[i] != expected) { mismatches++; if (fidx < 0) { fidx = i; fexp = expected; fgot = arr[i]; } }
            }
            Console.WriteLine(mismatches == 0
                ? "PASS SubViewImplicitLen write window"
                : $"FAIL SubViewImplicitLen write window: mismatches={mismatches} first@{fidx} got={fgot} exp={fexp}");
        }

        // VariableSubView minimal repro
        Console.WriteLine("\nVariableSubView repro");
        int n = NumThreads;
        using var dataLow = accel.Allocate1D<int>(n);
        using var dataHigh = accel.Allocate1D<int>(n);
        using var srcLong = accel.Allocate1D<long>(n);
        // Fill source with (int.MaxValue << 32) | 0xFFFF
        var srcArr = new long[n];
        for (int i = 0; i < n; i++) srcArr[i] = ((long)int.MaxValue << 32) | 0xFFFFL;
        srcLong.CopyFromCPU(accel.DefaultStream, srcArr);
        // Ensure fresh compile of the next kernel
        accel.ClearCache(ClearCacheMode.Everything);
        var varSubKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<long, Stride1D.Dense>>(VariableSubViewRepro.VariableSubViewKernel);
        dataLow.MemSetToZero(); dataHigh.MemSetToZero(); accel.Synchronize();
        varSubKernel(NumThreads, dataLow.View, dataHigh.View, srcLong.View);
        accel.Synchronize();
        var low = dataLow.GetAsArray1D();
        var high = dataHigh.GetAsArray1D();
        bool ok = true; int fidx2 = -1; int expL = 0, gotL = 0, expH = 0, gotH = 0;
        for (int i = 0; i < n; i++)
        {
            int eL = 0xFFFF; int eH = int.MaxValue;
            if (low[i] != eL || high[i] != eH) { ok = false; if (fidx2 < 0) { fidx2 = i; expL = eL; gotL = low[i]; expH = eH; gotH = high[i]; } }
        }
        Console.WriteLine(ok ? "PASS VariableSubView" : $"FAIL VariableSubView first@{fidx2} low got={gotL} exp={expL} high got={gotH} exp={expH}");
    }
}

static class VariableSubViewRepro
{
    public static void VariableSubViewKernel(
        Index1D index,
        ArrayView1D<int, Stride1D.Dense> data,
        ArrayView1D<int, Stride1D.Dense> data2,
        ArrayView1D<long, Stride1D.Dense> source)
    {
        var view = source.VariableView(index);
        data[index] = view.SubView<int>(0).Value;
        data2[index] = view.SubView<int>(sizeof(int)).Value;
    }
}
