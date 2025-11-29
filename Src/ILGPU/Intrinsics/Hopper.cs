using ILGPU.Runtime;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using static ILGPU.Intrinsics.Cuda.PTX;

namespace ILGPU.Intrinsics.Cuda;

public static class Hopper
{
    public abstract class WGMMA: PTX.WGMMA
    {
        public static void CommitGroup() => ThrowPlatformNotSupported();
        public static void WaitGroup(int maxPending) => ThrowPlatformNotSupported();
        public static void Fence() => ThrowPlatformNotSupported();
    }
}
