using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Vulkan;
using System.Linq;


static void AddKernel(Index1D i, ArrayView<int> a, ArrayView<int> b, ArrayView<int> c)
    => c[i] = a[i] + b[i];

using var context = Context.Create(builder => builder.Vulkan());

// Pick first Vulkan device
var dev = context.Devices.OfType<VulkanDevice>().FirstOrDefault();
if (dev is null)
{
    Console.Error.WriteLine("No Vulkan device found.");
    return;
}

using var accel = dev.CreateAccelerator(context);

const int N = 1024;
using var bufA = accel.Allocate1D<int>(N);
using var bufB = accel.Allocate1D<int>(N);
using var bufC = accel.Allocate1D<int>(N);

var hostA = new int[N];
var hostB = new int[N];
for (int i = 0; i < N; i++) { hostA[i] = i; hostB[i] = 2 * i + 1; }

bufA.View.CopyFromCPU(hostA);
bufB.View.CopyFromCPU(hostB);

var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(AddKernel);
kernel(N, bufA.View, bufB.View, bufC.View);
accel.Synchronize();

var result = bufC.GetAsArray1D();
for (int i = 0; i < N; i++)
{
    if (result[i] != hostA[i] + hostB[i])
        throw new Exception($"Mismatch at {i}: {result[i]} != {hostA[i] + hostB[i]}");
}
Console.WriteLine("ILGPU Vulkan backend add OK for {0} elements.", N);
