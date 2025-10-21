using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.SPIRV;
using Silk.NET.Vulkan;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using VkBuffer = Silk.NET.Vulkan.Buffer;
using Sop = Silk.NET.SPIRV.Op;
using Silk.NET.Vulkan.Extensions.KHR;

// Minimal Vulkan compute playground.
// - Builds a tiny SPIR-V module in code: writes 42u into a storage buffer at index gl_GlobalInvocationID.x
// - Dispatches N work items and verifies the buffer on CPU.

internal static class Program
{
    private const uint N = 256;

    private static void Main(string[] args)
    {
        if (args.Length > 0)
        {
            switch (args[0].ToLowerInvariant())
            {
                case "list-devices":
                case "devices":
                    VulkanInfo.ListDevices();
                    return;
                case "list-queues":
                case "queues":
                    int? deviceIndex = null;
                    if (args.Length > 1 && int.TryParse(args[1], out var di)) deviceIndex = di;
                    VulkanInfo.ListQueues(deviceIndex);
                    return;
            }
        }

        using var app = new VulkanComputePlayground(N);
        app.Run();
    }
}

file unsafe sealed class VulkanComputePlayground : IDisposable
{
    private readonly uint _count;
    private readonly Vk _vk;

    private Instance _instance;
    private PhysicalDevice _physicalDevice;
    private Device _device;
    private uint _queueFamilyIndex;
    private Queue _queue;
    private CommandPool _cmdPool;
    private CommandBuffer _cmd;
    private VkBuffer _bufA;
    private VkBuffer _bufB;
    private VkBuffer _bufC;
    private DeviceMemory _memA;
    private DeviceMemory _memB;
    private DeviceMemory _memC;
    private DescriptorSetLayout _dsl;
    private PipelineLayout _pipelineLayout;
    private DescriptorPool _descPool;
    private DescriptorSet _descSet;
    private ShaderModule _shaderModule;
    private Pipeline _pipeline;

    public VulkanComputePlayground(uint count)
    {
        _count = Math.Max(1u, count);
        _vk = Vk.GetApi();
        CreateInstance();
        PickPhysicalDeviceAndQueue();
        CreateDeviceAndQueue();
        CreateBuffersAndFill();
        CreateDescriptors();
        CreatePipeline();
        CreateCommandResources();
    }

    public void Dispose()
    {
        var vk = _vk;
        if (_device.Handle != 0)
        {
            if (_pipeline.Handle != 0) vk.DestroyPipeline(_device, _pipeline, null);
            if (_shaderModule.Handle != 0) vk.DestroyShaderModule(_device, _shaderModule, null);
            if (_descPool.Handle != 0) vk.DestroyDescriptorPool(_device, _descPool, null);
            if (_pipelineLayout.Handle != 0) vk.DestroyPipelineLayout(_device, _pipelineLayout, null);
            if (_dsl.Handle != 0) vk.DestroyDescriptorSetLayout(_device, _dsl, null);
            if (_cmdPool.Handle != 0) vk.DestroyCommandPool(_device, _cmdPool, null);
            if (_bufA.Handle != 0) vk.DestroyBuffer(_device, _bufA, null);
            if (_bufB.Handle != 0) vk.DestroyBuffer(_device, _bufB, null);
            if (_bufC.Handle != 0) vk.DestroyBuffer(_device, _bufC, null);
            if (_memA.Handle != 0) vk.FreeMemory(_device, _memA, null);
            if (_memB.Handle != 0) vk.FreeMemory(_device, _memB, null);
            if (_memC.Handle != 0) vk.FreeMemory(_device, _memC, null);
            vk.DeviceWaitIdle(_device);
            vk.DestroyDevice(_device, null);
        }
        if (_instance.Handle != 0)
        {
            vk.DestroyInstance(_instance, null);
        }
    }

    public void Run()
    {
        RecordAndSubmit();
        ReadbackAndValidate();
        Console.WriteLine("Vulkan compute add OK: {0} elements.", _count);
    }

    private unsafe void CreateInstance()
    {
        ApplicationInfo appInfo = new()
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)SilkMarshal.StringToPtr("ILGPU.Vulkan.Playground"),
            ApplicationVersion = new Version32(1, 0, 0),
            PEngineName = (byte*)SilkMarshal.StringToPtr("ILGPU"),
            EngineVersion = new Version32(1, 0, 0),
            ApiVersion = Vk.Version12,
        };

        InstanceCreateInfo createInfo = new()
        {
            SType = StructureType.InstanceCreateInfo,
            PApplicationInfo = &appInfo,
        };

        fixed (Instance* pInstance = &_instance)
        {
            Check(_vk.CreateInstance(in createInfo, null, pInstance));
        }

        SilkMarshal.Free((nint)appInfo.PApplicationName);
        SilkMarshal.Free((nint)appInfo.PEngineName);
    }

    private unsafe void PickPhysicalDeviceAndQueue()
    {
        uint count = 0;
        _vk.EnumeratePhysicalDevices(_instance, ref count, null);
        if (count == 0) throw new InvalidOperationException("No Vulkan devices.");
        var devices = new PhysicalDevice[count];
        fixed (PhysicalDevice* p = devices)
        {
            Check(_vk.EnumeratePhysicalDevices(_instance, ref count, p));
        }

        foreach (var pd in devices)
        {
            uint qCount = 0;
            _vk.GetPhysicalDeviceQueueFamilyProperties(pd, ref qCount, null);
            if (qCount == 0) continue;
            var qprops = new QueueFamilyProperties[qCount];
            fixed (QueueFamilyProperties* qp = qprops)
            {
                _vk.GetPhysicalDeviceQueueFamilyProperties(pd, ref qCount, qp);
            }
            for (uint i = 0; i < qCount; i++)
            {
                if ((qprops[i].QueueFlags & QueueFlags.ComputeBit) != 0)
                {
                    _physicalDevice = pd;
                    _queueFamilyIndex = i;
                    return;
                }
            }
        }
        throw new InvalidOperationException("No compute-capable queue family found.");
    }

    private unsafe void CreateDeviceAndQueue()
    {
        float priority = 1.0f;
        DeviceQueueCreateInfo qinfo = new()
        {
            SType = StructureType.DeviceQueueCreateInfo,
            QueueFamilyIndex = _queueFamilyIndex,
            QueueCount = 1,
            PQueuePriorities = &priority,
        };

        DeviceCreateInfo dinfo = new()
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &qinfo,
        };

        fixed (Device* p = &_device)
        {
            Check(_vk.CreateDevice(_physicalDevice, in dinfo, null, p));
        }
        _vk.GetDeviceQueue(_device, _queueFamilyIndex, 0, out _queue);
    }

    private unsafe (VkBuffer, DeviceMemory) CreateBufferHostVisible(ulong size)
    {
        BufferCreateInfo binfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = BufferUsageFlags.StorageBufferBit |
                    BufferUsageFlags.TransferSrcBit |
                    BufferUsageFlags.TransferDstBit,
            SharingMode = SharingMode.Exclusive,
        };
        VkBuffer buffer;
        Check(_vk.CreateBuffer(_device, in binfo, null, out buffer));

        MemoryRequirements memReq;
        _vk.GetBufferMemoryRequirements(_device, buffer, out memReq);

        _vk.GetPhysicalDeviceMemoryProperties(_physicalDevice, out var memProps);
        int memIndex = FindMemoryTypeIndex(memReq.MemoryTypeBits, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, memProps);
        if (memIndex < 0) throw new InvalidOperationException("No HOST_VISIBLE|HOST_COHERENT memory type.");

        MemoryAllocateInfo ainfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReq.Size,
            MemoryTypeIndex = (uint)memIndex,
        };
        Check(_vk.AllocateMemory(_device, in ainfo, null, out var memory));
        Check(_vk.BindBufferMemory(_device, buffer, memory, 0));

        return (buffer, memory);
    }

    private unsafe void CreateBuffersAndFill()
    {
        ulong size = _count * 4;
        (_bufA, _memA) = CreateBufferHostVisible(size);
        (_bufB, _memB) = CreateBufferHostVisible(size);
        (_bufC, _memC) = CreateBufferHostVisible(size);

        // Fill A[i]=i, B[i]=2*i, clear C
        void WriteSeq(DeviceMemory mem, Func<uint, uint> f)
        {
            void* p;
            Check(_vk.MapMemory(_device, mem, 0, size, 0, &p));
            var span = new Span<uint>(p, (int)_count);
            for (int i = 0; i < span.Length; i++) span[i] = f((uint)i);
            _vk.UnmapMemory(_device, mem);
        }
        WriteSeq(_memA, i => i);
        WriteSeq(_memB, i => 2u * i);
        void* pc;
        Check(_vk.MapMemory(_device, _memC, 0, size, 0, &pc));
        new Span<byte>(pc, (int)size).Clear();
        _vk.UnmapMemory(_device, _memC);
    }

    private unsafe void CreateDescriptors()
    {
        var bindings = stackalloc DescriptorSetLayoutBinding[3];
        for (uint i = 0; i < 3; i++)
        {
            bindings[i] = new DescriptorSetLayoutBinding
            {
                Binding = i,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                StageFlags = ShaderStageFlags.ComputeBit,
            };
        }

        DescriptorSetLayoutCreateInfo dslInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 3,
            PBindings = bindings,
        };
        fixed (DescriptorSetLayout* pDsl = &_dsl)
        {
            Check(_vk.CreateDescriptorSetLayout(_device, in dslInfo, null, pDsl));
        }

        var dslLocal = _dsl;
        DescriptorSetLayout* pLayouts = &dslLocal;
        {
            PipelineLayoutCreateInfo plInfo = new()
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                PSetLayouts = pLayouts,
            };
            fixed (PipelineLayout* pPl = &_pipelineLayout)
            {
                Check(_vk.CreatePipelineLayout(_device, in plInfo, null, pPl));
            }
        }

        DescriptorPoolSize poolSize = new()
        {
            Type = DescriptorType.StorageBuffer,
            DescriptorCount = 3,
        };
        DescriptorPoolCreateInfo dpInfo = new()
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            PoolSizeCount = 1,
            PPoolSizes = &poolSize,
            MaxSets = 1,
        };
        fixed (DescriptorPool* pDp = &_descPool)
        {
            Check(_vk.CreateDescriptorPool(_device, in dpInfo, null, pDp));
        }

        var layout = _dsl;
        DescriptorSetAllocateInfo dsAlloc = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = _descPool,
            DescriptorSetCount = 1,
            PSetLayouts = &layout,
        };
        fixed (DescriptorSet* pDs = &_descSet)
        {
            Check(_vk.AllocateDescriptorSets(_device, in dsAlloc, pDs));
        }

        var writes = stackalloc WriteDescriptorSet[3];
        var infos = stackalloc DescriptorBufferInfo[3];
        infos[0] = new DescriptorBufferInfo { Buffer = _bufA, Offset = 0, Range = _count * 4 };
        infos[1] = new DescriptorBufferInfo { Buffer = _bufB, Offset = 0, Range = _count * 4 };
        infos[2] = new DescriptorBufferInfo { Buffer = _bufC, Offset = 0, Range = _count * 4 };
        for (uint i = 0; i < 3; i++)
        {
            writes[i] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = _descSet,
                DstBinding = i,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                PBufferInfo = &infos[i],
            };
        }
        _vk.UpdateDescriptorSets(_device, 3, writes, 0, null);
    }

    private unsafe void CreatePipeline()
    {
        // Build tiny SPIR-V module in-memory (returns u32 words for proper alignment)
        var spirv = SpirvKernels.BuildAddABtoCShader(localSizeX: 1, localSizeY: 1, localSizeZ: 1);

        fixed (uint* codePtr = spirv)
        {
            ShaderModuleCreateInfo smInfo = new()
            {
                SType = StructureType.ShaderModuleCreateInfo,
                CodeSize = new UIntPtr((uint)(spirv.Length * 4)),
                PCode = codePtr,
            };
            fixed (ShaderModule* p = &_shaderModule)
            {
                Check(_vk.CreateShaderModule(_device, in smInfo, null, p));
            }
        }

        nint entry = SilkMarshal.StringToPtr("main", NativeStringEncoding.UTF8);
        PipelineShaderStageCreateInfo stage = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.ComputeBit,
            Module = _shaderModule,
            PName = (byte*)entry,
        };

        ComputePipelineCreateInfo cpInfo = new()
        {
            SType = StructureType.ComputePipelineCreateInfo,
            Stage = stage,
            Layout = _pipelineLayout,
        };
        fixed (Pipeline* pPipe = &_pipeline)
        {
            Check(_vk.CreateComputePipelines(_device, default, 1, in cpInfo, null, pPipe));
        }
        SilkMarshal.Free(entry);
    }

    private unsafe void CreateCommandResources()
    {
        CommandPoolCreateInfo poolInfo = new()
        {
            SType = StructureType.CommandPoolCreateInfo,
            QueueFamilyIndex = _queueFamilyIndex,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
        };
        fixed (CommandPool* p = &_cmdPool)
        {
            Check(_vk.CreateCommandPool(_device, in poolInfo, null, p));
        }

        CommandBufferAllocateInfo alloc = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = _cmdPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1,
        };
        fixed (CommandBuffer* pCmd = &_cmd)
        {
            Check(_vk.AllocateCommandBuffers(_device, in alloc, pCmd));
        }
    }

    private unsafe void RecordAndSubmit()
    {
        CommandBufferBeginInfo begin = new() { SType = StructureType.CommandBufferBeginInfo };
        Check(_vk.BeginCommandBuffer(_cmd, in begin));
        _vk.CmdBindPipeline(_cmd, PipelineBindPoint.Compute, _pipeline);
        _vk.CmdBindDescriptorSets(_cmd, PipelineBindPoint.Compute, _pipelineLayout, 0, 1, in _descSet, 0, null);
        _vk.CmdDispatch(_cmd, _count, 1, 1);
        Check(_vk.EndCommandBuffer(_cmd));

        var cmdLocal = _cmd;
        CommandBuffer* pCmd = &cmdLocal;
        {
            SubmitInfo submit = new()
            {
                SType = StructureType.SubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = pCmd,
            };
            Check(_vk.QueueSubmit(_queue, 1, in submit, default));
        }
        Check(_vk.QueueWaitIdle(_queue));
    }

    private unsafe void ReadbackAndValidate()
    {
        void* data;
        ulong size = _count * 4;
        Check(_vk.MapMemory(_device, _memC, 0, size, 0, &data));
        Span<uint> span = MemoryMarshal.Cast<byte, uint>(new Span<byte>(data, (int)size));
        for (int i = 0; i < span.Length; i++)
        {
            uint expected = (uint)i + 2u * (uint)i;
            if (span[i] != expected)
                throw new InvalidOperationException($"Validation failed at {i}: {span[i]} != {expected}");
        }
        _vk.UnmapMemory(_device, _memC);
    }

    private static int FindMemoryTypeIndex(uint typeBits, MemoryPropertyFlags required, PhysicalDeviceMemoryProperties props)
    {
        for (int i = 0; i < props.MemoryTypeCount; i++)
        {
            if ((typeBits & (1u << i)) == 0) continue;
            var flags = props.MemoryTypes[i].PropertyFlags;
            if ((flags & required) == required)
                return i;
        }
        return -1;
    }

    private static void Check(Result result)
    {
        if (result != Result.Success)
            throw new VulkanException(result);
    }
}

file static class VulkanInfo
{
    public static unsafe void ListDevices()
    {
        var vk = Vk.GetApi();
        Instance instance = default;
        try
        {
            CreateInstance(vk, out instance);
            uint count = 0;
            vk.EnumeratePhysicalDevices(instance, ref count, null);
            if (count == 0)
            {
                Console.WriteLine("No Vulkan devices found.");
                return;
            }
            var devices = new PhysicalDevice[count];
            fixed (PhysicalDevice* p = devices)
                vk.EnumeratePhysicalDevices(instance, ref count, p);

            for (int idx = 0; idx < devices.Length; idx++)
            {
                var pd = devices[idx];
                vk.GetPhysicalDeviceProperties(pd, out var props);
                string name = SilkMarshal.PtrToString((nint)Unsafe.AsPointer(ref props.DeviceName[0]));
                var api = props.ApiVersion;
                Console.WriteLine($"[{idx}] {name}  (API {DecodeMajor(api)}.{DecodeMinor(api)}.{DecodePatch(api)})  Vendor=0x{props.VendorID:X}  Type={props.DeviceType}");

                // Enumerate device extensions
                uint extCount = 0;
                vk.EnumerateDeviceExtensionProperties(pd, (byte*)null, ref extCount, null);
                var extProps = new ExtensionProperties[extCount == 0 ? 1 : extCount];
                fixed (ExtensionProperties* pExt = extProps)
                {
                    if (extCount > 0)
                        vk.EnumerateDeviceExtensionProperties(pd, (byte*)null, ref extCount, pExt);
                }
                var extNames = new HashSet<string>(StringComparer.Ordinal);
                foreach (ref var e in extProps.AsSpan(0, (int)extCount))
                {
                    string en = SilkMarshal.PtrToString((nint)Unsafe.AsPointer(ref e.ExtensionName[0]));
                    if (!string.IsNullOrEmpty(en)) extNames.Add(en);
                }

                bool hasBf16 = extNames.Contains("VK_KHR_shader_bfloat16");
                bool hasCoopKHR = extNames.Contains(KhrCooperativeMatrix.ExtensionName);
                bool hasCoopNV = extNames.Contains("VK_NV_cooperative_matrix");
                bool hasCoopNV2 = extNames.Contains("VK_NV_cooperative_matrix2");
                Console.WriteLine($"    Extensions: BF16(KHR)={YesNo(hasBf16)}, CoopMat(KHR)={YesNo(hasCoopKHR)}, CoopMatNV-v1={YesNo(hasCoopNV)}, CoopMatNV-v2={YesNo(hasCoopNV2)}");
            }
        }
        finally
        {
            if (instance.Handle != 0)
            {
                Vk.GetApi().DestroyInstance(instance, null);
            }
        }
    }

    static string YesNo(bool value) => value ? "yes" : "no";

    public static unsafe void ListQueues(int? deviceIndex)
    {
        var vk = Vk.GetApi();
        Instance instance = default;
        try
        {
            CreateInstance(vk, out instance);
            uint count = 0;
            vk.EnumeratePhysicalDevices(instance, ref count, null);
            if (count == 0)
            {
                Console.WriteLine("No Vulkan devices found.");
                return;
            }
            var devices = new PhysicalDevice[count];
            fixed (PhysicalDevice* p = devices)
                vk.EnumeratePhysicalDevices(instance, ref count, p);

            int start = 0, end = devices.Length - 1;
            if (deviceIndex.HasValue)
            {
                if (deviceIndex.Value < 0 || deviceIndex.Value >= devices.Length)
                {
                    Console.WriteLine($"Device index out of range. 0..{devices.Length - 1}");
                    return;
                }
                start = end = deviceIndex.Value;
            }

            for (int idx = start; idx <= end; idx++)
            {
                var pd = devices[idx];
                vk.GetPhysicalDeviceProperties(pd, out var props);
                string name = SilkMarshal.PtrToString((nint)Unsafe.AsPointer(ref props.DeviceName[0]));
                Console.WriteLine($"[{idx}] {name}");

                uint qCount = 0;
                vk.GetPhysicalDeviceQueueFamilyProperties(pd, ref qCount, null);
                if (qCount == 0)
                {
                    Console.WriteLine("    (no queue families)");
                    continue;
                }
                var qprops = new QueueFamilyProperties[qCount];
                fixed (QueueFamilyProperties* qp = qprops)
                    vk.GetPhysicalDeviceQueueFamilyProperties(pd, ref qCount, qp);

                for (uint qi = 0; qi < qCount; qi++)
                {
                    var f = qprops[qi].QueueFlags;
                    Console.WriteLine($"    QueueFamily {qi}: count={qprops[qi].QueueCount} flags=[{f}] timestamps={qprops[qi].TimestampValidBits}");
                }

                // Extension support summary again for convenience
                uint extCount = 0;
                vk.EnumerateDeviceExtensionProperties(pd, (byte*)null, ref extCount, null);
                var extProps = new ExtensionProperties[extCount == 0 ? 1 : extCount];
                fixed (ExtensionProperties* pExt = extProps)
                {
                    if (extCount > 0)
                        vk.EnumerateDeviceExtensionProperties(pd, (byte*)null, ref extCount, pExt);
                }
                var extNames = new HashSet<string>(StringComparer.Ordinal);
                foreach (ref var e in extProps.AsSpan(0, (int)extCount))
                {
                    string en = SilkMarshal.PtrToString((nint)Unsafe.AsPointer(ref e.ExtensionName[0]));
                    if (!string.IsNullOrEmpty(en)) extNames.Add(en);
                }
                bool hasBf16 = extNames.Contains("VK_KHR_bfloat16");
                bool hasCoopKHR = extNames.Contains("VK_KHR_cooperative_matrix");
                bool hasCoopNV = extNames.Contains("VK_NV_cooperative_matrix");
                bool hasCoopNV2 = extNames.Contains("VK_NV_cooperative_matrix2");
                Console.WriteLine($"    Extensions: BF16(KHR)={(hasBf16 ? "yes" : "no")}, CoopMat(KHR)={(hasCoopKHR ? "yes" : "no")}, CoopMatNV-v1={(hasCoopNV ? "yes" : "no")}, CoopMatNV-v2={(hasCoopNV2 ? "yes" : "no")}");
            }
        }
        finally
        {
            if (instance.Handle != 0)
            {
                Vk.GetApi().DestroyInstance(instance, null);
            }
        }
    }

    private static unsafe void CreateInstance(Vk vk, out Instance instance)
    {
        Instance local = default;
        byte* app = (byte*)SilkMarshal.StringToPtr("VulkanInfo");
        byte* eng = (byte*)SilkMarshal.StringToPtr("ILGPU");
        try
        {
            ApplicationInfo appInfo = new()
            {
                SType = StructureType.ApplicationInfo,
                PApplicationName = app,
                ApplicationVersion = new Version32(1, 0, 0),
                PEngineName = eng,
                EngineVersion = new Version32(1, 0, 0),
                ApiVersion = Vk.Version12,
            };
            InstanceCreateInfo ci = new()
            {
                SType = StructureType.InstanceCreateInfo,
                PApplicationInfo = &appInfo,
            };
            var res = vk.CreateInstance(in ci, null, out local);
            Check(res);
        }
        finally
        {
            SilkMarshal.Free((nint)app);
            SilkMarshal.Free((nint)eng);
        }
        instance = local;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DecodeMajor(uint version) => (int)((version >> 22) & 0x3FF);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DecodeMinor(uint version) => (int)((version >> 12) & 0x3FF);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DecodePatch(uint version) => (int)(version & 0xFFF);

    private static void Check(Result result)
    {
        if (result != Result.Success)
            throw new VulkanException(result);
    }
}

file sealed class VulkanException : Exception
{
    public VulkanException(Result result) : base($"Vulkan error: {result}") { }
}

// Very small SPIR-V writer for a fixed compute shader that writes a constant to a storage buffer.
file static class SpirvKernels
{
    public static uint[] BuildAddABtoCShader(uint localSizeX, uint localSizeY, uint localSizeZ)
    {
        var w = new SpirvWriter();
        // Header (will patch bound later)
        w.Header(version: 0x00010000, generator: 0, bound: 0, schema: 0);

        // Capabilities and memory model
        w.Op(Op.Capability, (uint)Capability.Shader);
        w.Op(Op.MemoryModel, (uint)AddressingModel.Logical, (uint)MemoryModel.Glsl450);

        // IDs
        const uint idVoid = 1;
        const uint idFuncType = 2;
        const uint idUint = 3;
        const uint idV3u = 4;
        const uint idPtrInputV3u = 5;
        const uint idGlGlobalInvocationId = 6;
        const uint idRuntimeArray = 7;
        const uint idStruct = 8;
        const uint idPtrStorageStruct = 9;
        const uint idBufA = 10;
        const uint idBufB = 11;
        const uint idBufC = 12;
        const uint idPtrStorageUint = 13;
        const uint idConst0 = 14;
        const uint idFunc = 15;
        const uint idLabel = 16;
        const uint idGid = 17;
        const uint idIndex = 18;
        const uint idPtrElemA = 19;
        const uint idPtrElemB = 20;
        const uint idPtrElemC = 21;
        const uint idValA = 22;
        const uint idValB = 23;
        const uint idSum = 24;

        // Entry point + interface
        w.OpEntryPointCompute(idFunc, "main", new[] { idGlGlobalInvocationId, idBufA, idBufB, idBufC });
        w.OpExecutionModeLocalSize(idFunc, localSizeX, localSizeY, localSizeZ);

        // Types
        w.Op(Op.TypeVoid, idVoid);
        w.Op(Op.TypeFunction, idFuncType, idVoid);
        w.Op(Op.TypeInt, idUint, 32u, 0u);
        w.Op(Op.TypeVector, idV3u, idUint, 3u);
        w.Op(Op.TypePointer, idPtrInputV3u, (uint)StorageClass.Input, idV3u);
        w.Op(Op.TypeRuntimeArray, idRuntimeArray, idUint);
        w.OpDecorate(idRuntimeArray, Decoration.ArrayStride, 4u);
        w.Op(Op.TypeStruct, idStruct, idRuntimeArray);
        w.OpMemberDecorate(idStruct, 0u, Decoration.Offset, 0u);
        w.OpDecorate(idStruct, Decoration.Block);
        w.Op(Op.TypePointer, idPtrStorageStruct, (uint)StorageClass.StorageBuffer, idStruct);
        w.Op(Op.TypePointer, idPtrStorageUint, (uint)StorageClass.StorageBuffer, idUint);

        // Builtin + variables
        w.OpDecorate(idGlGlobalInvocationId, Decoration.BuiltIn, (uint)BuiltIn.GlobalInvocationId);
        w.Op(Op.Variable, idPtrInputV3u, idGlGlobalInvocationId, (uint)StorageClass.Input);
        // Three storage buffers at set 0 bindings 0,1,2
        w.OpDecorate(idBufA, Decoration.DescriptorSet, 0u);
        w.OpDecorate(idBufA, Decoration.Binding, 0u);
        w.Op(Op.Variable, idPtrStorageStruct, idBufA, (uint)StorageClass.StorageBuffer);
        w.OpDecorate(idBufB, Decoration.DescriptorSet, 0u);
        w.OpDecorate(idBufB, Decoration.Binding, 1u);
        w.Op(Op.Variable, idPtrStorageStruct, idBufB, (uint)StorageClass.StorageBuffer);
        w.OpDecorate(idBufC, Decoration.DescriptorSet, 0u);
        w.OpDecorate(idBufC, Decoration.Binding, 2u);
        w.Op(Op.Variable, idPtrStorageStruct, idBufC, (uint)StorageClass.StorageBuffer);

        // Constants
        w.Op(Op.Constant, idUint, idConst0, 0u);
        // no constants besides 0

        // Function
        w.Op(Op.Function, idVoid, idFunc, 0u, idFuncType);
        w.Op(Op.Label, idLabel);
        w.Op(Op.Load, idV3u, idGid, idGlGlobalInvocationId);
        w.Op(Op.CompositeExtract, idUint, idIndex, idGid, 0u);
        // A[index]
        w.Op(Op.AccessChain, idPtrStorageUint, idPtrElemA, idBufA, idConst0, idIndex);
        w.Op(Op.Load, idUint, idValA, idPtrElemA);
        // B[index]
        w.Op(Op.AccessChain, idPtrStorageUint, idPtrElemB, idBufB, idConst0, idIndex);
        w.Op(Op.Load, idUint, idValB, idPtrElemB);
        // Sum
        w.Op(Op.IAdd, idUint, idSum, idValA, idValB);
        // C[index] = sum
        w.Op(Op.AccessChain, idPtrStorageUint, idPtrElemC, idBufC, idConst0, idIndex);
        w.Op(Op.Store, idPtrElemC, idSum);
        w.Op(Op.Return);
        w.Op(Op.FunctionEnd);

        w.PatchBound(maxId: 24);
        return w.ToUIntArray();
    }
}

file sealed class SpirvWriter
{
    private readonly List<uint> _words = new();
    private int _headerIndex = -1;

    public void Header(uint version, uint generator, uint bound, uint schema)
    {
        _headerIndex = _words.Count;
        _words.Add(0x07230203u); // Magic
        _words.Add(version);
        _words.Add(generator);
        _words.Add(bound);   // will patch later
        _words.Add(schema);
    }

    public void PatchBound(uint maxId)
    {
        if (_headerIndex < 0) throw new InvalidOperationException("Header not written");
        _words[_headerIndex + 3] = maxId + 1; // bound is max id + 1
    }

    public void Op(Sop op, params uint[] operands)
    {
        uint wc = (uint)(1 + (operands?.Length ?? 0));
        _words.Add(((uint)op) | (wc << 16));
        if (operands != null)
            _words.AddRange(operands);
    }

    public void OpDecorate(uint targetId, Decoration decoration, params uint[] literals)
    {
        var list = new List<uint> { targetId, (uint)decoration };
        if (literals is { Length: > 0 }) list.AddRange(literals);
        Op(Sop.Decorate, list.ToArray());
    }

    public void OpMemberDecorate(uint structId, uint member, Decoration decoration, params uint[] literals)
    {
        var list = new List<uint> { structId, member, (uint)decoration };
        if (literals is { Length: > 0 }) list.AddRange(literals);
        Op(Sop.MemberDecorate, list.ToArray());
    }

    public void OpEntryPointCompute(uint funcId, string name, uint[] interfaceIds)
    {
        var raw = new List<uint> { (uint)ExecutionModel.GLCompute, funcId };
        AppendString(raw, name);
        if (interfaceIds is { Length: > 0 }) raw.AddRange(interfaceIds);
        EmitRaw(Sop.EntryPoint, raw);
    }

    public void OpExecutionModeLocalSize(uint funcId, uint x, uint y, uint z)
    {
        Op(Sop.ExecutionMode, funcId, (uint)ExecutionMode.LocalSize, x, y, z);
    }

    public byte[] ToArray()
    {
        var bytes = new byte[_words.Count * 4];
        for (int i = 0; i < _words.Count; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4, 4), _words[i]);
        }
        return bytes;
    }

    public uint[] ToUIntArray() => _words.ToArray();

    private void EmitRaw(Sop op, List<uint> operands)
    {
        uint wc = (uint)(1 + operands.Count);
        _words.Add(((uint)op) | (wc << 16));
        _words.AddRange(operands);
    }

    private static void AppendString(List<uint> list, string s)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
        for (int i = 0; i < bytes.Length; i += 4)
        {
            uint w = 0;
            if (i + 0 < bytes.Length) w |= bytes[i + 0];
            if (i + 1 < bytes.Length) w |= (uint)bytes[i + 1] << 8;
            if (i + 2 < bytes.Length) w |= (uint)bytes[i + 2] << 16;
            if (i + 3 < bytes.Length) w |= (uint)bytes[i + 3] << 24;
            list.Add(w);
        }
    }
}







