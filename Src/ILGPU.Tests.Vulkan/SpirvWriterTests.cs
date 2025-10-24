using ILGPU.Backends.Vulkan;
using Silk.NET.SPIRV;
using System;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Vulkan
{
    public class SpirvWriterTests
    {
        private readonly ITestOutputHelper _output;
        public SpirvWriterTests(ITestOutputHelper output) => _output = output;

        [Fact]
        public void OpWordEncoding_TypeInt()
        {
            var w = new SpirvWriter();
            w.Header(0x00010300u, 0u, 1u, 0u);
            uint resultId = 123u;
            w.Write(Op.TypeInt, resultId, 32u, 0u);
            var words = w.ToUIntArray();

            Assert.True(words.Length >= 5 + 4, "insufficient words written");
            var opword = words[5];
            // Expect high 16 bits word count, low 16 bits opcode
            uint expected = (4u << 16) | (uint)Op.TypeInt;
            Assert.Equal(expected, opword);
            Assert.Equal(resultId, words[6]);
            Assert.Equal(32u, words[7]);
            Assert.Equal(0u, words[8]);
        }

        [Fact]
        public void OpWordEncoding_EntryPoint()
        {
            var w = new SpirvWriter();
            w.Header(0x00010300u, 0u, 1u, 0u);
            uint funcId = 3u;
            w.OpEntryPointCompute(funcId, "main", Array.Empty<uint>());
            var words = w.ToUIntArray();

            Assert.True(words.Length >= 6 + 1, "insufficient words for entry point");
            var opword = words[5];
            // Operands: ExecutionModel (1), FuncId (1), Name (2 words for \"main\"), Interfaces (0)
            uint wc = 1u + 1u + 2u + 1u; // includes the opcode word
            uint expected = (wc << 16) | (uint)Op.EntryPoint;
            Assert.Equal(expected, opword);
            Assert.Equal((uint)ExecutionModel.GLCompute, words[6]);
            Assert.Equal(funcId, words[7]);
        }

        [Fact]
        public void Validate_MinimalModule()
        {
            // Build a minimal valid module and validate with spirv-val if available
            var w = new SpirvWriter();
            w.Header(0x00010300u, 0u, 1u, 0u);
            w.Write(Op.Capability, (uint)Capability.Shader);
            w.Write(Op.MemoryModel, (uint)AddressingModel.Logical, 1u); // GLSL450
            uint idVoid = 1u; // next ids will be patched by builder in real flow; here we craft a trivial func
            // TypeVoid
            w.Write(Op.TypeVoid, idVoid);
            // Function: void %2 = OpFunction %void None %3 (function type omitted here; this is not a full module)
            // This test is only for encoding sanity; use SpirvModuleBuilder for a full validation run

            var bytes = w.ToArray();
            var tmp = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"vl-optest-{Guid.NewGuid():N}.spv");
            System.IO.File.WriteAllBytes(tmp, bytes);
            string output;
            bool hasTools = SpirvValidator.TryValidate(tmp, out output);
            if (!hasTools)
                return; // Tools not available; skip validation
            _output.WriteLine(output ?? string.Empty);
        }

        [Fact]
        public void Validate_MinimalComputeKernel()
        {
            var w = new SpirvWriter();
            w.Header(0x00010300u, 0u, 1u, 0u);
            // Capabilities and memory model
            w.Write(Op.Capability, (uint)Capability.Shader);
            w.Write(Op.MemoryModel, (uint)AddressingModel.Logical, 1u); // GLSL450
            // Entry point and execution mode first, per spec-valid ordering
            uint funcType = 2u; // will define later
            uint funcId = 3u;
            w.OpEntryPointCompute(funcId, "main", Array.Empty<uint>());
            w.OpExecutionModeLocalSize(funcId, 1u, 1u, 1u);
            // Types
            uint voidId = 1u;
            w.Write(Op.TypeVoid, voidId);
            w.Write(Op.TypeFunction, funcType, voidId);
            // Function
            w.Write(Op.Function, voidId, funcId, 0u, funcType);
            uint labelId = 4u;
            w.Write(Op.Label, labelId);
            w.Write(Op.Return);
            w.Write(Op.FunctionEnd);

            var bytes = w.ToArray();
            var tmp = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"vl-min-{Guid.NewGuid():N}.spv");
            System.IO.File.WriteAllBytes(tmp, bytes);
            string output;
            bool hasTools = SpirvValidator.TryValidate(tmp, out output);
            if (!hasTools)
                return; // skip if tools not present
            Assert.True(output.Contains("error") == false, output);
        }
    }
}
