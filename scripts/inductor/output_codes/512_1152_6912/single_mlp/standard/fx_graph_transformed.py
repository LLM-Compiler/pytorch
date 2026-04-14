class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1152, 6912]", arg1_1: "f32[512, 1152]", arg2_1: "f32[6912]", arg3_1: "f32[6912, 1152]", arg4_1: "f32[1152]"):
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_pytorch_demo.py:51 in forward, code: y = torch.addmm(b1, x, a)
        addmm: "f32[512, 6912]" = torch.ops.aten.addmm.default(arg2_1, arg1_1, arg0_1);  arg2_1 = arg1_1 = arg0_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_pytorch_demo.py:52 in forward, code: y = torch.nn.functional.gelu(y)
        mul: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(addmm, 0.5)
        mul_1: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476);  addmm = None
        erf: "f32[512, 6912]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[512, 6912]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_pytorch_demo.py:53 in forward, code: return torch.addmm(b2, y, b)
        addmm_1: "f32[512, 1152]" = torch.ops.aten.addmm.default(arg4_1, mul_2, arg3_1);  arg4_1 = mul_2 = arg3_1 = None
        return (addmm_1,)
        