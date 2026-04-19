class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8]", arg1_1: "f32[2, 4]", arg2_1: "f32[8]", arg3_1: "f32[8, 4]", arg4_1: "f32[4]", arg5_1: "f32[4, 8]", arg6_1: "f32[8]", arg7_1: "f32[8, 4]", arg8_1: "f32[4]"):
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:50 in forward, code: y1 = torch.addmm(b1_1, x, a1)
        addmm: "f32[2, 8]" = torch.ops.aten.addmm.default(arg2_1, arg1_1, arg0_1);  arg2_1 = arg1_1 = arg0_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:51 in forward, code: y1 = torch.nn.functional.gelu(y1)
        mul: "f32[2, 8]" = torch.ops.aten.mul.Tensor(addmm, 0.5)
        mul_1: "f32[2, 8]" = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476);  addmm = None
        erf: "f32[2, 8]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[2, 8]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[2, 8]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:52 in forward, code: y1 = torch.addmm(b1_2, y1, b1_w)
        addmm_1: "f32[2, 4]" = torch.ops.aten.addmm.default(arg4_1, mul_2, arg3_1);  arg4_1 = mul_2 = arg3_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:54 in forward, code: y2 = torch.addmm(b2_1, y1, a2)
        addmm_2: "f32[2, 8]" = torch.ops.aten.addmm.default(arg6_1, addmm_1, arg5_1);  arg6_1 = addmm_1 = arg5_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:55 in forward, code: y2 = torch.nn.functional.gelu(y2)
        mul_3: "f32[2, 8]" = torch.ops.aten.mul.Tensor(addmm_2, 0.5)
        mul_4: "f32[2, 8]" = torch.ops.aten.mul.Tensor(addmm_2, 0.7071067811865476);  addmm_2 = None
        erf_1: "f32[2, 8]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[2, 8]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_5: "f32[2, 8]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_pytorch_demo.py:56 in forward, code: y2 = torch.addmm(b2_2, y2, b2_w)
        addmm_3: "f32[2, 4]" = torch.ops.aten.addmm.default(arg8_1, mul_5, arg7_1);  arg8_1 = mul_5 = arg7_1 = None
        return (addmm_3,)
        