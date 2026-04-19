class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8]", arg1_1: "f32[2, 4]", arg2_1: "f32[8]", arg3_1: "f32[8, 4]", arg4_1: "f32[4]", arg5_1: "f32[4, 8]", arg6_1: "f32[8]", arg7_1: "f32[8, 4]", arg8_1: "f32[4]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 2, 1)
        slice_tensor_1: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 2, 1)
        slice_tensor_2: "f32[2]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, 2, 1)
        slice_tensor_3: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 2, 4, 1);  slice_tensor_3 = None
        slice_tensor_4: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 2, 4, 1);  slice_tensor_4 = None
        slice_tensor_5: "f32[2]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, 4, 1);  slice_tensor_5 = None
        slice_tensor_6: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 4, 6, 1);  slice_tensor_6 = None
        slice_tensor_7: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 4, 6, 1);  slice_tensor_7 = None
        slice_tensor_8: "f32[2]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 4, 6, 1);  slice_tensor_8 = None
        slice_tensor_9: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 6, 8, 1);  arg0_1 = slice_tensor_9 = None
        slice_tensor_10: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 6, 8, 1);  arg3_1 = slice_tensor_10 = None
        slice_tensor_11: "f32[2]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 6, 8, 1);  arg2_1 = slice_tensor_11 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:41 in forward, code: y1 = torch.addmm(b1_1, x, a1)
        addmm: "f32[2, 2]" = torch.ops.aten.addmm.default(slice_tensor_2, arg1_1, slice_tensor);  slice_tensor_2 = arg1_1 = slice_tensor = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:42 in forward, code: y1 = torch.nn.functional.gelu(y1)
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(addmm, 0.5)
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476);  addmm = None
        erf: "f32[2, 2]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:43 in forward, code: y1 = torch.addmm(b1_2, y1, b1_w)
        addmm_1: "f32[2, 4]" = torch.ops.aten.addmm.default(arg4_1, mul_2, slice_tensor_1);  arg4_1 = mul_2 = slice_tensor_1 = None
        
        # No stacktrace found for following nodes
        slice_tensor_12: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 0, 2, 1)
        slice_tensor_13: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, 2, 1)
        slice_tensor_14: "f32[2]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, 2, 1)
        slice_tensor_15: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 2, 4, 1);  slice_tensor_15 = None
        slice_tensor_16: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, 4, 1);  slice_tensor_16 = None
        slice_tensor_17: "f32[2]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 2, 4, 1);  slice_tensor_17 = None
        slice_tensor_18: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 4, 6, 1);  slice_tensor_18 = None
        slice_tensor_19: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 4, 6, 1);  slice_tensor_19 = None
        slice_tensor_20: "f32[2]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 4, 6, 1);  slice_tensor_20 = None
        slice_tensor_21: "f32[4, 2]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 6, 8, 1);  arg5_1 = slice_tensor_21 = None
        slice_tensor_22: "f32[2, 4]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 6, 8, 1);  arg7_1 = slice_tensor_22 = None
        slice_tensor_23: "f32[2]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 6, 8, 1);  arg6_1 = slice_tensor_23 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:45 in forward, code: y2 = torch.addmm(b2_1, y1, a2)
        addmm_2: "f32[2, 2]" = torch.ops.aten.addmm.default(slice_tensor_14, addmm_1, slice_tensor_12);  slice_tensor_14 = addmm_1 = slice_tensor_12 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:46 in forward, code: y2 = torch.nn.functional.gelu(y2)
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(addmm_2, 0.5)
        mul_4: "f32[2, 2]" = torch.ops.aten.mul.Tensor(addmm_2, 0.7071067811865476);  addmm_2 = None
        erf_1: "f32[2, 2]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_5: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/2_consec_MLPs/two_consec_mlp_torch_compile_demo.py:47 in forward, code: y2 = torch.addmm(b2_2, y2, b2_w)
        addmm_3: "f32[2, 4]" = torch.ops.aten.addmm.default(arg8_1, mul_5, slice_tensor_13);  arg8_1 = mul_5 = slice_tensor_13 = None
        return (addmm_3,)
        