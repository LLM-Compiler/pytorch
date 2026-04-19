class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1152, 6912]", arg1_1: "f32[512, 1152]", arg2_1: "f32[6912]", arg3_1: "f32[6912, 1152]", arg4_1: "f32[1152]"):
        # No stacktrace found for following nodes
        slice_tensor: "f32[1152, 1728]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 1728, 1)
        slice_tensor_1: "f32[1728, 1152]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 1728, 1)
        slice_tensor_2: "f32[1728]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, 1728, 1)
        slice_tensor_3: "f32[1152, 1728]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1728, 3456, 1);  slice_tensor_3 = None
        slice_tensor_4: "f32[1728, 1152]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 1728, 3456, 1);  slice_tensor_4 = None
        slice_tensor_5: "f32[1728]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1728, 3456, 1);  slice_tensor_5 = None
        slice_tensor_6: "f32[1152, 1728]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 3456, 5184, 1);  slice_tensor_6 = None
        slice_tensor_7: "f32[1728, 1152]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 3456, 5184, 1);  slice_tensor_7 = None
        slice_tensor_8: "f32[1728]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 3456, 5184, 1);  slice_tensor_8 = None
        slice_tensor_9: "f32[1152, 1728]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 5184, 6912, 1);  arg0_1 = slice_tensor_9 = None
        slice_tensor_10: "f32[1728, 1152]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 5184, 6912, 1);  arg3_1 = slice_tensor_10 = None
        slice_tensor_11: "f32[1728]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 5184, 6912, 1);  arg2_1 = slice_tensor_11 = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_torch_compile_demo.py:67 in forward, code: y = torch.addmm(b1, x, a)
        addmm: "f32[512, 1728]" = torch.ops.aten.addmm.default(slice_tensor_2, arg1_1, slice_tensor);  slice_tensor_2 = arg1_1 = slice_tensor = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_torch_compile_demo.py:68 in forward, code: y = torch.nn.functional.gelu(y)
        mul: "f32[512, 1728]" = torch.ops.aten.mul.Tensor(addmm, 0.5)
        mul_1: "f32[512, 1728]" = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476);  addmm = None
        erf: "f32[512, 1728]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[512, 1728]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[512, 1728]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        
         # File: /home/shavin/llm-cpu-compiler/pytorch/scripts/inductor/mlp_block_torch_compile_demo.py:69 in forward, code: return torch.addmm(b2, y, b)
        addmm_1: "f32[512, 1152]" = torch.ops.aten.addmm.default(arg4_1, mul_2, slice_tensor_1);  arg4_1 = mul_2 = slice_tensor_1 = None
        return (addmm_1,)
        