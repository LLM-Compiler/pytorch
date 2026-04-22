class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024]", primals_2: "f32[1152, 1024]", primals_3: "f32[512, 1152]", primals_4: "f32[1024]", primals_5: "f32[1152, 1024]", primals_6: "f32[1024]", primals_7: "f32[1152, 1024]", primals_8: "f32[1152]", primals_9: "f32[1024, 1152]", primals_10: "f32[6912]", primals_11: "f32[1152, 6912]", primals_12: "f32[1152]", primals_13: "f32[6912, 1152]"):
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:88 in forward, code: q = torch.addmm(self.bq, x, self.Wq).view(B, S, H, E).transpose(1, 2)
        addmm: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_1, primals_3, primals_2);  primals_1 = primals_2 = None
        view: "f32[1, 512, 4, 256]" = torch.ops.aten.reshape.default(addmm, [1, 512, 4, 256]);  addmm = None
        permute: "f32[1, 4, 512, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:89 in forward, code: k = torch.addmm(self.bk, x, self.Wk).view(B, S, H, E).transpose(1, 2)
        addmm_1: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_4, primals_3, primals_5);  primals_4 = primals_5 = None
        view_1: "f32[1, 512, 4, 256]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 4, 256]);  addmm_1 = None
        permute_1: "f32[1, 4, 512, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:90 in forward, code: v = torch.addmm(self.bv, x, self.Wv).view(B, S, H, E).transpose(1, 2)
        addmm_2: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_6, primals_3, primals_7);  primals_6 = primals_7 = None
        view_2: "f32[1, 512, 4, 256]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 4, 256]);  addmm_2 = None
        permute_2: "f32[1, 4, 512, 256]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:92 in forward, code: y = F.scaled_dot_product_attention(q, k, v)
        mul: "f32[1, 4, 512, 256]" = torch.ops.aten.mul.Scalar(permute, 0.25);  permute = None
        permute_3: "f32[1, 4, 256, 512]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        mul_1: "f32[1, 4, 256, 512]" = torch.ops.aten.mul.Scalar(permute_3, 0.25);  permute_3 = None
        expand: "f32[1, 4, 512, 256]" = torch.ops.aten.expand.default(mul, [1, 4, 512, 256]);  mul = None
        view_3: "f32[4, 512, 256]" = torch.ops.aten.reshape.default(expand, [4, 512, 256]);  expand = None
        expand_1: "f32[1, 4, 256, 512]" = torch.ops.aten.expand.default(mul_1, [1, 4, 256, 512]);  mul_1 = None
        view_4: "f32[4, 256, 512]" = torch.ops.aten.reshape.default(expand_1, [4, 256, 512]);  expand_1 = None
        bmm: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_3, view_4)
        view_5: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 4, 512, 512]);  bmm = None
        amax: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_5, [-1], True)
        sub: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_5, amax);  amax = None
        exp: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        eq: "b8[1, 4, 512, 512]" = torch.ops.aten.eq.Scalar(view_5, -inf);  view_5 = None
        logical_not: "b8[1, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default: "f32[1, 4, 512, 512]" = torch.ops.aten.full.default([1, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "f32[1, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default, div);  logical_not_1 = full_default = div = None
        expand_2: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(where, [1, 4, 512, 512])
        view_6: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [4, 512, 512]);  expand_2 = None
        expand_3: "f32[1, 4, 512, 256]" = torch.ops.aten.expand.default(permute_2, [1, 4, 512, 256]);  permute_2 = None
        view_7: "f32[4, 512, 256]" = torch.ops.aten.reshape.default(expand_3, [4, 512, 256]);  expand_3 = None
        bmm_1: "f32[4, 512, 256]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = None
        view_8: "f32[1, 4, 512, 256]" = torch.ops.aten.reshape.default(bmm_1, [1, 4, 512, 256]);  bmm_1 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:93 in forward, code: y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        permute_4: "f32[1, 512, 4, 256]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone: "f32[1, 512, 4, 256]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9: "f32[512, 1024]" = torch.ops.aten.reshape.default(clone, [512, 1024]);  clone = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:94 in forward, code: attn_out = torch.addmm(self.bo, y, self.Wo)
        addmm_3: "f32[512, 1152]" = torch.ops.aten.addmm.default(primals_8, view_9, primals_9);  primals_8 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:95 in forward, code: h = attn_out + x
        add: "f32[512, 1152]" = torch.ops.aten.add.Tensor(addmm_3, primals_3);  addmm_3 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:97 in forward, code: u = torch.addmm(self.mlp_fc1_b, h, self.mlp_fc1_w)
        addmm_4: "f32[512, 6912]" = torch.ops.aten.addmm.default(primals_10, add, primals_11);  primals_10 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:98 in forward, code: u = F.gelu(u)
        mul_2: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(addmm_4, 0.5)
        mul_3: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(addmm_4, 0.7071067811865476)
        erf: "f32[512, 6912]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_1: "f32[512, 6912]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[512, 6912]" = torch.ops.aten.mul.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:99 in forward, code: mlp_out = torch.addmm(self.mlp_fc2_b, u, self.mlp_fc2_w)
        addmm_5: "f32[512, 1152]" = torch.ops.aten.addmm.default(primals_12, mul_4, primals_13);  primals_12 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:100 in forward, code: return mlp_out + h
        add_2: "f32[512, 1152]" = torch.ops.aten.add.Tensor(addmm_5, add);  addmm_5 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:99 in forward, code: mlp_out = torch.addmm(self.mlp_fc2_b, u, self.mlp_fc2_w)
        permute_5: "f32[1152, 6912]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_6: "f32[6912, 512]" = torch.ops.aten.permute.default(mul_4, [1, 0]);  mul_4 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:97 in forward, code: u = torch.addmm(self.mlp_fc1_b, h, self.mlp_fc1_w)
        permute_7: "f32[6912, 1152]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_8: "f32[1152, 512]" = torch.ops.aten.permute.default(add, [1, 0]);  add = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:94 in forward, code: attn_out = torch.addmm(self.bo, y, self.Wo)
        permute_9: "f32[1152, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_10: "f32[1024, 512]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:92 in forward, code: y = F.scaled_dot_product_attention(q, k, v)
        permute_13: "f32[4, 256, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_14: "f32[4, 256, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        permute_15: "f32[4, 512, 256]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        
         # File: /home/kumara/pytorch-private/scripts/inductor/simple_llm_torch_compile_demo.py:90 in forward, code: v = torch.addmm(self.bv, x, self.Wv).view(B, S, H, E).transpose(1, 2)
        permute_18: "f32[1152, 512]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        return (add_2, where, addmm_4, permute_5, permute_6, permute_7, permute_8, permute_9, permute_10, permute_13, permute_14, permute_15, permute_18)
        