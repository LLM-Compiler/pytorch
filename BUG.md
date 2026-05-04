## Root Cause Analysis

The segfault is **not** an OOM, NUMA, or autotune issue. It is a **32-bit byte-offset integer overflow inside one specific Triton-CPU kernel** that only fires when an attention intermediate exceeds 2 GB — which happens for the first time in your sweep at `S=8192`.

### 1. What the model is doing

In `gpt2_multi_size_speedup_cli.py` (lines 134-135), every transformer block forces:

```128:138:scripts/inductor/gpt2_multi_size_speedup_cli.py
    def _attn(self, xn: torch.Tensor, B: int, S: int) -> torch.Tensor:
        H, E = self.H, self.E
        x_flat = xn.reshape(B * S, self.H_in)
        q = torch.addmm(self.bq, x_flat, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x_flat, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x_flat, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
```

`SDPBackend.MATH` materialises the full attention matrix `softmax(QKᵀ/√d) @ V`. With `B=1, H=12, S=8192` Inductor decomposes this into three giant intermediates (visible in the captured `output_code.py`, lines 1615-1619):

```1615:1619:gpt2_compile_debug_artifacts_B2_S8192_117m/torch_compile_debug_runs/M117m_B1_S8192_standard/torch_compile_debug/run_2026_05_04_06_49_40_008087-pid_501632/torchinductor/model__0_inference_0.0/output_code.py
    buf8 = empty_strided_cpu((12, 8192, 8192), (67108864, 8192, 1), torch.float32)
    ...
    buf11 = empty_strided_cpu((1, 12, 8192, 8192), (805306368, 67108864, 8192, 1), torch.float32)
    buf14 = empty_strided_cpu((1, 12, 8192, 8192), (805306368, 67108864, 8192, 1), torch.float32)
```

| Sequence S | Element count of `buf14` | Bytes |
|---|---|---|
| 4096 | 12·4096·4096 = 201M | 805 MB (< 2 GB) |
| **8192** | 12·8192·8192 = **805M** | **3.0 GB (> 2 GB)** |

So `S=8192` is the first sweep point at which **a single SDPA intermediate crosses the 2 GB boundary**. Up to S=4096 every offset within the kernel fits in a signed int32; at S=8192 it does not.

### 2. Where exactly it crashes

With `OMP_NUM_THREADS=1 + faulthandler` the trace pinpoints the failing kernel call:

```
Current thread 0x… (most recent call first):
  File ".../triton/backends/cpu/driver.py", line 857 in __call__
  File "<string>", line 14 in launcher
  File ".../torch/_inductor/runtime/triton_heuristics.py", line 945 in run
  File "/tmp/torchinductor_kumara/lu/...py", line 1629 in call
```

Output line 1629 is:

```1629:1629:/tmp/torchinductor_kumara/lu/cluygwkyuauydn5qsjgevwns43qqv2oumnvkgyq5ljgbippi3kzs.py
    triton_tem_fused_bmm_6.run(buf14, buf13, buf15, grid=torch._inductor.kernel.bmm.bmm_grid(12, 8192, 64, meta0), stream=streamNone)
```

i.e. the **`Softmax(QKᵀ) @ V`** BMM. `M=8192, N=64, K=8192`, batch=12, A = `buf14` (3 GB), with `stride_aq = 67,108,864`, `stride_am = 8,192`.

The kernel-log entries you saw earlier all match this: 10 OMP threads, `error 4` (unmapped read), all at the **same instruction pointer** in this kernel's `.so` — i.e. one deterministic load instruction reads from a wild address.

### 3. The actual bug — i32 byte-offset overflow in Triton-CPU's gather load lowering

In `triton_tem_fused_bmm_6.ttir` the address arithmetic is element-based and stays in i32 (which is fine):

```47:51:/tmp/gpt2_speedup_inductor_cache_wjxrshwj/triton/None/QPGKYIELMBNQAVFVF6NR57G5YJG73SZXNEBPMFKMA2VNS22PSEYA/triton_tem_fused_bmm_6.ttir
    %25 = arith.addi %23, %24 : tensor<32x16xi32>
    %26 = arith.muli %19, %c67108864_i32 : i32 loc(#loc25)
    %27 = tt.splat %26 : i32 -> tensor<32x16xi32>
    %28 = arith.addi %25, %27 : tensor<32x16xi32>
    %30 = tt.addptr %29, %28 : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
```

Worst-case **element** index for `idx_q=11, ram=8191, rk=15` is `8191·8192 + 15 + 11·67,108,864 = 805,306,367` → still inside i32.

But when Triton-CPU lowers the in-loop `tt.load` of A to LLVM, it does:

```192:228:/tmp/gpt2_speedup_inductor_cache_wjxrshwj/triton/None/QPGKYIELMBNQAVFVF6NR57G5YJG73SZXNEBPMFKMA2VNS22PSEYA/triton_tem_fused_bmm_6.llir
  %192 = ptrtoint ptr %0 to i64
  %193 = insertelement <16 x i64> poison, i64 %192, i64 0
  %194 = shufflevector <16 x i64> %193, ...                       ; base ptr replicated
  %195 = shl <16 x i32> %160, splat (i32 2)                       ;  *** byte_offset = idx << 2 IN i32 ***
  ...
  %227 = sext <16 x i32> %195 to <16 x i64>                       ; sign-extend AFTER the shift
  ...
  %259 = add  <16 x i64> %194, %227                               ; base + (negative) byte offset
  ...
  %477 = load <16 x float>, ptr %476, align 4                     ; <-- segfault here
```

The shift-left-by-2 (×4 for sizeof(float)) is performed **as i32 before** the `sext` to i64. For the last element index `805,306,367`:

* `805,306,367 << 2` (i32) = `0xBFFF_FFFC` = **−1,073,741,828** as signed i32  
* `sext` to i64 → `0xFFFF_FFFF_BFFF_FFFC` (huge negative)  
* `base_ptr + huge_negative` → unmapped page → segfault

Same crash IP across all OMP threads, all faulting reads at offsets that are exactly `base − ~1 GB` modulo the program-id splat. The CUDA backend doesn't have this issue because it lowers via `getelementptr float, ptr, i64 (sext idx)` and lets LLVM compute `idx*4` in i64.

The earlier `triton_red_fused__safe_softmax_5` kernel happens to be lowered with the GEP-form (`getelementptr float, ptr, i64 %offset`) so it survives even at S=8192. The store-side of `bmm_4` likewise uses the GEP form. Only the **gather-load lowering used inside the K-loop of the BMM templates** uses the `ptrtoint + (i32<<2) + sext` form, which is the one that overflows.

This is also why every `S ≤ 4096` worked: the worst-case element index was `≤ 12·4096·4096 + … = 201,326,591`, byte offset `805,306,364` — bit 31 stays clear, no overflow.

### 4. Confirming the diagnosis

Removing only the `with sdpa_kernel(SDPBackend.MATH):` line (so SDPA picks the CPU flash-attention path that does not materialise the (B,H,S,S) tensor) makes `B=1, S=8192, 117m` run cleanly:

```
Starting B=1 S=8192 (NO MATH SDPA constraint)
Calling compiled() ...
Output shape: torch.Size([1, 8192, 768]) finite= True
SUCCESS
```

Same model, same world-size, same triton-cpu, same NUMA setup — only the 3 GB intermediates are gone, and the segfault disappears. That is the definitive proof that the trigger is the >2 GB tensor combined with the i32 byte-offset codegen, not memory pressure or anything NUMA-related.

### 5. Recommended fixes

In order of effort:

1. **Workaround for the benchmark (immediate, no rebuild)**  
   In `gpt2_multi_size_speedup_cli.py::GPT2TransformerBlock._attn`, drop the `sdpa_kernel(SDPBackend.MATH)` context manager, or expose it as a CLI flag and default to the standard SDPA dispatcher. The CPU flash-attention path will be picked automatically and never materialises the full attention matrix, so all sequence lengths up to 8192 (and beyond) will run.

2. **Workaround if you specifically need MATH for parity testing**  
   Run S=8192 with `B=1, H=12` in `bf16` (`model = model.to(torch.bfloat16)`, inputs as bf16). Halving the element size puts the worst-case byte offset back under 2³¹.

3. **Real fix in Triton-CPU**  
   In the CPU backend's lowering of `tt.load` / `tt.store` for tensors of pointers (the `ptrtoint + shl + sext + add` pattern shown above), perform the `sext` *before* the `shl` (or use `getelementptr float, ptr, <N x i64>`). That makes the byte offset i64 throughout and matches the safe lowering already used for the GEP form — covering tensors larger than 2 GB on CPU. This is the same class of fix that PyTorch already applies on GPU (compile-time selection of i64 indexing when `numel * sizeof(elem) > INT32_MAX`); it just isn't being honoured by the Triton-CPU vector-gather path.

4. **Defensive guard in the sweep script**  
   When generating the (B,S,H) configurations, refuse to launch a config whose SDPA-MATH intermediate `B*H*S*S*4 ≥ 2³¹` *until* fix #3 lands, so future sweeps fail fast with a readable error instead of a `Segmentation fault (core dumped)`.

If you'd like, I can land the minimal one-liner workaround in `gpt2_multi_size_speedup_cli.py` (option 1), and/or open a tracking note describing the codegen fix for `triton-cpu-private`'s gather lowering.