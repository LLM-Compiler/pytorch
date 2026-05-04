# NUMA attention tensor-parallel FX pass: shape collision fixes

This note describes fixes to `torch/_inductor/fx_passes/numa_attention_tp.py` (and the regression test added in `test/inductor/test_numa_attention_tp_pass.py`) that address incorrect rewriting of tensor shapes when **sequence length**, **hidden width**, and **attention decomposition layouts** collide numerically.

The motivating failure was **`torch.compile(..., backend="inductor")`** with NUMA tensor parallelism enabled on **GPT-2 345M**-style configs where **`seq_len == d_model == H_qkv`** (e.g. batch 2, sequence 1024, 16 heads × 64 head dim = 1024). **Standard** compile succeeded; **`numa_tp`** failed at runtime with reshape errors such as:

```text
RuntimeError: shape '[2, 16, 1024, 1024]' is invalid for input of size 8388608
```

117M-style runs (**`d_model = 768`**, **`seq_len = 1024`**) did not hit the same collisions because **`H_qkv`** did not equal **`S`** on attention/score tensors.

---

## Background: what the pass does

The **NUMA attention TP pass** (`numa_attention_tp_pass`) runs on the **post-grad FX graph** after scaled-dot-product attention has been decomposed (e.g. **MATH** backend) into **`bmm`**, softmax, masks, etc. It:

1. **Shards Q/K/V linear weights** column-wise across **`world_size`** sockets (head-aligned when **`num_heads % world_size == 0`**).
2. **Shards the output projection `Wo`** row-wise.
3. Rewrites explicit **shape arguments** on **`view` / `reshape` / `_unsafe_view` / `expand`** (and similar) so that literal **`num_heads`** and **`H_qkv`** dimensions reflect the **per-shard** layout.
4. Refreshes **`meta["val"]`** on the touched subgraph via **`_refresh_chain_vals_via_fake_tensor`** so downstream Inductor lowering sees consistent **FakeTensor** shapes.

The rewriter **`_try_update_shape_param`** historically assumed that any dimension equal to **`H_qkv`** was the **QKV feature width** (safe to replace with **`shard_d = H_qkv // world_size`**), and that shrinking ratios tied cleanly to **`world_size`**. Those assumptions break under common GPT-2-shaped geometry.

---

## Failure mode A: `[..., S, S]` mistaken for QKV width

### Symptom

Attention **scores / softmax / probs** tensors often have layout **`[B·H, S, S]`** or **`[B, H, S, S]`**. When **`S == H_qkv`** (e.g. **345M** at **`S = 1024`**), **both** trailing sequence axes numerically equal **`H_qkv`**.

### Bug

`_try_update_shape_param` (and **`_shrink_shape`**) walked dimensions equal to **`h_qkv`** and replaced the **first** match with **`shard_d`**. On **`[BH, S, S]`**, that could incorrectly shrink a **sequence** axis (e.g. **`[32, 1024, 1024]` → `[32, 512, 1024]`**), breaking element counts versus later reshapes such as **`[2, H, S, S]`**.

### Fix

Introduce **`_skip_h_qkv_for_attn_square_axes(shape, axis_idx)`**: when the tensor has rank ≥ 3 and the **last two** dimensions are **equal** (the canonical **`[..., S, S]`** pattern for score-like tensors), **do not** apply **`h_qkv → shard_d`** on those trailing axes.

---

## Failure mode B: `(B, H, S, E)` with `S == H_qkv`

### Symptom

After **`transpose`**, activations are often **`[B, H, S, E]`** (**BHSE**). Existing **`_is_bshe`** detection covers **`[B, S, H, E]`** only. When **`S == H_qkv`** (including the structural case **`S = num_heads × head_dim`** for typical GPT-2 widths), **`shape[2] == H_qkv`** is **sequence**, not QKV width.

### Bug

The **`h_qkv`** replacement ran on **dimension index 2**, corrupting the sequence axis.

### Fix

Add **`_is_bhse`** to recognize **`[B, H, S, E]`** via **`dim 1 == num_heads`** and **`dim 3 == head_dim`**, and **skip** **`h_qkv → shard_d`** on **`dim 2`** (the sequence axis). **`_shrink_shape`** and the **`num_heads`** literal rewrite loops were aligned so **BHSE** shrinks heads on **`dim 1`** and **BSHE** on **`dim 2`**, analogous to the existing BSHE guards.

---

## Failure mode C: `expand` / views vs `world_size` ratio gate

### Symptom

Broadcast **`expand`** targets often keep a literal **full head count** while inputs are already column-sharded. The ratio between **expanded shape numel** and **input numel** can be **`S / head_dim`** (or similar)—**not** **`world_size`**.

### Bug

`_try_update_shape_param` required **`old_numel / new_input_numel == world_size`** before rewriting **`num_heads`**. **`expand`** nodes therefore kept **`[B, H_full, S, S]`** literals while inputs were **`[B, H_shard, S, E]`**, leading to inconsistent **`bmm`** inputs.

### Fix

Before that ratio gate, apply a **literal `num_heads → shard_heads`** pass for ops that carry explicit shape lists:

- **`aten.expand.default`**
- **`aten.view.default`**
- **`aten._unsafe_view.default`**
- **`aten.reshape.default`**

guarded by the same **BSHE / BHSE** axis rules used elsewhere (`_LITERAL_HEAD_SHAPE_OPS_WITHOUT_RATIO_GATE`).

---

## Failure mode D: post-`bmm` `_unsafe_view` still full-head

Even after **`expand`** was handled, SDPA graphs reshape **`bmm`** outputs back to **`[B, H, S, S]`** using **`view` / `_unsafe_view`**. For those nodes the input/output ratio can be **`world_size²`** (two ranks each shrink by **`world_size`**), so the **ratio gate** still skipped **`num_heads`** rewriting.

### Fix

The same **literal head shrinking without the ratio gate** (failure mode C) was extended to **`view` / `_unsafe_view` / `reshape`**, not only **`expand`**.

---

## Failure mode E: rank-3 tensors and `h_qkv`

### Symptom

After fixing D, **`_unsafe_view`** could become **`[2, H_shard, S, S]`**, but **`bmm`** meta still disagreed (**wrong numel**) because **`h_qkv → shard_d`** fired on **rank-3** intermediates such as **`[BH, A, B]`** where **one** of **`A` or **`B`** equals **`H_qkv`** but is **not** the token×feature axis (only **`(..., H_qkv)`** rank-2 matrices use **`dim 1`** as QKV width for **`shard_d`**).

### Bug

**`_skip_h_qkv_for_attn_square_axes`** only skips when **`shape[-2] == shape[-1]`**. Asymmetric rank-3 views (**`[8, 32, 64]`**) still matched **`h_qkv`** on **`64`** and were wrongly halved.

### Fix

**`_skip_h_qkv_replace_rank3`**: **never** apply **`h_qkv → shard_d`** on **rank-3** shapes inside these rewrite paths. Head merging / BH dims are still handled via **`num_heads`** literals or **`world_size`** divisibility on the batch×head dimension where appropriate.

---

## Failure mode F: subgraph refresh order

### Symptom

**`_refresh_chain_vals_via_fake_tensor`** iterated **`chain_nodes`** in **`graph.nodes` insertion order**, which is **not** guaranteed to be a valid topological schedule over the filtered subset.

### Bug

Consumers such as **`expand`** could run **before** producers had updated **`meta["val"]`**, causing inconsistent fake evaluations and cascading shrink fallbacks.

### Fix

**`_topo_sorted_subset`** now returns nodes in **dependency order** (inputs before users) via DFS post-order over the subset.

---

## Test coverage

**`test/inductor/test_numa_attention_tp_pass.py`** gained **`test_seq_len_equals_h_qkv_attention_square_axes`**, using **`S = H_qkv = num_heads × head_dim`** on small tensors so CI exercises:

- **`[..., S, S]`** collision handling,
- **BHSE** collision (**`S == H_qkv`** on **`transpose`** layouts),
- **`expand` / `view`** literal head rewriting without relying solely on **`ratio == world_size`**,
- rank-3 **`h_qkv`** non-replacement.

Existing tests (**metadata, numerical equivalence, world_size 1 noop, pre-SDPA view rewrite**) continue to guard regressions.

---

## Files touched

| File | Role |
|------|------|
| `torch/_inductor/fx_passes/numa_attention_tp.py` | All logic above: helpers, `_try_update_shape_param`, `_shrink_shape`, `_topo_sorted_subset` |
| `test/inductor/test_numa_attention_tp_pass.py` | Regression **`test_seq_len_equals_h_qkv_attention_square_axes`** |
| `scripts/inductor/NUMA_ATTENTION_TP_SHAPE_COLLISION_FIX.md` | This document |

---

## Operational takeaway

When benchmarking **GPT-2–shaped** models with **`torch.compile`** + NUMA TP on CPU Inductor, configurations where **`seq_len`** equals **`d_model`** (hence **`S == H_qkv`**) are **legal** and common for wide models at moderate sequence lengths. The pass must distinguish:

1. **QKV width** (**rank-2 `dim 1`**, rank-4 **BSHE/BHSE** head and seq axes),
2. **Attention map** axes (**`[..., S, S]`**),
3. **Rank-3** SDPA intermediates (never blindly **`shard_d`** on **`h_qkv`** literals),

and must rewrite **literal head counts** on **`expand` / view / reshape / `_unsafe_view`** even when **numel ratios are not **`world_size`**.
