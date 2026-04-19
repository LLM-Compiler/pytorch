#!/usr/bin/env python3
"""
Validate NUMA output buffers in a generated TorchInductor output_code.py for
the attention block compiled under ``numa_attention_tp_pass``.

Analogous to ``scripts/inductor/check_numa_output_correctness.py`` (MLP), this
script compares:
  1) eager MATH-backend SDPA reference: addmm(Q), addmm(K), addmm(V), SDPA,
     addmm(Out) in stock PyTorch with ``SDPBackend.MATH`` forced.
  2) generated compiled path's final output (the last buffer returned by
     ``call()``).
  3) per-node NUMA output clones of the row-parallel output kernel. If the
     wrapper emitted post-all-reduce per-node re-clones, each clone must
     match the reference; otherwise (terminal re-clones elided) the per-node
     tensors should be raw partial sums that reduce to the final output.

Interpreting failures:
  - ``sum(partial_i) == final`` only checks internal NUMA bookkeeping; it does
    **not** prove the compiled result matches math. If the final buffer contains
    NaN/Inf, the Inductor/Triton NUMA path is suspect, not the eager MATH
    reference (validate with a non-NUMA ``torch.compile`` run if needed).

Usage:
  python scripts/inductor/check_numa_attention_output_correctness.py \
    --output-code torch_compile_debug/run_xxx/torchinductor/model__0_inference_0.0/output_code.py

Optional explicit dims (for addmm Q/K/V -> SDPA -> addmm output):
  python scripts/inductor/check_numa_attention_output_correctness.py \
    --output-code <path> \
    --batch 1 --seq 6 --h-in 12 --num-heads 4 --head-dim 8 --dump-max-error
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re
import runpy
import sys
import textwrap
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-code", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seq", type=int, default=None)
    p.add_argument("--h-in", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument(
        "--dump-max-error",
        action="store_true",
        help=(
            "Print per-buffer max_abs_diff and max_rel_diff against reference. "
            "Keep in mind the error tolerance is usually tight; most of the "
            "time the margin is extremely narrow and small mismatches are "
            "due to non-associative reductions across sockets. If the PASS "
            "check fails, first check whether the delta is within 1e-4 ish "
            "before treating it as a correctness bug."
        ),
    )
    return p.parse_args()


def _enable_numa() -> Tuple[object, bool]:
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError("NUMA runtime hooks are unavailable in this build.")
    prev_enabled = bool(cpu_mod._is_numa_enabled())
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError("Failed to enable NUMA runtime.")
    return cpu_mod, prev_enabled


def _find_final_return_name(call_src: str) -> str:
    """The generated wrapper ends with ``return (bufN, )`` - extract bufN."""
    m = re.search(r"return\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\)\s*$", call_src, re.MULTILINE)
    if m is None:
        raise RuntimeError(
            "Could not find single-tensor 'return (bufN, )' in generated call()."
        )
    return m.group(1)


def _make_call_with_debug(
    module_globals: dict,
) -> Tuple[Callable, List[str], bool, str]:
    """Return (wrapped_call, per_node_buf_names, per_node_matches_reduced, final_name).

    ``per_node_matches_reduced`` mirrors the MLP harness: if the wrapper
    emitted post-all-reduce re-clones of the reduced tensor back onto
    ``_numa_p*_<final>_n*``, then each of those clones should equal the
    reference output. Otherwise, per-node tensors are raw partials and
    must sum to ``final_name``.
    """
    call_fn = module_globals["call"]
    src = textwrap.dedent(inspect.getsource(call_fn))
    final_name = _find_final_return_name(src)

    per_node_buf_pattern = rf"_numa_p\d+_{re.escape(final_name)}_n\d+"
    buf_names = sorted(set(re.findall(per_node_buf_pattern, src)))
    if not buf_names:
        raise RuntimeError(
            f"No _numa_p*_{final_name}_n* buffers found in generated call(). "
            "Check that the output_code.py was produced with "
            "_numa_attention_tp=True."
        )

    per_node_matches_reduced = bool(
        re.search(
            rf"_numa_p\d+_{re.escape(final_name)}_n\d+\s*=\s*torch\.ops\.aten\.clone\.default\("
            r"\s*_numa_reduce_",
            src,
        )
    )

    ret_match = re.search(
        rf"^\s*return\s+\(\s*{re.escape(final_name)}\s*,\s*\)\s*$",
        src,
        flags=re.MULTILINE,
    )
    if ret_match is None:
        raise RuntimeError(
            f"Could not find expected return '({final_name}, )' in call()."
        )
    replacement = f"    return ({final_name}, {', '.join(buf_names)})"
    src = src.replace(ret_match.group(0), replacement)
    src = src.replace("def call(", "def call_with_debug(", 1)
    exec(src, module_globals)  # noqa: S102
    return module_globals["call_with_debug"], buf_names, per_node_matches_reduced, final_name


def _parse_expected_inputs(
    call_src: str,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for m in re.finditer(
        r"assert_size_stride\(\s*arg\d+_1,\s*(\([^\)]*\)),\s*(\([^\)]*\))\s*\)",
        call_src,
    ):
        shape = ast.literal_eval(m.group(1))
        stride = ast.literal_eval(m.group(2))
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(stride, int):
            stride = (stride,)
        out.append((tuple(shape), tuple(stride)))
    if len(out) != 9:
        raise RuntimeError(
            f"Expected 9 assert_size_stride inputs in ``call()`` graph order "
            f"(Dynamo may reorder vs eager); found {len(out)}."
        )
    return out


def _classify_attention_graph_roles(
    shapes: List[Tuple[int, ...]],
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """Map ``call()`` arg index -> semantic role by shape (Dynamo order varies).

    Returns indices in semantic order:
    ``(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)`` each value is an index into the 9 graph inputs.
    """
    if len(shapes) != 9:
        raise RuntimeError(f"Need 9 input tensors, got {len(shapes)}.")
    idx_1d = [i for i, s in enumerate(shapes) if len(s) == 1]
    if len(idx_1d) != 4:
        raise RuntimeError("Expected four 1-D inputs (bq, bk, bv, bo).")
    by_dim: dict[int, list[int]] = {}
    for i in idx_1d:
        by_dim.setdefault(shapes[i][0], []).append(i)
    h_in = None
    h_qkv = None
    for dim, idxs in by_dim.items():
        if len(idxs) == 1:
            h_in = dim
        elif len(idxs) == 3:
            h_qkv = dim
    if h_in is None or h_qkv is None:
        raise RuntimeError(
            "Could not classify biases: need one length-H_in and three length-H_qkv vectors."
        )
    bq_idx, bk_idx, bv_idx = sorted(i for i in idx_1d if shapes[i][0] == h_qkv)
    bo_idx = next(i for i in idx_1d if shapes[i][0] == h_in)

    idx_2d = [i for i, s in enumerate(shapes) if len(s) == 2]
    if len(idx_2d) != 5:
        raise RuntimeError("Expected five 2-D inputs (x, Wq, Wk, Wv, Wo).")
    wo_idx = next(
        i
        for i in idx_2d
        if shapes[i][0] == h_qkv and shapes[i][1] == h_in
    )
    w_idxs = sorted(
        i
        for i in idx_2d
        if shapes[i][0] == h_in and shapes[i][1] == h_qkv
    )
    if len(w_idxs) != 3:
        raise RuntimeError("Could not find three (H_in, H_qkv) projection weights.")
    wq_idx, wk_idx, wv_idx = w_idxs[0], w_idxs[1], w_idxs[2]
    x_candidates = [i for i in idx_2d if i not in (wq_idx, wk_idx, wv_idx, wo_idx)]
    if len(x_candidates) != 1:
        raise RuntimeError("Could not identify activations x [B*S, H_in].")
    x_idx = x_candidates[0]
    if shapes[x_idx][1] != h_in:
        raise RuntimeError("x must be [B*S, H_in].")
    return x_idx, wq_idx, wk_idx, wv_idx, bq_idx, bk_idx, bv_idx, wo_idx, bo_idx


def _semantic_tensors_from_graph(
    graph_args: List[torch.Tensor],
    expected: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> Tuple[torch.Tensor, ...]:
    shapes = [t[0] for t in expected]
    ix, iwq, iwk, iwv, ibq, ibk, ibv, iwo, ibo = _classify_attention_graph_roles(shapes)
    return (
        graph_args[ix],
        graph_args[iwq],
        graph_args[iwk],
        graph_args[iwv],
        graph_args[ibq],
        graph_args[ibk],
        graph_args[ibv],
        graph_args[iwo],
        graph_args[ibo],
    )


def _expected_inputs_graph_order_from_dims(
    template_expected: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    B: int,
    S: int,
    H_in: int,
    H: int,
    E: int,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Build ``assert_size_stride`` shapes/strides for each graph arg index."""
    H_qkv = H * E
    shapes = [t[0] for t in template_expected]
    ix, iwq, iwk, iwv, ibq, ibk, ibv, iwo, ibo = _classify_attention_graph_roles(shapes)
    by_idx: dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
        ix: ((B * S, H_in), (H_in, 1)),
        iwq: ((H_in, H_qkv), (H_qkv, 1)),
        iwk: ((H_in, H_qkv), (H_qkv, 1)),
        iwv: ((H_in, H_qkv), (H_qkv, 1)),
        ibq: ((H_qkv,), (1,)),
        ibk: ((H_qkv,), (1,)),
        ibv: ((H_qkv,), (1,)),
        iwo: ((H_qkv, H_in), (H_in, 1)),
        ibo: ((H_in,), (1,)),
    }
    return [by_idx[i] for i in range(9)]


def _infer_dims_from_expected(
    expected: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> Tuple[int, int, int, int, int]:
    """Infer B, S, H_in, H, E from parsed ``call()`` shapes (any arg order)."""
    shapes = [t[0] for t in expected]
    x_idx, _, _, _, _, _, _, iwo, _ = _classify_attention_graph_roles(shapes)
    BS, H_in = int(shapes[x_idx][0]), int(shapes[x_idx][1])
    H_qkv = int(shapes[iwo][0])
    if shapes[iwo][1] != H_in:
        raise RuntimeError("Wo must be [H_qkv, H_in].")
    E = 64 if H_qkv % 64 == 0 else 8
    while H_qkv % E != 0:
        E //= 2
    H = H_qkv // E
    B = 1
    S = BS // B
    if B * S != BS:
        raise RuntimeError(
            f"Cannot infer B/S from token count {BS}; pass --batch/--seq explicitly."
        )
    return B, S, H_in, H, E


def _reference_from_semantic(
    x: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    Wo: torch.Tensor,
    bo: torch.Tensor,
    B: int,
    S: int,
    H: int,
    E: int,
) -> torch.Tensor:
    q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
    k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
    v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        y = F.scaled_dot_product_attention(q, k, v)
    y = y.transpose(1, 2).contiguous().view(B * S, H * E)
    return torch.addmm(bo, y, Wo)


def _max_error_metrics(actual: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    actual64 = actual.detach().to(torch.float64)
    ref64 = ref.detach().to(torch.float64)
    if not torch.isfinite(actual64).all() or not torch.isfinite(ref64).all():
        return float("nan"), float("nan")
    abs_diff = (actual64 - ref64).abs()
    max_abs = float(abs_diff.max().item())
    denom = ref64.abs().clamp_min(1e-12)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max().item())
    return max_abs, max_rel


def main() -> int:
    args = _parse_args()
    output_code = args.output_code.resolve()
    if not output_code.exists():
        print(f"[error] output_code.py not found: {output_code}", file=sys.stderr)
        return 1

    cpu_mod, prev_enabled = _enable_numa()
    prev_argv = sys.argv[:]
    try:
        sys.argv = [str(output_code)]
        mod = runpy.run_path(str(output_code), run_name="__numa_check__")
        (
            call_with_debug,
            buf_names,
            per_node_matches_reduced,
            final_name,
        ) = _make_call_with_debug(mod)
        call_src = textwrap.dedent(inspect.getsource(mod["call"]))
        template_expected = _parse_expected_inputs(call_src)

        dims_specified = (
            args.batch is not None
            or args.seq is not None
            or args.h_in is not None
            or args.num_heads is not None
            or args.head_dim is not None
        )
        if dims_specified:
            missing = [
                name
                for name, val in (
                    ("--batch", args.batch),
                    ("--seq", args.seq),
                    ("--h-in", args.h_in),
                    ("--num-heads", args.num_heads),
                    ("--head-dim", args.head_dim),
                )
                if val is None
            ]
            if missing:
                raise RuntimeError(
                    f"Missing required dim args: {', '.join(missing)}"
                )
            B, S, H_in, H, E = args.batch, args.seq, args.h_in, args.num_heads, args.head_dim
            expected = _expected_inputs_graph_order_from_dims(
                template_expected, B, S, H_in, H, E
            )
        else:
            B, S, H_in, H, E = _infer_dims_from_expected(template_expected)
            expected = template_expected
            H_qkv = H * E
            print(
                f"[info] inferred B={B} S={S} H_in={H_in} H={H} E={E} H_qkv={H_qkv} "
                "(override with explicit --batch/--seq/--h-in/--num-heads/--head-dim "
                "if wrong)"
            )

        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        from torch._dynamo.testing import rand_strided

        call_args: List[torch.Tensor] = []
        for (shape, stride) in expected:
            call_args.append(
                rand_strided(shape, stride, device=device, dtype=torch.float32)
            )

        x, Wq, Wk, Wv, bq, bk, bv, Wo, bo = _semantic_tensors_from_graph(
            call_args, template_expected
        )
        ref = _reference_from_semantic(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo, B, S, H, E)
        out = call_with_debug(list(call_args))
        final = out[0]
        numa_bufs = out[1:]

        ref_ok = bool(torch.isfinite(ref).all())
        comp_ok = bool(torch.isfinite(final).all())
        print(f"reference finite: {ref_ok}  compiled {final_name} finite: {comp_ok}")
        if not comp_ok:
            print(
                "[note] Compiled output has NaN/Inf: the eager reference can still be "
                "correct; investigate Inductor/Triton CPU NUMA kernels / wrapper, not "
                "this harness."
            )

        ok_main = (
            comp_ok
            and ref_ok
            and torch.allclose(final, ref, atol=args.atol, rtol=args.rtol)
        )
        ok_all = ok_main
        print(f"reference vs {final_name}: {ok_main}")
        if args.dump_max_error:
            max_abs, max_rel = _max_error_metrics(final, ref)
            print(
                f"  {final_name} max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
            )

        if per_node_matches_reduced:
            # Wrapper emitted the post-all-reduce re-clone; each per-node
            # buffer should match the reference directly.
            for name, t in zip(buf_names, numa_bufs):
                ok = torch.allclose(t, ref, atol=args.atol, rtol=args.rtol)
                ok_all = ok_all and ok
                print(f"reference vs {name}: {ok}")
                if args.dump_max_error:
                    max_abs, max_rel = _max_error_metrics(t, ref)
                    print(
                        f"  {name} max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
                    )
        else:
            # Terminal row-parallel output: re-clones elided. Per-node
            # tensors hold raw partial sums; validate that they reduce to
            # ``final`` (the parallel plan's actual correctness contract).
            partial_sum = torch.zeros_like(final)
            for t in numa_bufs:
                partial_sum = partial_sum + t
            ok_partials = torch.allclose(
                partial_sum, final, atol=args.atol, rtol=args.rtol
            )
            ok_all = ok_all and ok_partials
            print(
                f"sum(_numa_p*_{final_name}_n*) vs {final_name}: {ok_partials}  "
                "(re-clones elided as terminal output)"
            )
            if args.dump_max_error:
                max_abs, max_rel = _max_error_metrics(partial_sum, final)
                print(
                    f"  partial_sum vs {final_name} max_abs_diff={max_abs:.6e} "
                    f"max_rel_diff={max_rel:.6e}"
                )

        if not ok_all:
            print(
                "[FAIL] mismatch found in NUMA attention output correctness check.",
                file=sys.stderr,
            )
            return 2

        print(
            f"[PASS] {final_name} matches reference and NUMA per-node buffers are consistent."
        )
        return 0
    finally:
        sys.argv = prev_argv
        cpu_mod._set_numa_enabled(prev_enabled)


if __name__ == "__main__":
    raise SystemExit(main())
