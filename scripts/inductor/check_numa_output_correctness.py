#!/usr/bin/env python3
"""
Validate NUMA output buffers in a generated TorchInductor output_code.py.

This script compares:
  1) standard (non-NUMA) reference: addmm -> gelu -> addmm
  2) generated compiled path output (buf2)
  3) per-node NUMA output clones (_numa_p*_buf2_n*)

Usage:
  python scripts/inductor/check_numa_output_correctness.py \
    --output-code torch_compile_debug/run_xxx/torchinductor/model__0_inference_0.0/output_code.py

Optional explicit dims (for addmm->gelu->addmm pattern):
  python scripts/inductor/check_numa_output_correctness.py \
    --output-code torch_compile_debug/run_xxx/torchinductor/model__0_inference_0.0/output_code.py \
    --s 2 --h 1152 --d 6912 --hout 1152 --dump-max-error
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
from typing import List, Tuple

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-code", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--s", type=int, default=None, help="Optional MLP sequence/batch dim")
    p.add_argument("--h", type=int, default=None, help="Optional MLP input hidden dim")
    p.add_argument("--d", type=int, default=None, help="Optional MLP intermediate dim")
    p.add_argument("--hout", type=int, default=None, help="Optional MLP output hidden dim")
    p.add_argument(
        "--dump-max-error",
        action="store_true",
        help="Print per-buffer max_abs_diff and max_rel_diff against reference.",
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


def _make_call_with_debug(module_globals: dict) -> Tuple[callable, List[str], bool]:
    """Returns (call_with_debug, buf_names, per_node_matches_reduced).

    ``per_node_matches_reduced`` is True when the generated wrapper emits the
    post-all-reduce per-node re-clone loop (so each ``_numa_p*_buf2_n*``
    tensor is expected to equal the reduced result). When the wrapper elides
    those re-clones for a terminal output (current optimization), per-node
    tensors hold raw partial sums and are expected to add up to ``buf2``.
    """
    call_fn = module_globals["call"]
    src = textwrap.dedent(inspect.getsource(call_fn))
    buf_names = sorted(set(re.findall(r"_numa_p\d+_buf2_n\d+", src)))
    if not buf_names:
        raise RuntimeError("No _numa_p*_buf2_n* buffers found in generated call().")

    # Detect whether the wrapper emitted post-all-reduce re-clones of the
    # reduced result onto the per-node names. The deferred-line mechanism
    # drops these lines when no in-graph consumer needs them (e.g. terminal
    # row-parallel output), so we must compare accordingly.
    per_node_matches_reduced = bool(
        re.search(
            r"_numa_p\d+_buf2_n\d+\s*=\s*torch\.ops\.aten\.clone\.default\("
            r"\s*_numa_reduce_",
            src,
        )
    )

    ret_match = re.search(r"^\s*return\s+\(buf2,\s*\)\s*$", src, flags=re.MULTILINE)
    if ret_match is None:
        raise RuntimeError("Could not find expected return '(buf2, )' in call().")

    replacement = f"    return (buf2, {', '.join(buf_names)})"
    src = src.replace(ret_match.group(0), replacement)
    src = src.replace("def call(", "def call_with_debug(", 1)
    exec(src, module_globals)  # noqa: S102
    return module_globals["call_with_debug"], buf_names, per_node_matches_reduced


def _reference_from_call_order(args: List[torch.Tensor]) -> torch.Tensor:
    # Generated call() argument order for this MLP pattern is:
    #   [a, x, b1, b, b2]
    # while math is:
    #   y = addmm(b1, x, a); out = addmm(b2, gelu(y), b)
    a, x, b1, b, b2 = args
    y = torch.addmm(b1, x, a)
    y = torch.nn.functional.gelu(y)
    return torch.addmm(b2, y, b)


def _parse_expected_inputs(call_src: str) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
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
    if len(out) != 5:
        raise RuntimeError(
            f"Expected 5 assert_size_stride inputs, but found {len(out)}."
        )
    return out


def _expected_inputs_from_dims(
    s: int, h: int, d: int, hout: int
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    # Generated call arg order for this MLP pattern:
    # [a(h,d), x(s,h), b1(d), b(d,hout), b2(hout)]
    return [
        ((h, d), (d, 1)),
        ((s, h), (h, 1)),
        ((d,), (1,)),
        ((d, hout), (hout, 1)),
        ((hout,), (1,)),
    ]


def _max_error_metrics(actual: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    actual64 = actual.detach().to(torch.float64)
    ref64 = ref.detach().to(torch.float64)
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
        # Avoid forwarding this script's CLI args into generated argparse.
        sys.argv = [str(output_code)]
        mod = runpy.run_path(str(output_code), run_name="__numa_check__")
        call_with_debug, buf_names, per_node_matches_reduced = _make_call_with_debug(mod)
        call_src = textwrap.dedent(inspect.getsource(mod["call"]))
        expected = _parse_expected_inputs(call_src)
        dims_specified = (
            args.s is not None
            or args.h is not None
            or args.d is not None
            or args.hout is not None
        )
        if dims_specified:
            if None in (args.s, args.h, args.d, args.hout):
                raise RuntimeError("Provide all of --s/--h/--d/--hout together.")
            expected = _expected_inputs_from_dims(args.s, args.h, args.d, args.hout)

        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        from torch._dynamo.testing import rand_strided

        (x_shape, x_stride), (a_shape, a_stride), (b1_shape, b1_stride), (
            b_shape,
            b_stride,
        ), (b2_shape, b2_stride) = expected
        x = rand_strided(x_shape, x_stride, device=device, dtype=torch.float32)
        a = rand_strided(a_shape, a_stride, device=device, dtype=torch.float32)
        b1 = rand_strided(b1_shape, b1_stride, device=device, dtype=torch.float32)
        b = rand_strided(b_shape, b_stride, device=device, dtype=torch.float32)
        b2 = rand_strided(b2_shape, b2_stride, device=device, dtype=torch.float32)

        call_args = [x, a, b1, b, b2]
        ref = _reference_from_call_order(call_args)
        out = call_with_debug(call_args)
        buf2 = out[0]
        numa_bufs = out[1:]

        ok_main = torch.allclose(buf2, ref, atol=args.atol, rtol=args.rtol)
        ok_all = ok_main
        print(f"reference vs buf2: {ok_main}")
        if args.dump_max_error:
            max_abs, max_rel = _max_error_metrics(buf2, ref)
            print(f"  buf2 max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}")

        if per_node_matches_reduced:
            # Wrapper emitted the per-node re-clone of the reduced result, so
            # each per-node buffer should match the reference directly.
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
            # Terminal row-parallel output: the wrapper's dead re-clones were
            # correctly elided. Per-node tensors now hold raw partial sums
            # (one contribution per NUMA node). Validate that they reduce to
            # ``buf2`` — the actual correctness contract of the parallel plan.
            partial_sum = torch.zeros_like(buf2)
            for t in numa_bufs:
                partial_sum = partial_sum + t
            ok_partials = torch.allclose(
                partial_sum, buf2, atol=args.atol, rtol=args.rtol
            )
            ok_all = ok_all and ok_partials
            print(
                f"sum(_numa_p*_buf2_n*) vs buf2: {ok_partials}  "
                "(re-clones elided as terminal output)"
            )
            if args.dump_max_error:
                max_abs, max_rel = _max_error_metrics(partial_sum, buf2)
                print(
                    f"  partial_sum vs buf2 max_abs_diff={max_abs:.6e} "
                    f"max_rel_diff={max_rel:.6e}"
                )

        if not ok_all:
            print("[FAIL] mismatch found in NUMA output correctness check.", file=sys.stderr)
            return 2

        print("[PASS] buf2 matches reference and NUMA per-node buffers are consistent.")
        return 0
    finally:
        sys.argv = prev_argv
        cpu_mod._set_numa_enabled(prev_enabled)


if __name__ == "__main__":
    raise SystemExit(main())
