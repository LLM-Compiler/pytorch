#!/usr/bin/env python3
# mypy: ignore-errors
"""Run a generated Inductor ``output_code.py`` ``call()`` with per-kernel finite checks.

Enables NUMA (required for ``clone(..., numa_node=...)`` in the wrapper), builds
inputs from ``assert_size_stride`` like ``check_numa_attention_output_correctness.py``,
then executes an instrumented ``call`` that prints ``torch.isfinite`` / min / max
after each ``triton_*.run(...)`` and after the row-parallel ``add.Tensor`` reduce.

Example::

  python scripts/inductor/debug_output_code_intermediate_finite.py \\
    --output-code torch_compile_debug/run_xxx/torchinductor/model__0_inference_0.0/output_code.py

Before each ``aten.add.Tensor`` NUMA row-parallel reduce, prints the two operand
buffers (e.g. ``_numa_p0_buf13_n0`` / ``_numa_p0_buf13_n1``) so you can see if
NaNs appear **before** the sum. Use ``--contiguous-randn`` to avoid
``rand_strided`` overlap artifacts.
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
    p.add_argument(
        "--contiguous-randn",
        action="store_true",
        help="Fill inputs with torch.randn(shape) (contiguous) instead of rand_strided.",
    )
    return p.parse_args()


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
        raise RuntimeError(f"Expected 9 assert_size_stride lines, got {len(out)}.")
    return out


def _output_buf_from_triton_run_line(line: str) -> str | None:
    m = re.search(r"\.run\((.*)\)\s*$", line)
    if not m:
        return None
    inner = m.group(1)
    bufs = re.findall(r"\b(buf\d+)\b", inner)
    return bufs[-1] if bufs else None


def _instrument_call_source(call_src: str) -> str:
    """Append finite checks after each triton *.run(...); before/after NUMA add.Tensor reduce."""
    lines = call_src.splitlines(keepends=True)
    out: List[str] = []
    for line in lines:
        if "_numa_reduce_" in line and "torch.ops.aten.add.Tensor" in line:
            op_m = re.search(
                r"aten\.add\.Tensor\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)",
                line,
            )
            if op_m:
                left, right = op_m.group(1), op_m.group(2)
                out.append(
                    f'    __finite("{left} (pre-reduce)", {left})\n'
                    f'    __finite("{right} (pre-reduce)", {right})\n'
                )
            out.append(line)
            lhs_m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=", line)
            if lhs_m:
                lhs = lhs_m.group(1)
                out.append(f'    __finite("{lhs} (post-reduce)", {lhs})\n')
            continue

        out.append(line)
        if "triton_" in line and ".run(" in line and line.strip().endswith(")"):
            ob = _output_buf_from_triton_run_line(line)
            if ob and re.match(r"^buf\d+$", ob):
                out.append(f"    __finite({ob!r}, {ob})\n")

    return "".join(out)


def __finite(label: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        print(f"{label}: (not a tensor)")
        return
    finite = bool(torch.isfinite(t).all().item())
    t_f = t.detach().float()
    print(
        f"{label}: finite_all={finite}  "
        f"min={float(t_f.min()):.4g}  max={float(t_f.max()):.4g}  "
        f"shape={tuple(t.shape)}"
    )


def main() -> int:
    args = _parse_args()
    path = args.output_code.resolve()
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        print("NUMA hooks missing", file=sys.stderr)
        return 1
    prev = bool(cpu_mod._is_numa_enabled())
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        print("could not enable NUMA", file=sys.stderr)
        return 1
    print(
        "numa enabled:",
        cpu_mod._is_numa_enabled(),
        "nodes:",
        int(cpu_mod._num_numa_nodes()),
    )

    try:
        sys.argv = [str(path)]
        mod = runpy.run_path(str(path), run_name="__out_code_debug__")
        raw_src = textwrap.dedent(inspect.getsource(mod["call"]))
        inst_src = _instrument_call_source(raw_src)
        inst_src = inst_src.replace("def call(", "def call_instrumented(", 1)

        g = dict(mod)
        g["__finite"] = __finite
        exec(inst_src, g)  # noqa: S102

        expected = _parse_expected_inputs(raw_src)
        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        from torch._dynamo.testing import rand_strided

        call_args: List[torch.Tensor] = []
        for shape, stride in expected:
            if args.contiguous_randn:
                call_args.append(
                    torch.randn(shape, device=device, dtype=torch.float32)
                )
            else:
                call_args.append(
                    rand_strided(shape, stride, device=device, dtype=torch.float32)
                )

        print("--- per-kernel / intermediate finite checks ---")
        out = g["call_instrumented"](list(call_args))
        final = out[0]
        print("--- final return ---")
        __finite("return[0] (buf13)", final)
        return 0 if bool(torch.isfinite(final).all()) else 2
    finally:
        cpu_mod._set_numa_enabled(prev)


if __name__ == "__main__":
    raise SystemExit(main())
