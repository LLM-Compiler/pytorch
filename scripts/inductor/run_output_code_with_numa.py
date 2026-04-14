#!/usr/bin/env python3
"""
Run a generated TorchInductor output_code.py with NUMA enabled.

Example:
  python scripts/inductor/run_output_code_with_numa.py \
    torch_compile_debug/run_.../torchinductor/model__0_inference_0.0/output_code.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a generated output_code.py with NUMA enabled."
    )
    parser.add_argument(
        "output_code",
        type=Path,
        help="Path to generated torchinductor output_code.py",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target = args.output_code.resolve()
    if not target.exists():
        print(f"[run_output_code_with_numa] file not found: {target}", file=sys.stderr)
        return 1

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        print(
            "[run_output_code_with_numa] torch._C._cpu NUMA hooks are unavailable",
            file=sys.stderr,
        )
        return 2

    prev_enabled = bool(cpu_mod._is_numa_enabled())
    cpu_mod._set_numa_enabled(True)
    prev_argv = sys.argv[:]
    try:
        nodes = int(cpu_mod._num_numa_nodes()) if hasattr(cpu_mod, "_num_numa_nodes") else -1
        print(
            f"[run_output_code_with_numa] NUMA enabled=True, num_nodes={nodes}, file={target}",
            flush=True,
        )
        # output_code.py has its own argparse; avoid forwarding this wrapper's
        # CLI arguments.
        sys.argv = [str(target)]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = prev_argv
        cpu_mod._set_numa_enabled(prev_enabled)
        print(
            f"[run_output_code_with_numa] restored NUMA enabled={prev_enabled}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
