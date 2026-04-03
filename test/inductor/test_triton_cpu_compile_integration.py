# Owner(s): ["module: inductor"]
"""
Runs test/inductor/triton_cpu_compile_integration.py in a subprocess so
TORCH_COMPILE_DEBUG is set before any torch._inductor import (required for trace output).
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch._inductor.test_case import TestCase, run_tests
from torch.testing._internal.inductor_utils import HAS_CPU
from torch.utils._triton import has_triton


if has_triton():
    import triton

    TRITON_HAS_CPU = "cpu" in triton.backends.backends
else:
    TRITON_HAS_CPU = False

_SCRIPT = Path(__file__).resolve().parent / "triton_cpu_compile_integration.py"


@unittest.skipIf(not HAS_CPU, "requires CPU")
@unittest.skipIf(not TRITON_HAS_CPU, "requires Triton with CPU backend")
class TestTritonCpuCompileIntegration(TestCase):
    def test_llm_style_compile_smoke(self) -> None:
        """Post-build smoke: Inductor CPU + Triton generates output_code with Triton kernels."""
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["TORCH_COMPILE_DEBUG"] = "1"
            env["TORCH_COMPILE_DEBUG_DIR"] = tmp
            # CI-friendly: fast + allow extern GEMM (upstream CPU often keeps mm/addmm as extern)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--fast",
                    "--fresh",
                    "--allow-extern-gemm",
                ],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                self.fail(
                    "triton_cpu_compile_integration.py failed:\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests()
