# Owner(s): ["module: inductor"]
"""End-to-end correctness tests for the NUMA attention tensor-parallel pipeline.

Verifies that the compiled NUMA attention block produces output matching the
eager MATH-SDPA reference within fp32 matmul tolerance, with particular focus
on the row-parallel output projection (the historical bug site).

These tests require a build with USE_NUMA=1 and at least 2 NUMA nodes visible
to the runtime.
"""
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing._internal.inductor_utils import HAS_CPU


def _has_numa(min_nodes: int = 2) -> bool:
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None:
        return False
    if not all(
        hasattr(cpu_mod, attr)
        for attr in ("_set_numa_enabled", "_is_numa_enabled", "_num_numa_nodes")
    ):
        return False
    prev = bool(cpu_mod._is_numa_enabled())
    cpu_mod._set_numa_enabled(True)
    try:
        return int(cpu_mod._num_numa_nodes()) >= min_nodes
    finally:
        cpu_mod._set_numa_enabled(prev)


def _eager_reference(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo, B, S, H, E):
    """MATH-backend SDPA reference for the attention block."""
    with sdpa_kernel(SDPBackend.MATH):
        q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
    y = y.transpose(1, 2).contiguous().view(B * S, H * E)
    return torch.addmm(bo, y, Wo)


class AttentionBlock(nn.Module):
    def __init__(self, B, S, H, E):
        super().__init__()
        self.B, self.S, self.H, self.E = B, S, H, E

    def forward(self, x, Wq, Wk, Wv, bq, bk, bv, Wo, bo):
        B, S, H, E = self.B, self.S, self.H, self.E
        with sdpa_kernel(SDPBackend.MATH):
            q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
            k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
            v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(bo, y, Wo)


def _make_inputs(seed, B, S, H_in, H, E):
    H_qkv = H * E
    torch.manual_seed(seed)
    return (
        torch.randn(B * S, H_in),
        torch.randn(H_in, H_qkv),
        torch.randn(H_in, H_qkv),
        torch.randn(H_in, H_qkv),
        torch.randn(H_qkv),
        torch.randn(H_qkv),
        torch.randn(H_qkv),
        torch.randn(H_qkv, H_in),
        torch.randn(H_in),
    )


# Small shapes: all dims distinct, H divisible by 2 for 2-shard TP.
_B, _S, _H_IN, _H, _E = 1, 6, 12, 4, 8  # H_qkv=32, shard_d=16


@unittest.skipUnless(HAS_CPU, "CPU required")
@unittest.skipUnless(_has_numa(2), "requires >= 2 NUMA nodes")
class TestNumaAttentionRowParallel(TestCase):
    """Numerical correctness of the compiled NUMA attention pipeline."""

    def setUp(self):
        cpu_mod = torch._C._cpu
        self._prev_numa = bool(cpu_mod._is_numa_enabled())
        cpu_mod._set_numa_enabled(True)

    def tearDown(self):
        torch._C._cpu._set_numa_enabled(self._prev_numa)

    def _compile_and_run(self, seed):
        inputs = _make_inputs(seed, _B, _S, _H_IN, _H, _E)
        model = AttentionBlock(_B, _S, _H, _E)
        with inductor_config.patch(
            {
                "_numa_attention_tp": True,
                "_numa_attention_tp_world_size": 2,
                "_numa_attention_tp_compute_shard": 0,
                # Force Triton GEMM so the NUMA all-reduce path is exercised.
                # Without this, Inductor falls back to extern_kernels.addmm
                # which bypasses the NUMA dispatch entirely.
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            }
        ):
            compiled = torch.compile(model)
            # warm up
            compiled(*inputs)
            return compiled(*inputs)

    def _reference(self, seed):
        inputs = _make_inputs(seed, _B, _S, _H_IN, _H, _E)
        x, Wq, Wk, Wv, bq, bk, bv, Wo, bo = inputs
        return _eager_reference(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo, _B, _S, _H, _E)

    def test_output_matches_eager_seed0(self):
        """NUMA compiled output matches eager MATH-SDPA reference (seed=0)."""
        compiled_out = self._compile_and_run(0)
        ref = self._reference(0)
        self.assertTrue(torch.isfinite(compiled_out).all(), "compiled output has NaN/Inf")
        self.assertTrue(
            torch.allclose(compiled_out, ref, atol=1e-4, rtol=1e-4),
            f"max_abs_diff={float((compiled_out - ref).abs().max()):.3e}",
        )

    def test_output_matches_eager_seed1(self):
        """NUMA compiled output matches eager MATH-SDPA reference (seed=1)."""
        compiled_out = self._compile_and_run(1)
        ref = self._reference(1)
        self.assertTrue(torch.isfinite(compiled_out).all(), "compiled output has NaN/Inf")
        self.assertTrue(
            torch.allclose(compiled_out, ref, atol=1e-4, rtol=1e-4),
            f"max_abs_diff={float((compiled_out - ref).abs().max()):.3e}",
        )

    def test_output_matches_eager_seed2(self):
        """NUMA compiled output matches eager MATH-SDPA reference (seed=2)."""
        compiled_out = self._compile_and_run(2)
        ref = self._reference(2)
        self.assertTrue(torch.isfinite(compiled_out).all(), "compiled output has NaN/Inf")
        self.assertTrue(
            torch.allclose(compiled_out, ref, atol=1e-4, rtol=1e-4),
            f"max_abs_diff={float((compiled_out - ref).abs().max()):.3e}",
        )

    def test_output_varies_across_seeds(self):
        """Different RNG seeds produce different outputs.

        Catches the historical regression where per-node buffers had
        constant statistics regardless of seed (reading uninitialised memory).
        """
        out0 = self._compile_and_run(0)
        out1 = self._compile_and_run(1)
        self.assertFalse(
            torch.allclose(out0, out1, atol=1e-6, rtol=1e-6),
            "outputs are identical across seeds — indicates constant/stale buffer reads",
        )

    def test_output_is_finite(self):
        """Compiled output has no NaN or Inf for three independent seeds."""
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                out = self._compile_and_run(seed)
                self.assertTrue(
                    torch.isfinite(out).all(),
                    f"seed={seed}: compiled output contains NaN/Inf",
                )


if __name__ == "__main__":
    run_tests()
