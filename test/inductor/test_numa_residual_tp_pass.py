# Owner(s): ["module: inductor"]
import math

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch._inductor.fx_passes.numa_attention_tp import numa_attention_tp_pass
from torch._inductor.fx_passes.numa_layer_norm_tp import numa_layer_norm_tp_pass
from torch._inductor.fx_passes.numa_mlp_tp import numa_mlp_tp_pass
from torch._inductor.fx_passes.numa_residual_tp import numa_residual_tp_pass
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


def _attn_residual_fn_factory(B, S, H, E):
    H_qkv = H * E

    def fn(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo):
        q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
        scale = 1.0 / math.sqrt(E)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B * S, H_qkv)
        z = torch.addmm(bo, y, Wo)
        return z + x

    return fn


def _mlp_residual_fn(x, A, b1, B_, b2):
    h = torch.addmm(b1, x, A)
    h = torch.nn.functional.gelu(h)
    z = torch.addmm(b2, h, B_)
    return z + x


def _attn_inputs(B, S, H_in, H, E):
    H_qkv = H * E
    torch.manual_seed(0)
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


def _mlp_inputs(M, H, D):
    torch.manual_seed(1)
    return (
        torch.randn(M, H),
        torch.randn(H, D),
        torch.randn(D),
        torch.randn(D, H),
        torch.randn(H),
    )


_ATTN_DIMS = dict(B=1, S=6, H_in=12, H=4, E=5)  # H_qkv=20, shard_d=10 @ W=2


def _iter_target(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


class TestNumaResidualTpPass(TestCase):
    def _trace(self, fn, args):
        return make_fx(fn, tracing_mode="fake")(*args)

    def _run_pipeline(self, gm, world_size, *, run_attention=False, run_mlp=False):
        with config.patch(
            {
                "_numa_attention_tp_world_size": world_size,
                "_numa_mlp_tp_world_size": world_size,
                "_numa_residual_tp": True,
            }
        ):
            if run_attention:
                numa_attention_tp_pass(gm.graph)
            if run_mlp:
                numa_mlp_tp_pass(gm.graph)
            numa_residual_tp_pass(gm.graph)
        gm.recompile()

    # ------------------------------------------------------------------
    # Attention block
    # ------------------------------------------------------------------
    def test_attention_residual_tagged(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        B, S, H_in, H, E = (
            _ATTN_DIMS["B"], _ATTN_DIMS["S"], _ATTN_DIMS["H_in"],
            _ATTN_DIMS["H"], _ATTN_DIMS["E"],
        )
        ws = 2
        args = _attn_inputs(B, S, H_in, H, E)
        gm = self._trace(_attn_residual_fn_factory(B, S, H, E), args)
        self._run_pipeline(gm, ws, run_attention=True)

        adds = _iter_target(gm, torch.ops.aten.add.Tensor)
        # Should be exactly one residual add carrying the residual_add plan.
        residual_adds = [
            a for a in adds
            if isinstance(a.meta.get("numa_shard_plan"), dict)
            and a.meta["numa_shard_plan"].get("kind") == "residual_add"
        ]
        self.assertEqual(
            len(residual_adds), 1,
            f"expected 1 residual_add tagged node, got {len(residual_adds)}: "
            f"all adds={[a.name for a in adds]}",
        )
        plan = residual_adds[0].meta["numa_shard_plan"]
        self.assertEqual(plan["world_size"], ws)
        self.assertEqual(plan["kind"], "residual_add")
        self.assertEqual(residual_adds[0].meta.get("numa_role"), "z_shard")
        # Plan must agree with upstream attention plan_id so codegen treats
        # both regions as a single contiguous TP plan.
        self.assertIn("plan_id", plan)
        self.assertIn("x_node", plan)

    # ------------------------------------------------------------------
    # MLP block
    # ------------------------------------------------------------------
    def test_mlp_residual_tagged(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        ws = 2
        # M, H, D — H_in divisible by ws, D divisible by ws.
        args = _mlp_inputs(8, 16, 32)
        gm = self._trace(_mlp_residual_fn, args)
        self._run_pipeline(gm, ws, run_mlp=True)

        adds = _iter_target(gm, torch.ops.aten.add.Tensor)
        residual_adds = [
            a for a in adds
            if isinstance(a.meta.get("numa_shard_plan"), dict)
            and a.meta["numa_shard_plan"].get("kind") == "residual_add"
        ]
        self.assertEqual(len(residual_adds), 1)
        self.assertEqual(residual_adds[0].meta.get("numa_role"), "z_shard")

    # ------------------------------------------------------------------
    # Idempotence + gating
    # ------------------------------------------------------------------
    def test_pass_disabled_when_flag_off(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        B, S, H_in, H, E = (
            _ATTN_DIMS["B"], _ATTN_DIMS["S"], _ATTN_DIMS["H_in"],
            _ATTN_DIMS["H"], _ATTN_DIMS["E"],
        )
        ws = 2
        args = _attn_inputs(B, S, H_in, H, E)
        gm = self._trace(_attn_residual_fn_factory(B, S, H, E), args)
        with config.patch(
            {
                "_numa_attention_tp_world_size": ws,
                "_numa_residual_tp": False,  # disabled
            }
        ):
            numa_attention_tp_pass(gm.graph)
            numa_residual_tp_pass(gm.graph)
        gm.recompile()

        adds = _iter_target(gm, torch.ops.aten.add.Tensor)
        for a in adds:
            plan = a.meta.get("numa_shard_plan")
            if isinstance(plan, dict):
                self.assertNotEqual(plan.get("kind"), "residual_add")

    def test_pass_idempotent(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        B, S, H_in, H, E = (
            _ATTN_DIMS["B"], _ATTN_DIMS["S"], _ATTN_DIMS["H_in"],
            _ATTN_DIMS["H"], _ATTN_DIMS["E"],
        )
        ws = 2
        args = _attn_inputs(B, S, H_in, H, E)
        gm = self._trace(_attn_residual_fn_factory(B, S, H, E), args)
        self._run_pipeline(gm, ws, run_attention=True)
        # Re-run residual pass — must remain a single residual_add tag.
        with config.patch({"_numa_residual_tp": True}):
            numa_residual_tp_pass(gm.graph)
        gm.recompile()
        adds = _iter_target(gm, torch.ops.aten.add.Tensor)
        residual_adds = [
            a for a in adds
            if isinstance(a.meta.get("numa_shard_plan"), dict)
            and a.meta["numa_shard_plan"].get("kind") == "residual_add"
        ]
        self.assertEqual(len(residual_adds), 1)

    # ------------------------------------------------------------------
    # Pre-LN cases (residual operand is the LN's INPUT, not the LN output).
    # ------------------------------------------------------------------
    def test_preLN_attention_residual_tagged(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        B, S, H_in, H, E = (
            _ATTN_DIMS["B"], _ATTN_DIMS["S"], _ATTN_DIMS["H_in"],
            _ATTN_DIMS["H"], _ATTN_DIMS["E"],
        )
        H_qkv = H * E
        ws = 2

        def fn(x, w_ln, b_ln, Wq, Wk, Wv, bq, bk, bv, Wo, bo):
            residual = x
            xn = F.layer_norm(x, (H_in,), w_ln, b_ln, 1e-5)
            q = torch.addmm(bq, xn, Wq).view(B, S, H, E).transpose(1, 2)
            k = torch.addmm(bk, xn, Wk).view(B, S, H, E).transpose(1, 2)
            v = torch.addmm(bv, xn, Wv).view(B, S, H, E).transpose(1, 2)
            scale = 1.0 / math.sqrt(E)
            scores = torch.matmul(q * scale, k.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B * S, H_qkv)
            z = torch.addmm(bo, y, Wo)
            return z + residual

        torch.manual_seed(0)
        args = (
            torch.randn(B * S, H_in), torch.randn(H_in), torch.randn(H_in),
        ) + _attn_inputs(B, S, H_in, H, E)[1:]
        gm = self._trace(fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": True,
                "_numa_layer_norm_tp_world_size": ws,
                "_numa_attention_tp_world_size": ws,
                "_numa_residual_tp": True,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
            numa_attention_tp_pass(gm.graph)
            numa_residual_tp_pass(gm.graph)
        gm.recompile()

        residual_adds = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.add.Tensor
            and isinstance(n.meta.get("numa_shard_plan"), dict)
            and n.meta["numa_shard_plan"].get("kind") == "residual_add"
        ]
        self.assertEqual(len(residual_adds), 1)
        # The matched residual operand must be the LN's input, NOT the LN's output.
        plan = residual_adds[0].meta["numa_shard_plan"]
        # x_node on the residual plan now records the actual residual operand.
        self.assertNotIn("layer_norm", plan["x_node"])

    def test_preLN_mlp_residual_tagged(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        ws = 2
        H_in, D = 16, 32

        def fn(x, w_ln, b_ln, A, b1, B_, b2):
            residual = x
            xn = F.layer_norm(x, (H_in,), w_ln, b_ln, 1e-5)
            h = torch.addmm(b1, xn, A)
            h = F.gelu(h)
            z = torch.addmm(b2, h, B_)
            return z + residual

        torch.manual_seed(0)
        args = (
            torch.randn(8, H_in), torch.randn(H_in), torch.randn(H_in),
            torch.randn(H_in, D), torch.randn(D),
            torch.randn(D, H_in), torch.randn(H_in),
        )
        gm = self._trace(fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": True,
                "_numa_layer_norm_tp_world_size": ws,
                "_numa_mlp_tp_world_size": ws,
                "_numa_residual_tp": True,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
            numa_mlp_tp_pass(gm.graph)
            numa_residual_tp_pass(gm.graph)
        gm.recompile()

        residual_adds = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.add.Tensor
            and isinstance(n.meta.get("numa_shard_plan"), dict)
            and n.meta["numa_shard_plan"].get("kind") == "residual_add"
        ]
        self.assertEqual(len(residual_adds), 1)

    def test_no_match_when_other_operand_is_not_x(self):
        """A trailing add whose other operand is unrelated to x_node must NOT
        be tagged."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        B, S, H_in, H, E = (
            _ATTN_DIMS["B"], _ATTN_DIMS["S"], _ATTN_DIMS["H_in"],
            _ATTN_DIMS["H"], _ATTN_DIMS["E"],
        )
        ws = 2
        H_qkv = H * E

        def fn(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo, junk):
            q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
            k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
            v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
            scale = 1.0 / math.sqrt(E)
            scores = torch.matmul(q * scale, k.transpose(-2, -1))
            attn = torch.softmax(scores, dim=-1)
            y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B * S, H_qkv)
            z = torch.addmm(bo, y, Wo)
            return z + junk  # not the residual x

        torch.manual_seed(0)
        args = _attn_inputs(B, S, H_in, H, E) + (torch.randn(B * S, H_in),)
        gm = self._trace(fn, args)
        self._run_pipeline(gm, ws, run_attention=True)

        adds = _iter_target(gm, torch.ops.aten.add.Tensor)
        for a in adds:
            plan = a.meta.get("numa_shard_plan")
            if isinstance(plan, dict):
                self.assertNotEqual(plan.get("kind"), "residual_add")


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
