# Owner(s): ["module: inductor"]
import math

import torch
from torch._inductor import config
from torch._inductor.fx_passes.numa_attention_tp import numa_attention_tp_pass
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


def _attn_fn_factory(B, S, H, E):
    """Functional attention block matching the pattern the pass targets.

    MATH-backend SDPA is manually decomposed (mul-scale + matmul + softmax +
    matmul) so the aten-level graph produced by ``make_fx`` contains the
    primitive ops the pass can walk through.
    """
    H_qkv = H * E

    def attn_fn(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo):
        q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
        scale = 1.0 / math.sqrt(E)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B * S, H_qkv)
        return torch.addmm(bo, out, Wo)

    return attn_fn


def _make_example_inputs(B, S, H_in, H, E):
    H_qkv = H * E
    torch.manual_seed(0)
    x = torch.randn(B * S, H_in)
    Wq = torch.randn(H_in, H_qkv)
    Wk = torch.randn(H_in, H_qkv)
    Wv = torch.randn(H_in, H_qkv)
    bq = torch.randn(H_qkv)
    bk = torch.randn(H_qkv)
    bv = torch.randn(H_qkv)
    Wo = torch.randn(H_qkv, H_in)
    bo = torch.randn(H_in)
    return (x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)


# All dims kept distinct so the shape rewriter can unambiguously find the
# head dim by literal match (B=1 is a common shape value; keep the rest
# distinct from each other and from h_qkv=H*E).
_DIMS = dict(B=1, S=6, H_in=12, H=4, E=5)  # H_qkv = 20, shard_d = 10 at N=2


def _iter_call_function(gm: torch.fx.GraphModule, target):
    return [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target == target
    ]


class TestNumaAttentionTpPass(TestCase):
    def _trace(self, fn, example_inputs):
        return make_fx(fn, tracing_mode="fake")(*example_inputs)

    def test_graph_rewrite_and_metadata(self):
        """Pass inserts per-shard slices and annotates plan/role metadata."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        B, S, H_in, H, E = _DIMS["B"], _DIMS["S"], _DIMS["H_in"], _DIMS["H"], _DIMS["E"]
        world_size = 2
        args = _make_example_inputs(B, S, H_in, H, E)
        gm_pre = self._trace(_attn_fn_factory(B, S, H, E), args)
        clones_before = len(_iter_call_function(gm_pre, torch.ops.aten.clone.default))
        gm = self._trace(_attn_fn_factory(B, S, H, E), args)

        with config.patch({"_numa_attention_tp_world_size": world_size}):
            numa_attention_tp_pass(gm.graph)
        gm.recompile()

        clone_nodes = _iter_call_function(gm, torch.ops.aten.clone.default)
        # Pass must not insert any new clone nodes (pre-existing clone from
        # .contiguous() in the traced graph is fine).
        self.assertEqual(
            len(clone_nodes),
            clones_before,
            f"Pass inserted {len(clone_nodes) - clones_before} new clone node(s); "
            "it must only emit slice nodes at the FX level.",
        )

        slice_nodes = _iter_call_function(gm, torch.ops.aten.slice.Tensor)
        # For each of Wq/Wk/Wv/Wo/bq/bk/bv we emit ``world_size`` slice nodes
        # (7 tensors x 2 sockets = 14 slice nodes).
        self.assertEqual(
            len(slice_nodes),
            7 * world_size,
            f"expected {7 * world_size} slice nodes, got {len(slice_nodes)}",
        )

        roles = [n.meta.get("numa_role") for n in gm.graph.nodes]
        self.assertEqual(roles.count("wq_slice"), world_size)
        self.assertEqual(roles.count("wk_slice"), world_size)
        self.assertEqual(roles.count("wv_slice"), world_size)
        self.assertEqual(roles.count("bq_slice"), world_size)
        self.assertEqual(roles.count("bk_slice"), world_size)
        self.assertEqual(roles.count("bv_slice"), world_size)
        self.assertEqual(roles.count("wo_slice"), world_size)
        self.assertEqual(roles.count("x_replica"), 1)
        self.assertEqual(roles.count("q_shard"), 1)
        self.assertEqual(roles.count("k_shard"), 1)
        self.assertEqual(roles.count("v_shard"), 1)
        self.assertEqual(roles.count("z_shard"), 1)

        z_shard = next(
            n for n in gm.graph.nodes if n.meta.get("numa_role") == "z_shard"
        )
        shard_plan = z_shard.meta.get("numa_shard_plan")
        self.assertIsNotNone(shard_plan)
        self.assertEqual(shard_plan["kind"], "attention")
        self.assertEqual(shard_plan["world_size"], world_size)
        H_qkv = H * E
        self.assertEqual(shard_plan["h_qkv"], H_qkv)
        self.assertEqual(shard_plan["shard_d"], H_qkv // world_size)
        self.assertEqual(shard_plan["num_heads"], H)
        self.assertEqual(shard_plan["head_dim"], E)
        # Column-parallel weights + biases sliced on dim 1 / 0.
        col_nodes = {name: dim for name, dim in shard_plan["column_parallel_nodes"]}
        # Expect 3 weights sliced dim=1, 3 biases sliced dim=0.
        self.assertEqual(sum(1 for d in col_nodes.values() if d == 1), 3)
        self.assertEqual(sum(1 for d in col_nodes.values() if d == 0), 3)
        row_nodes = {name: dim for name, dim in shard_plan["row_parallel_nodes"]}
        self.assertEqual(len(row_nodes), 1)
        self.assertEqual(next(iter(row_nodes.values())), 0)
        self.assertEqual(len(shard_plan["zero_on_nonzero_nodes"]), 1)

    def test_numerical_equivalence_compute_shard(self):
        """Rewritten graph returns compute-shard-0 partial output matching a
        hand-sliced MATH-SDPA reference."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        B, S, H_in, H, E = _DIMS["B"], _DIMS["S"], _DIMS["H_in"], _DIMS["H"], _DIMS["E"]
        world_size = 2
        args = _make_example_inputs(B, S, H_in, H, E)
        H_qkv = H * E
        shard_d = H_qkv // world_size
        shard_heads = H // world_size
        (x, Wq, Wk, Wv, bq, bk, bv, Wo, bo) = args

        # Hand-sliced reference: compute the compute-shard-0 contribution to
        # the row-parallel output using the same MATH decomposition, then add
        # the bias once (shard 0 is where the bias is kept non-zero).
        Wq0 = Wq[:, :shard_d]
        Wk0 = Wk[:, :shard_d]
        Wv0 = Wv[:, :shard_d]
        bq0 = bq[:shard_d]
        bk0 = bk[:shard_d]
        bv0 = bv[:shard_d]
        Wo0 = Wo[:shard_d, :]

        q0 = torch.addmm(bq0, x, Wq0).view(B, S, shard_heads, E).transpose(1, 2)
        k0 = torch.addmm(bk0, x, Wk0).view(B, S, shard_heads, E).transpose(1, 2)
        v0 = torch.addmm(bv0, x, Wv0).view(B, S, shard_heads, E).transpose(1, 2)
        scale = 1.0 / math.sqrt(E)
        scores0 = torch.matmul(q0 * scale, k0.transpose(-2, -1))
        attn0 = torch.softmax(scores0, dim=-1)
        y0 = (
            torch.matmul(attn0, v0)
            .transpose(1, 2)
            .contiguous()
            .view(B * S, shard_heads * E)
        )
        expected = torch.addmm(bo, y0, Wo0)

        gm = self._trace(_attn_fn_factory(B, S, H, E), args)
        with config.patch({"_numa_attention_tp_world_size": world_size}):
            numa_attention_tp_pass(gm.graph)
        gm.recompile()

        out = gm(*args)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_pass_noop_when_world_size_one(self):
        """world_size=1 is a pure no-op — no slices, no metadata."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        B, S, H_in, H, E = _DIMS["B"], _DIMS["S"], _DIMS["H_in"], _DIMS["H"], _DIMS["E"]
        args = _make_example_inputs(B, S, H_in, H, E)
        gm = self._trace(_attn_fn_factory(B, S, H, E), args)

        with config.patch({"_numa_attention_tp_world_size": 1}):
            numa_attention_tp_pass(gm.graph)
        gm.recompile()

        slice_nodes = _iter_call_function(gm, torch.ops.aten.slice.Tensor)
        # The module may have pre-existing slice/select nodes from aten
        # transposition; the pass itself should not have added attention-role
        # metadata to any.
        for n in gm.graph.nodes:
            self.assertNotIn(
                n.meta.get("numa_role"),
                ("wq_slice", "wk_slice", "wv_slice", "wo_slice", "q_shard", "z_shard"),
            )

    def test_view_shape_rewrite_on_preSDPA_views(self):
        """View nodes feeding SDPA are rewritten with H -> H // world_size."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        B, S, H_in, H, E = _DIMS["B"], _DIMS["S"], _DIMS["H_in"], _DIMS["H"], _DIMS["E"]
        world_size = 2
        args = _make_example_inputs(B, S, H_in, H, E)
        gm = self._trace(_attn_fn_factory(B, S, H, E), args)

        with config.patch({"_numa_attention_tp_world_size": world_size}):
            numa_attention_tp_pass(gm.graph)
        gm.recompile()

        # There should be at least one view.default node whose second arg is
        # a list containing the sharded head count (H // world_size = 2) and
        # head_dim (E = 4).
        rewritten = 0
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            if n.target not in (
                torch.ops.aten.view.default,
                torch.ops.aten._unsafe_view.default,
                torch.ops.aten.reshape.default,
            ):
                continue
            if len(n.args) < 2 or not isinstance(n.args[1], (list, tuple)):
                continue
            shape = tuple(n.args[1])
            if shape == (B, S, H // world_size, E):
                rewritten += 1
        self.assertGreaterEqual(rewritten, 3, "Expected 3 (Q/K/V) pre-SDPA views rewritten")


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
