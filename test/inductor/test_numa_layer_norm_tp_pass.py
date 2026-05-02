# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config
from torch._inductor.fx_passes.numa_layer_norm_tp import numa_layer_norm_tp_pass
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


def _ln_fn(x, w, b):
    return torch.nn.functional.layer_norm(x, (16,), w, b, 1e-5)


def _make_inputs(M, H):
    torch.manual_seed(0)
    return torch.randn(M, H), torch.randn(H), torch.randn(H)


def _iter_target(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


class TestNumaLayerNormTpPass(TestCase):
    def _trace(self, fn, args):
        return make_fx(fn, tracing_mode="fake")(*args)

    def test_layer_norm_tagged(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        ws = 2
        args = _make_inputs(8, 16)
        gm = self._trace(_ln_fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": True,
                "_numa_layer_norm_tp_world_size": ws,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
        gm.recompile()

        ln_nodes = _iter_target(gm, torch.ops.aten.native_layer_norm.default)
        self.assertEqual(len(ln_nodes), 1)
        ln = ln_nodes[0]
        plan = ln.meta.get("numa_shard_plan")
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["kind"], "layer_norm")
        self.assertEqual(plan["world_size"], ws)
        self.assertEqual(ln.meta.get("numa_role"), "z_shard")
        # Weight + bias must be in replicated_nodes (full clones per socket).
        rep = set(plan["replicated_nodes"])
        self.assertIn(plan["weight_node"], rep)
        self.assertIn(plan["bias_node"], rep)
        self.assertIn(plan["x_node"], rep)
        # No column/row parallel weight slicing for LN.
        self.assertEqual(plan["column_parallel_nodes"], [])
        self.assertEqual(plan["row_parallel_nodes"], [])

    def test_pass_disabled_when_flag_off(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        args = _make_inputs(8, 16)
        gm = self._trace(_ln_fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": False,
                "_numa_layer_norm_tp_world_size": 2,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
        gm.recompile()
        ln_nodes = _iter_target(gm, torch.ops.aten.native_layer_norm.default)
        self.assertEqual(len(ln_nodes), 1)
        self.assertIsNone(ln_nodes[0].meta.get("numa_shard_plan"))

    def test_pass_idempotent(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        ws = 2
        args = _make_inputs(8, 16)
        gm = self._trace(_ln_fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": True,
                "_numa_layer_norm_tp_world_size": ws,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
            numa_layer_norm_tp_pass(gm.graph)  # second run must be a no-op
        gm.recompile()
        ln_nodes = _iter_target(gm, torch.ops.aten.native_layer_norm.default)
        # Plan is the same dict; key fact: idempotent (no double-tag, no error).
        self.assertEqual(len(ln_nodes), 1)
        plan = ln_nodes[0].meta.get("numa_shard_plan")
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["kind"], "layer_norm")

    def test_world_size_one_skips(self):
        if not HAS_CPU:
            self.skipTest("CPU-only test")
        args = _make_inputs(8, 16)
        gm = self._trace(_ln_fn, args)
        with config.patch(
            {
                "_numa_layer_norm_tp": True,
                "_numa_layer_norm_tp_world_size": 1,
            }
        ):
            numa_layer_norm_tp_pass(gm.graph)
        gm.recompile()
        ln_nodes = _iter_target(gm, torch.ops.aten.native_layer_norm.default)
        self.assertIsNone(ln_nodes[0].meta.get("numa_shard_plan"))


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
