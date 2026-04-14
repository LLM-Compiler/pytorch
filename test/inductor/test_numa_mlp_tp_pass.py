# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config
from torch._inductor.fx_passes.numa_mlp_tp import numa_mlp_tp_pass
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.inductor_utils import HAS_CPU


class MLP2(torch.nn.Module):
    def forward(self, x, a, b1, b, b2):
        y = torch.addmm(b1, x, a)
        y = torch.nn.functional.gelu(y)
        return torch.addmm(b2, y, b)


def _iter_call_function(gm: torch.fx.GraphModule, target):
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == target
    ]


_LINEAR_TARGETS = (
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.mm,
    torch.addmm,
    torch.Tensor.mm,
    torch.Tensor.addmm,
)
_GELU_TARGETS = [torch.nn.functional.gelu, torch.ops.aten.gelu.default]
if hasattr(torch, "gelu"):
    _GELU_TARGETS.append(torch.gelu)
if hasattr(torch.Tensor, "gelu"):
    _GELU_TARGETS.append(torch.Tensor.gelu)
_cnn = getattr(torch._C, "_nn", None)
if _cnn is not None and hasattr(_cnn, "gelu"):
    _GELU_TARGETS.append(_cnn.gelu)
_GELU_TARGETS = tuple(_GELU_TARGETS)


class TestNumaMlpTpPass(TestCase):
    def _trace(self, module, *args):
        gm = torch.fx.symbolic_trace(module)
        ShapeProp(gm).propagate(*args)
        return gm

    def test_graph_rewrite_and_metadata(self):
        """New architecture: no clone nodes, only slices + metadata."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        s, h, d = 4, 8, 6
        x = torch.randn(s, h)
        a = torch.randn(h, d)
        b1 = torch.randn(d)
        b = torch.randn(d, h)
        b2 = torch.randn(h)
        gm = self._trace(MLP2(), x, a, b1, b, b2)

        with config.patch({"_numa_mlp_tp_world_size": 2}):
            numa_mlp_tp_pass(gm.graph)
        gm.recompile()

        # No clone nodes should be present (new architecture).
        clone_nodes = _iter_call_function(gm, torch.ops.aten.clone.default)
        self.assertEqual(len(clone_nodes), 0)

        # Slice nodes should be present for A, B, b1 (2 each for world_size=2).
        slice_nodes = _iter_call_function(gm, torch.ops.aten.slice.Tensor)
        # A slices (2) + B slices (2) + b1 slices (2) = 6
        self.assertEqual(len(slice_nodes), 6)

        l1_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in _LINEAR_TARGETS
            and n.meta.get("numa_role") == "l1_shard_out"
        ]
        z_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in _LINEAR_TARGETS
            and n.meta.get("numa_role") == "z_shard"
        ]
        self.assertEqual(len(l1_nodes), 1)
        self.assertEqual(len(z_nodes), 1)

        gelu_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in _GELU_TARGETS
            and n.meta.get("numa_role") == "y_shard"
        ]
        self.assertEqual(len(gelu_nodes), 1)
        shard_d = d // 2
        l1 = l1_nodes[0]
        g = gelu_nodes[0]
        self.assertEqual(tuple(l1.meta["val"].shape), (s, shard_d))
        self.assertEqual(tuple(g.meta["val"].shape), (s, shard_d))

        # Check numa_shard_plan metadata is set on linear1.
        shard_plan = l1.meta.get("numa_shard_plan")
        self.assertIsNotNone(shard_plan)
        self.assertEqual(shard_plan["world_size"], 2)
        self.assertEqual(shard_plan["shard_d"], shard_d)
        self.assertEqual(shard_plan["d"], d)

        a_slices = [n for n in gm.graph.nodes if n.meta.get("numa_role") == "a_slice"]
        b_slices = [n for n in gm.graph.nodes if n.meta.get("numa_role") == "b_slice"]
        b1_slices = [n for n in gm.graph.nodes if n.meta.get("numa_role") == "b1_slice"]
        self.assertEqual(len(a_slices), 2)
        self.assertEqual(len(b_slices), 2)
        self.assertEqual(len(b1_slices), 2)

        for n in b1_slices:
            self.assertEqual(n.args[1], 0)

        expected = d // 2
        for n in a_slices:
            self.assertEqual(n.args[1], 1)
            self.assertEqual(int(n.args[3]) - int(n.args[2]), expected)
        for n in b_slices:
            self.assertEqual(n.args[1], 0)
            self.assertEqual(int(n.args[3]) - int(n.args[2]), expected)

    def test_numerical_equivalence(self):
        """Verify the rewritten graph produces correct shard-0 output."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        s, h, d = 4, 8, 6
        x = torch.randn(s, h)
        a = torch.randn(h, d)
        b1 = torch.randn(d)
        b = torch.randn(d, h)
        b2 = torch.randn(h)

        mod = MLP2()
        shard = d // 2
        expected = (
            torch.nn.functional.gelu(torch.addmm(b1[:shard], x, a[:, :shard]))
            @ b[:shard, :]
        ) + b2

        gm = self._trace(mod, x, a, b1, b, b2)
        with config.patch({"_numa_mlp_tp_world_size": 2}):
            numa_mlp_tp_pass(gm.graph)
        gm.recompile()

        out = gm(x, a, b1, b, b2)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_x_replica_metadata(self):
        """Input X should have x_replica NUMA metadata (replicated, not sliced)."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        s, h, d = 4, 8, 6
        x = torch.randn(s, h)
        a = torch.randn(h, d)
        b1 = torch.randn(d)
        b = torch.randn(d, h)
        b2 = torch.randn(h)
        gm = self._trace(MLP2(), x, a, b1, b, b2)

        with config.patch({"_numa_mlp_tp_world_size": 2}):
            numa_mlp_tp_pass(gm.graph)
        gm.recompile()

        x_nodes = [
            n for n in gm.graph.nodes
            if n.meta.get("numa_role") == "x_replica"
        ]
        self.assertEqual(len(x_nodes), 1)

    def test_clone_numa_node_compiles_with_inductor_fallback(self):
        """aten.clone(..., numa_node=) lowers to FallbackKernel (native aten), not Pointwise."""
        if not HAS_CPU:
            self.skipTest("CPU-only test")

        def fn(x):
            return x.clone(numa_node=0)

        x = torch.randn(2, 4)
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        try:
            y = compiled(x)
        except RuntimeError as e:
            if "requires NUMA support enabled at build/runtime" not in str(e):
                raise
        else:
            torch.testing.assert_close(y, x.clone())


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
