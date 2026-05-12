# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import dis
import functools
import inspect
import logging
import operator
import re
import tempfile
from itertools import count
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import sympy
from sympy import Expr

import torch
import torch._ops
from torch import dtype as torch_dtype
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codegen.debug_utils import DebugPrinterManager
from torch._inductor.codegen.multi_kernel import MultiKernelState
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import async_compile, config, ir
from ..codecache import output_code_log
from ..ir import ReinterpretView
from ..runtime import triton_heuristics
from ..runtime.hints import DeviceProperties
from ..utils import (
    cache_on_self,
    DeferredLineBase,
    get_benchmark_name,
    LineContext,
    sympy_product,
    sympy_str,
)
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter


class _NumaReCloneLine(DeferredLineBase):
    """Emit a post-all-reduce per-node re-clone only if a downstream kernel
    actually consumed this tensor name. Tracking lives on the owning wrapper
    codegen via ``_numa_reclone_consumed``; names added there at emission
    time are kept, names never added are elided at flush time (saving a full
    tensor ``clone`` per socket per dead reduce output).
    """

    def __init__(self, line: str, name: str, consumed_set: "Set[str]") -> None:
        super().__init__(line)
        self.name = name
        self._consumed_set = consumed_set

    def __call__(self) -> Optional[str]:
        return self.line if self.name in self._consumed_set else None

    def _new_line(self, line: str) -> "_NumaReCloneLine":
        return _NumaReCloneLine(line, self.name, self._consumed_set)


class _NumaReassembleLine(DeferredLineBase):
    """Emit a per-socket M-shard ``copy_`` line only if ``out_arg`` was
    actually read by a non-NUMA consumer. Tracking lives on the owning
    wrapper via ``_numa_reassemble_consumed``; names added there during
    ``writeline`` are kept at flush time. Names absent at flush time are
    elided (saving a per-row-parallel-kernel full-output-size memcpy).
    """

    def __init__(self, line: str, name: str, consumed_set: "Set[str]") -> None:
        super().__init__(line)
        self.name = name
        self._consumed_set = consumed_set

    def __call__(self) -> Optional[str]:
        return self.line if self.name in self._consumed_set else None

    def _new_line(self, line: str) -> "_NumaReassembleLine":
        return _NumaReassembleLine(line, self.name, self._consumed_set)


class _NUMAFreshAllocLine(DeferredLineBase):
    """Emit a per-NUMA-node empty_strided allocation only if it was NOT
    superseded by a reinterpret alias propagated from a buffer-reuse chain.

    When ``wrapper._numa_suppressed_allocs`` contains ``name``, the allocation
    is skipped and the caller is responsible for having written an equivalent
    definition (a ``reinterpret_tensor`` alias) earlier in the output.
    """

    def __init__(self, line: str, name: str, suppressed_set: "Set[str]") -> None:
        super().__init__(line)
        self.name = name
        self._suppressed_set = suppressed_set

    def __call__(self) -> Optional[str]:
        return None if self.name in self._suppressed_set else self.line

    def _new_line(self, line: str) -> "_NUMAFreshAllocLine":
        return _NUMAFreshAllocLine(line, self.name, self._suppressed_set)


def _emit_numa_weight_cache(
    wrapper: "PythonWrapperCodegen",
    key: str,
    init_expr: str,
    var_name: str,
    suppressed_set: "Set[str]",
) -> None:
    """Emit a cache-guarded NUMA weight clone / output alloc as two
    independent single-line deferred writelines (so each is prefixed
    exactly once by ``IndentedBuffer.writeline``). If ``var_name`` is
    superseded by a buffer-reuse alias, both lines are suppressed.

    Emits::

        if '<key>' not in _numa_weight_cache: _numa_weight_cache['<key>'] = <init_expr>
        <var_name> = _numa_weight_cache['<key>']
    """
    line1 = (
        f"if {key!r} not in _numa_weight_cache:"
        f" _numa_weight_cache[{key!r}] = {init_expr}"
    )
    line2 = f"{var_name} = _numa_weight_cache[{key!r}]"
    wrapper.writeline(_NUMAFreshAllocLine(line1, var_name, suppressed_set))
    wrapper.writeline(_NUMAFreshAllocLine(line2, var_name, suppressed_set))


from .triton_utils import config_of, should_unwrap_unspec_arg, signature_to_meta


if TYPE_CHECKING:
    import triton

    from ..graph import GraphLowering


pexpr = PythonPrinter().doprint


ReuseKey = Tuple[torch.device, torch.dtype, str]


def buffer_reuse_key(node: ir.Buffer) -> ReuseKey:
    return (
        node.get_device(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(node.layout.storage_size())),
    )


def convert_arg_type(arg: torch.Argument) -> str:
    from .cpp import CONTAINER_PYTHON_TO_CPP, PYTHON_TO_CPP

    # use x.real_type instead of x.type so that we get ScalarType instead of int
    python_type = repr(arg.real_type)  # type: ignore[attr-defined]

    if python_type == "Tensor":
        # Conversions rules follow https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#func
        if arg.alias_info is not None and arg.alias_info.is_write:
            return f"at::{python_type}&"
        else:
            return f"at::{python_type} const&"

    if python_type in PYTHON_TO_CPP:
        cpp_type = PYTHON_TO_CPP[python_type]
        return cpp_type

    # Convert args of container types e.g. Optional[*]
    for py_container, cpp_container in CONTAINER_PYTHON_TO_CPP.items():
        container_match = re.findall(py_container + r"\[([a-zA-Z_]+)]", python_type)
        if len(container_match) == 1:
            contained_type = container_match[0]
            assert (
                contained_type in PYTHON_TO_CPP
            ), f"unsupported {py_container} type in convert_arg_type: {contained_type}"
            cpp_contained_type = PYTHON_TO_CPP[contained_type]
            return f"{cpp_container}<{cpp_contained_type}>"

    raise AssertionError(f"unsupport python_type: {python_type}")


def convert_return_type(ret: torch.Argument) -> str:
    # use x.real_type instead of x.type so that we get ScalarType instead of int
    python_type = repr(ret.real_type)  # type: ignore[attr-defined]
    python_to_cpp = {
        "Tensor": "at::Tensor",
        "List[Tensor]": "std::vector<at::Tensor>",
    }

    cpp_type = python_to_cpp.get(python_type, None)
    assert cpp_type is not None, f"NYI return type: {python_type}"
    # An output aliasing an input is returned by reference only when it's a
    # Tensor, not when it's a Tensor[]. For example, aten.split.Tensor's output
    # aliases the input tensor, but the op returns a vector by value.
    if python_type == "Tensor" and ret.alias_info is not None:
        cpp_type += "&"
    return cpp_type


def get_cpp_op_schema(kernel: torch._ops.OpOverload) -> str:
    args = kernel._schema.arguments
    returns = kernel._schema.returns

    num_returns = len(returns)
    assert num_returns > 0, "must have at least one return value"

    if num_returns == 1:
        cpp_return_value = convert_return_type(returns[0])
    elif num_returns > 1:
        tuple_returns = ", ".join([convert_return_type(r) for r in returns])
        cpp_return_value = f"std::tuple<{tuple_returns}>"

    cpp_arg_type = [f"{convert_arg_type(arg)} {arg.name}" for arg in args]
    return f"{cpp_return_value}({', '.join(cpp_arg_type)})"  # type: ignore[possibly-undefined]


# TODO: Move to a well known place
TritonMetaParams = Dict[str, int]
TritonGrid = Union[
    Tuple[Union[int, sympy.Expr], ...], Callable[[TritonMetaParams], Tuple[int, ...]]
]


def user_defined_kernel_grid_fn_code(
    name: str,
    configs: List[triton.Config],  # type: ignore[name-defined]
    grids: List[TritonGrid],
    wrapper: Optional[PythonWrapperCodegen] = None,
) -> Tuple[str, str]:
    output = IndentedBuffer()

    def _convert_to_sympy_expr(item: Union[int, sympy.Expr]) -> sympy.Expr:
        return item if isinstance(item, sympy.Expr) else sympy.Integer(item)

    def determine_grid(
        grid: TritonGrid,
    ):
        """
        This function return a tuple of two values: the first one is for the real grid
        which is used in the generated code; the second one is an example grid with
        concreate values which is used in the autotune block to run the generated
        kernels at compile time.
        """
        if wrapper is None or callable(grid):
            # return as-is when used in eager mode or when grid is callable
            return grid, grid
        # Grid contains ints/Expr, so utilize wrapper's expr printer for codegen
        sympy_grid = tuple(_convert_to_sympy_expr(g) for g in grid)
        return (
            wrapper.codegen_shape_tuple(sympy_grid),
            wrapper.codegen_shape_tuple(
                tuple(
                    wrapper.generate_example_arg_value(g, type(g)) for g in sympy_grid
                )
            )
            if config.triton.autotune_at_compile_time
            else None,
        )

    def writeline(line: str, example_grid: Optional[str] = None):
        output.writeline(line)
        if (
            wrapper
            and config.triton.autotune_at_compile_time
            and name not in wrapper.kernel_autotune_names
        ):
            wrapper.kernel_autotune_calls.writeline(example_grid or line)

    fn_name = f"grid_wrapper_for_{name}"
    writeline(f"def {fn_name}(meta):")
    kernel_autotune_calls_indent = (
        wrapper.kernel_autotune_calls.indent()
        if wrapper and config.triton.autotune_at_compile_time
        else contextlib.nullcontext()
    )
    with output.indent(), kernel_autotune_calls_indent:
        if len(grids) == 1:
            grid, example_grid = determine_grid(grids[0])
            writeline(f"return {grid}", f"return {example_grid}")
        else:
            assert len(grids) > 1
            assert len(grids) == len(configs)
            seen = set()
            for grid, c in zip(grids, configs):
                guards = [f"meta['{name}'] == {val}" for name, val in c.kwargs.items()]
                guards = " and ".join(guards)
                grid, example_grid = determine_grid(grid)
                statement = f"if {guards}: return {grid}"
                if statement in seen:
                    continue
                seen.add(statement)
                writeline(statement, f"if {guards}: return {example_grid}")

    return fn_name, output.getvalue()


@dataclasses.dataclass
class SymbolicCallArg:
    inner: str
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[
            ReuseKey, List[FreeIfNotReusedLine]
        ] = collections.defaultdict(list)
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key: ReuseKey) -> bool:
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        assert not item.is_reused
        self.reuse_pool[key].append(item)


class WrapperLine:
    pass


@dataclasses.dataclass
class EnterSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    graph: GraphLowering

    def __post_init__(self) -> None:
        self.wrapper.push_computed_sizes(self.wrapper.computed_sizes)

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.push_codegened_graph(self.graph)
        code.do_indent()


@dataclasses.dataclass
class ExitSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def __post_init__(self) -> None:
        self.wrapper.computed_sizes = self.wrapper.pop_computed_sizes()

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.pop_codegened_graph()
        code.do_unindent()


@dataclasses.dataclass
class EnterDeviceContextManagerLine(WrapperLine):
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer) -> None:
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if V.graph.aot_mode:
                # In AOT mode, we have a stream provided as a param. A stream is
                # associated with a device, so we never expect the device to change.
                # CUDAStreamGuard sets the stream and the device.
                if self.last_seen_device_guard_index is None:
                    if config.abi_compatible:
                        code.writeline(
                            f"{V.graph.device_ops.cpp_aoti_stream_guard()} stream_guard(stream, this->device_idx_);"
                        )
                    else:
                        code.writeline(
                            maybe_hipify_code_wrapper(
                                f"{V.graph.device_ops.cpp_stream_guard()} stream_guard("
                                + f"{V.graph.device_ops.cpp_getStreamFromExternal()}(stream, this->device_idx_));"
                            )
                        )
                else:
                    assert (
                        self.last_seen_device_guard_index == self.device_idx
                    ), "AOTInductor only supports running on one CUDA device"
            else:
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"{V.graph.device_ops.cpp_aoti_device_guard()} device_guard({self.device_idx});"
                        if config.abi_compatible
                        else maybe_hipify_code_wrapper(
                            f"{V.graph.device_ops.cpp_device_guard()} device_guard({self.device_idx});"
                        )
                    )
                else:
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))


class ExitDeviceContextManagerLine(WrapperLine):
    def codegen(self, code: IndentedBuffer) -> None:
        if not V.graph.cpp_wrapper:
            code.do_unindent()


@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        """Second pass to output code"""

    def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        args: List[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        if self.node.get_device().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: ir.Buffer
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: ir.Buffer
    reused_as: ir.Buffer
    delete_old: bool = True

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(
                self.node, self.reused_as, self.delete_old, code=code
            )
        )


class NullLine(MemoryPlanningLine):
    pass


BufferName = str


class PythonWrapperCodegen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter: Iterator[int] = count()
        self.imports = IndentedBuffer()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.kernel_autotune_defs = IndentedBuffer()
        self.kernel_autotune_calls = IndentedBuffer()
        self.kernel_autotune_names: Set[str] = set()
        # If the generated source code is exactly the same, reuse the
        # pre-existing kernel for it
        self.src_to_kernel: Dict[str, str] = {}
        self.kernel_numel_expr: Set[Tuple[str, GraphLowering]] = set()
        self.lines: List[Union[MemoryPlanningLine, LineContext]] = []
        self.declare = ""
        self.declare_maybe_reference = ""
        self.ending = ""
        self.open_bracket = "["
        self.closed_bracket = "]"
        self.comment = "#"
        self.namespace = ""
        self.none_str = "None"
        self.size = "size()"
        self.stride = "stride()"
        self.last_seen_device_guard_index: Optional[int] = None
        self.supports_intermediate_hooks = True
        self.expr_printer: Callable[[Any], str] = pexpr
        self.user_defined_kernel_cache: Dict[Tuple[Any, ...], Tuple[str, Any]] = {}
        self.unbacked_symbol_decls: Set[str] = set()  # str of sympy.Symbol
        self.computed_sizes: Set[sympy.Symbol] = set()
        # Maps a tensor variable name in wrapper code (e.g. "buf1") to a list of
        # per-NUMA-node tensor variables produced for that logical tensor.
        self._numa_tensor_cache: Dict[str, List[str]] = {}
        # Per-NUMA-node tensor variables whose post-all-reduce re-clone is
        # actually consumed by a downstream kernel. Names added here cause
        # the corresponding deferred re-clone line to be emitted at flush
        # time; names absent at flush time stay "unwritten". This elides
        # dead re-clone copies when the row-parallel output is terminal.
        self._numa_reclone_consumed: Set[str] = set()
        # Per-NUMA-node allocation variable names that have been superseded by
        # a reinterpret alias from a buffer-reuse chain. When a name appears
        # here, the corresponding _NUMAFreshAllocLine is suppressed at flush
        # time, preventing uninitialized memory from overriding the alias.
        self._numa_suppressed_allocs: Set[str] = set()
        # Row-parallel reassembly elision: candidates are ``out_arg`` names
        # whose per-socket M-shard copies are deferred pending a non-NUMA
        # consumer. When a later ``writeline`` references ``out_arg`` in a
        # non-NUMA context (any str line that does not contain the NUMA
        # kernel-call marker ``numa_args=``), the name is moved into
        # ``_numa_reassemble_consumed`` and the deferred copy_ lines are
        # kept at flush time. Names that stay absent cause the copies to
        # be elided entirely.
        self._numa_reassemble_candidates: Set[str] = set()
        self._numa_reassemble_consumed: Set[str] = set()
        # Whether the module-level ``_numa_weight_cache = {}`` dict has been
        # emitted to self.header yet. Set to True on first emit.
        self._numa_weight_cache_header_written: bool = False

        # this is used for tracking which GraphLowering instance---parent graph
        # or (nested) subgraph---is currently codegened; the primary use case is
        # including the graph instance into a cache key to avoid cross-graph
        # caching during lowering of nested subgraphs
        self.codegened_graph_stack = []
        self.computed_sizes_stack = []

        self.write_header()
        self.write_prefix()
        self.write_kernel_autotune_defs_header()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated: Set[BufferName] = set()
        self.freed: Set[BufferName] = set()

        # maps from reusing buffer to reused buffer
        self.reuses: Dict[BufferName, BufferName] = {}

        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

        @functools.lru_cache(None)
        def add_import_once(line: str) -> None:
            self.imports.writeline(line)
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(line)

        self.add_import_once = add_import_once
        self._metas: Dict[str, str] = {}
        self._meta_vars: Set[str] = set()
        self.multi_kernel_state = MultiKernelState()

        # intermediate tensor value printing utility
        self.debug_printer = DebugPrinterManager(
            debug_printer_level=config.aot_inductor.debug_intermediate_value_printer
        )

    def write_constant(self, name: str, hashed: str) -> None:
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self) -> None:
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        aot_inductor_debug_utils = ""
        if int(config.aot_inductor.debug_intermediate_value_printer) > 0:
            aot_inductor_debug_utils = "from torch._inductor.codegen.debug_utils import _print_debugging_tensor_value_info"
        
        # Check if extern_kernels is actually used in the generated code
        # For CPU code, we typically don't need extern_kernels
        extern_kernels_import = ""
        try:
            # Check if there are any external kernel calls in the graph
            has_extern_kernels = False
            if V.graph is not None:
                # Check scheduler for external kernel usage
                # This is a heuristic - if we're on CPU and using Triton, 
                # we likely don't need extern_kernels
                try:
                    device = V.graph.scheduler.get_current_device_or_throw()
                    # For CPU with Triton backend, extern_kernels is typically not needed
                    if device.type != "cpu":
                        has_extern_kernels = True
                except Exception:
                    # If we can't determine, include it for safety
                    has_extern_kernels = True
            else:
                # If graph is not available, include for safety
                has_extern_kernels = True
            
            if has_extern_kernels:
                extern_kernels_import = "from torch._inductor.select_algorithm import extern_kernels"
        except Exception:
            # If there's any error, include it for safety (backward compatibility)
            extern_kernels_import = "from torch._inductor.select_algorithm import extern_kernels"
        
        self.imports.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long, c_int
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.utils import empty_strided_cpu_numa
                from torch._inductor.codegen.memory_planning import _align as align
                from torch import device, empty_strided
                from {async_compile.__name__} import AsyncCompile
                {extern_kernels_import}
                from torch._inductor.codegen.multi_kernel import MultiKernelCall
                {aot_inductor_debug_utils}
            """,
            strip=True,
        )
        self.header.splice(
            """
                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()
            """,
            strip=True,
        )

    def include_extra_header(self, header: str):
        pass

    def write_kernel_autotune_defs_header(self) -> None:
        self.kernel_autotune_defs.splice(
            f"""
                import torch
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
            """
        )

    @cache_on_self
    def write_triton_header_once(self) -> None:
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
            """
        self.imports.splice(import_str, strip=True)
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
        self.write_get_raw_stream_header_once()

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        self.imports.writeline(
            V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        )
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    def add_meta_once(self, meta: TritonMetaParams) -> str:
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(f"{var} = {meta}")
                self._meta_vars.add(var)
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self) -> List[str]:
        return [x.codegen_reference(self.wrapper_call) for x in V.graph.graph_outputs]

    def mark_output_type(self) -> None:
        return

    def codegen_input_size_asserts(self) -> None:
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_shape_tuple(buf.get_size())
            stride = self.codegen_shape_tuple(buf.get_stride())
            self.prefix.writeline(f"assert_size_stride({name}, {size}, {stride})")

    def codegen_input_nan_asserts(self) -> None:
        self.prefix.writeline("# make sure graph inputs are not nan/inf")
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            line = f"assert not {name}.isnan().any().item()"
            self.prefix.writeline(line)
            line = f"assert not {name}.isinf().any().item()"
            self.prefix.writeline(line)

    def write_prefix(self) -> None:
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline(V.graph.device_ops.synchronize())
            if V.graph.graph_inputs:
                lhs = ", ".join(V.graph.graph_input_names)
                if len(V.graph.graph_input_names) == 1:
                    lhs += ","
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()
            if config.nan_asserts:
                self.codegen_input_nan_asserts()

    # this function (and below) takes a graph as input so
    # that stream caching happens per graph instance. this
    # is important for nested subgraph codegening.
    def write_get_raw_stream(self, device_idx: int, graph=None) -> str:
        self.write_get_raw_stream_header_once()
        name = f"stream{device_idx}"
        self.writeline(f"{name} = get_raw_stream({device_idx})")
        return name

    def get_codegened_graph(self):
        return self.codegened_graph_stack[-1]

    def push_codegened_graph(self, graph):
        self.codegened_graph_stack.append(graph)

    def pop_codegened_graph(self):
        return self.codegened_graph_stack.pop()

    def push_computed_sizes(self, computed_sizes):
        from copy import deepcopy

        return self.computed_sizes_stack.append(deepcopy(computed_sizes))

    def pop_computed_sizes(self):
        return self.computed_sizes_stack.pop()

    def next_kernel_suffix(self) -> str:
        return f"{next(self._names_iter)}"

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        self.writeline(
            EnterDeviceContextManagerLine(device_idx, self.last_seen_device_guard_index)
        )
        if config.triton.autotune_at_compile_time:
            # mimic logic of EnterDeviceContextManagerLine.codegen for the autotune code block
            self.write_triton_header_once()
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(device_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.set_device(device_idx)
            )
            self.kernel_autotune_calls.writeline(
                f"stream{device_idx} = get_raw_stream({device_idx})"
            )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self) -> None:
        self.writeline(ExitDeviceContextManagerLine())
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.do_unindent()

    def generate_return(self, output_refs: List[str]) -> None:
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_before_suffix(self, result: IndentedBuffer) -> None:
        return

    def generate_end(self, result: IndentedBuffer) -> None:
        return

    def generate_fallback_kernel(self, fallback_kernel, args):
        self.generate_extern_kernel_alloc(fallback_kernel, args)

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        # If it's a NoneLayout then the extern_kernel should essentially be
        # treated as if it doesn't return anything
        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            # view operation fallbacks cause issues since inductor
            # doesn't know the memory is still needed and might reuse it.
            ending = f".clone(){ending}"

        if no_return:
            self.writeline(f"{self.declare}{kernel_name}({', '.join(args)}){ending}")
        else:
            self.writeline(
                f"{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}"
            )
            if (
                self.supports_intermediate_hooks
                and config.generate_intermediate_hooks
                and origin_node is not None
            ):
                counters["inductor"]["intermediate_hooks"] += 1
                self.writeline(
                    f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
                )

    def generate_extern_kernel_out(
        self, kernel: str, out: str, out_view: Optional[str], args: List[str]
    ):
        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
        args.append(f"out={out_view if out_view else out}")
        with debug_printer_manager:
            self.writeline(f"{kernel}({', '.join(args)})")

    def generate_user_defined_triton_kernel(
        self,
        kernel_name: str,
        raw_args: List[Any],
        grid: List[Any],
        configs,
        triton_meta,
        constexprs,
    ):
        grid_fn, code = user_defined_kernel_grid_fn_code(
            kernel_name, configs, grid, wrapper=self
        )
        # Must happen after free symbols are already codegened
        # Emit the grid wrapper function right before the call
        for line in code.split("\n"):
            self.writeline(line)

        args = [self.val_to_arg_str(v) for v in raw_args]
        arg_types = [
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]
        self.generate_kernel_call(
            kernel_name, args, grid_fn=grid_fn, arg_types=arg_types, raw_args=raw_args
        )

    def generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
    ):
        line = f"{python_kernel_name}({','.join(map(str, inputs))}"
        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += ", ".join([""] + kwargs)
        else:
            if reduce:
                line += f", reduce={repr(reduce)}"
        line += ")"
        self.writeline(line)

    def generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        indices_str = f"{self.open_bracket}{', '.join(indices)}{self.closed_bracket}"
        args = [x, indices_str, values, accumulate]
        self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        cpp_op_schema: str,
        cpp_kernel_key: str,
        cpp_kernel_overload_name: str = "",
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        self.writeline(f"{buf_name} = {python_kernel_name}({', '.join(codegen_args)})")

    def generate(self, is_inference):
        with dynamo_timed("PythonWrapperCodegen.generate"):
            return self._generate(is_inference)

    def _generate(self, is_inference):
        if config.profile_bandwidth:
            self.write_triton_header_once()
        result = IndentedBuffer()
        result.splice(self.imports)
        result.writeline("")
        result.splice(self.header)
        # We do not want the cpp header for intermediate const graph. Headers would be
        # rendered by the main module instead.
        if V.graph.aot_mode and V.graph.cpp_wrapper and V.graph.is_const_graph:
            result = IndentedBuffer()

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()

            # We disable planning during training because it presently increases peak memory consumption.
            if is_inference and config.memory_planning:
                self.memory_plan()
            else:
                self.memory_plan_reuse()

            if config.triton.store_cubin:
                self.generate_reset_kernel_saved_flags()

            for line in self.lines:
                if isinstance(line, WrapperLine):
                    line.codegen(self.wrapper_call)
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline(V.graph.device_ops.synchronize())

            if config.profile_bandwidth:
                self.generate_end_graph()

            if config.triton.store_cubin:
                self.generate_save_uncompiled_kernels()

            if config.triton.autotune_at_compile_time:
                self.generate_and_run_autotune_block()

            self.generate_return(output_refs)

        self.finalize_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_before_suffix(result)
        result.splice(self.suffix)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def generate_and_run_autotune_block(self):
        """
        Compose self.kernel_autotune_defs and self.kernel_autotune_calls into a single block of
        code and execute it to trigger Triton kernel compilation and auto-tuning
        """
        self.kernel_autotune_defs.splice(
            """
            async_compile.wait(globals())
            del async_compile
        """
        )
        scope = {}  # type: ignore[var-annotated]
        tuning_code = (
            self.kernel_autotune_defs.getvalue() + self.kernel_autotune_calls.getvalue()
        )
        if output_code_log.level == logging.DEBUG:
            # Save the autotuning code block into a file
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                dir=cache_dir(), suffix=".py", delete=False
            ) as f:
                f.write(tuning_code.encode("utf-8"))
                file_path = f.name
            output_code_log.debug(
                "\nCompile-time auto-tuning code: \n%s\nAuto-tuning code written to %s",
                tuning_code,
                file_path,
            )
        # Execute the code to autotune kernels
        exec(tuning_code, scope)

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        assert len(planning_states) == 0

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"{self.declare}{name}_size = {name}.{self.size}{self.ending}")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(
            f"{self.declare}{name}_stride = {name}.{self.stride}{self.ending}"
        )

    def codegen_inputs(
        self, code: IndentedBuffer, graph_inputs: Dict[str, ir.TensorBox]
    ):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        # Assign all symbolic shapes needed to local variables
        bound_vars: Set[sympy.Symbol] = set()

        def is_expr(x):
            return isinstance(x[1], sympy.Expr)

        graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
        graph_inputs_tensors = list(
            filter(lambda x: not is_expr(x), graph_inputs.items())
        )

        for name, shape in graph_inputs_expr:
            if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                code.writeline(f"{self.declare}{shape} = {name}{self.ending}")
                bound_vars.add(shape)

        for name, value in graph_inputs_tensors:
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                    code.writeline(
                        f"{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}"
                    )
                    bound_vars.add(shape)

        for name, value in graph_inputs_tensors:
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                if isinstance(shape, sympy.Symbol) and shape not in bound_vars:
                    code.writeline(
                        f"{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}"
                    )
                    bound_vars.add(shape)

    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(
                f"{self.declare}{sym} = {self.expr_printer(expr)}{self.ending}"
            )

    def finalize_prefix(self):
        pass

    def codegen_python_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        return pexpr(x, simplify=simplify)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        return f"{basename}[{index}]"

    def codegen_python_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_python_sizevar, shape))
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        return self.codegen_python_shape_tuple(shape)

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    str(dtype),
                    self.codegen_shape_tuple(shape),
                    self.codegen_shape_tuple(stride),
                ]
            )
        )

    def codegen_reinterpret_view(
        self, data, size, stride, offset, writer, dtype=None
    ) -> str:
        if (
            size == data.layout.size
            and stride == data.layout.stride
            and offset == data.layout.offset
        ):
            if dtype is not None and dtype != data.dtype:
                return f"aten.view.dtype({data.get_name()}, {dtype})"
            else:
                return f"{data.get_name()}"
        else:
            size = self.codegen_shape_tuple(size)
            stride = self.codegen_shape_tuple(stride)
            offset = self.codegen_sizevar(offset)
            if dtype is not None and dtype != data.dtype:
                return f"aten.view.dtype(reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset}), {dtype})"
            else:
                return (
                    f"reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset})"
                )

    def codegen_device_copy(self, src, dst):
        self.writeline(f"{dst}.copy_({src})")

    def codegen_multi_output(self, name, value):
        self.writeline(f"{self.declare}{name} = {value}{self.ending}")

    def codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        if len(node.keypath) == 0:
            self.writeline(f"{node.sym} = {data}.item()")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"{node.sym} = 1 if {data}.item() else 0")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            self.writeline(f"{node.sym}_undivided = {data}.item()")
            self.writeline(
                f"assert {node.sym}_undivided % {node.keypath[0].divisor} == 0, "
                f"f'{{{node.sym}_undivided}} not divisible by {node.keypath[0].divisor}'"
            )
            self.writeline(
                f"{node.sym} = {node.sym}_undivided // {node.keypath[0].divisor}"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")
        # No one should ever use this buffer, but for uniformity
        # define the variable and assign it None
        self.writeline(f"{node.get_name()} = None")

    def benchmark_compiled_module(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        def add_torchbind_input(name, value):
            import pickle

            output.writeline(f"{name} = pickle.loads({pickle.dumps(value)!r})")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            if len(V.graph.torchbind_constants) > 0:
                output.writeline("import pickle")
                for name, torchbind_obj in V.graph.torchbind_constants.items():
                    # all the constants are global variables, that's why we need
                    # these 'global var_name' lines
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, torchbind_obj)

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    # TODO: this fallback and those below actually will generate possibly
                    # invalid benchmark code, because it's not guaranteed 42
                    # is actually a valid value for the kernel in question.
                    # See https://github.com/pytorch/pytorch/issues/124686
                    add_expr_input(name, V.graph.sizevars.size_hint(value, fallback=42))
                else:
                    shape = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_size()
                    ]
                    stride = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_stride()
                    ]
                    add_fake_input(
                        name,
                        shape,
                        stride,
                        value.get_device(),
                        value.get_dtype(),
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline("return print_performance(fn, times=times, repeat=repeat)")

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        self.benchmark_compiled_module(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(
                [
                    "from torch._inductor.wrapper_benchmark import compiled_module_main",
                    f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)",
                ]
            )

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, gpu=True
    ):
        metadata_comment = f"{metadata}\n" if metadata else ""
        body = f"\n\n{metadata_comment}{name} = {kernel}"
        self.header.splice(body)
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_defs.splice(body)

    def define_user_defined_triton_kernel(self, kernel, configs, kwargs):
        from torch.utils._triton import patch_triton_dtype_repr

        patch_triton_dtype_repr()

        original_name = kernel.__name__

        from .common import KernelArgType, SizeArg, TensorArg

        signature: List[KernelArgType] = []
        constants: Dict[str, Any] = {}
        non_constant_indices = []
        equal_to_1_args: List[str] = []
        for idx, key in enumerate(kernel.arg_names):
            if key not in kwargs:
                continue
            arg = kwargs[key]
            if idx in kernel.constexprs:
                constants[key] = arg
            else:
                non_constant_indices.append(idx)
                if isinstance(arg, ir.Buffer):
                    signature.append(
                        TensorArg(
                            name=key,
                            buffer=arg.get_name(),
                            dtype=arg.get_dtype(),
                        )
                    )
                elif isinstance(arg, ir.ReinterpretView):
                    # for ReinterpretView we use the underlying
                    # buffer name and note the (possibly non-zero)
                    # offset relative to the underlying buffer
                    signature.append(
                        TensorArg(
                            name=key,
                            buffer=arg.data.get_name(),
                            dtype=arg.get_dtype(),
                            offset=arg.layout.offset,
                        )
                    )
                else:
                    signature.append(SizeArg(key, arg))
                    if isinstance(
                        arg, (int, sympy.Integer)
                    ) and V.graph.sizevars.statically_known_equals(
                        arg, 1  # type: ignore[arg-type]
                    ):
                        equal_to_1_args.append(key)
        index_dtype = "tl.int32"
        triton_meta = {
            "signature": signature_to_meta(
                signature,
                size_dtype=index_dtype,
                indices=non_constant_indices,
                argdefs=kernel.arg_names,
            ),
            "device": DeviceProperties.create(
                V.graph.scheduler.get_current_device_or_throw()
            ),
            # Triton compiler includes equal_to_1 args into constants even
            # when they are not constexpr. otherwise there may be a segfault
            # during launching the Inductor-compiled Triton kernel.
            # TODO(aakhundov): add None args to constants, too. currently, this
            # causes CUDA errors in test_aot_inductor.test_triton_kernel_with_none_input.
            # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
            # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
            "constants": {
                **constants,
                **dict.fromkeys(equal_to_1_args, 1),
            },
            "configs": [
                config_of(
                    signature,
                    indices=non_constant_indices,
                )
            ],
        }

        # Distinguish between different functions using function id
        cache_key: List[Any] = [id(kernel.fn)]
        if len(configs) > 0:
            for arg in kwargs.values():
                # We need to key on non tensor arg only in autotune mode
                if not isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                    cache_key.append(arg)
        cache_key.append(str(triton_meta))
        cache_key = tuple(cache_key)

        if cache_key in self.user_defined_kernel_cache:
            return self.user_defined_kernel_cache[cache_key]

        name = f"{original_name}_{len(self.user_defined_kernel_cache)}"
        # Add to the cache for the next use
        self.user_defined_kernel_cache[cache_key] = (name, triton_meta)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")

        from .triton import gen_common_triton_imports, TritonKernel

        compile_wrapper.splice(gen_common_triton_imports())

        inductor_meta = {
            "kernel_name": name,
            **TritonKernel.inductor_meta_common(),
        }

        configs = [
            {
                "kwargs": config.kwargs,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
            }
            for config in configs
        ]

        compile_wrapper.splice(
            f"""
            @triton_heuristics.user_autotune(
                configs={configs!r},
                inductor_meta={inductor_meta!r},
                triton_meta={triton_meta!r},
                filename=__file__,
                custom_kernel=True,
            )
            @triton.jit
            """
        )
        compile_wrapper.splice(kernel.src, strip=True)

        # Also include any possible kernel being called indirectly
        from triton import JITFunction  # type: ignore[name-defined, attr-defined]
        from triton.language import constexpr  # type: ignore[name-defined]

        # global constexpr vars handled above
        symbols_included = {original_name}

        def traverse(cur_kernel):
            # here we extract the unqualified names (i.e., not attributes and
            # without prepended module name) loaded in the kernel code, which
            # are matched with the co_names and __globals__ below to codegen
            # the respective imports necessary for the kernel compilation
            unqualified_loads = {
                inst.argval
                for inst in dis.Bytecode(cur_kernel.fn)
                if inst.opname == "LOAD_GLOBAL"
            }
            global_annotations = cur_kernel.fn.__globals__.get("__annotations__", {})
            for symbol_name in cur_kernel.fn.__code__.co_names:
                if symbol_name in symbols_included:
                    continue
                if symbol_name in cur_kernel.fn.__globals__:
                    symbol = cur_kernel.fn.__globals__[symbol_name]
                    if isinstance(symbol, JITFunction):
                        compile_wrapper.newline()
                        compile_wrapper.writeline("@triton.jit")
                        compile_wrapper.splice(symbol.src, strip=True)
                        symbols_included.add(symbol_name)
                        traverse(symbol)
                    elif isinstance(symbol, (int, str, bool, constexpr)):
                        compile_wrapper.newline()
                        if isinstance(symbol, constexpr):
                            symbol_str = f"tl.constexpr({symbol.value!r})"
                        else:
                            symbol_str = f"{symbol!r}"
                        if annotation := global_annotations.get(symbol_name):
                            annotion_code = ""
                            if isinstance(annotation, type):
                                annotation_code = (
                                    f": {annotation.__module__}.{annotation.__name__}"
                                )
                            else:
                                annotation_code = f": {annotation!r}"
                            compile_wrapper.writeline(
                                f"{symbol_name}{annotation_code} = {symbol_str}"
                            )
                        else:
                            compile_wrapper.writeline(f"{symbol_name} = {symbol!r}")
                        symbols_included.add(symbol_name)
                    elif (
                        symbol_name in unqualified_loads
                        and symbol_name != "tl"  # already imported
                        and hasattr(symbol, "__module__")
                        # only codegen imports from triton; JITFunctions
                        # imported from other modules will be codegened
                        # in the separate branch above
                        and symbol.__module__.startswith("triton")
                    ):
                        # a global symbol imported from triton is referenced
                        # without module qualification (i.e., `store` instead
                        # of `tl.store`): need to codegen an import
                        compile_wrapper.writeline(
                            f"from {symbol.__module__} import {symbol.__name__} as {symbol_name}"
                        )
                        symbols_included.add(symbol_name)

        traverse(kernel)

        current_device = V.graph.scheduler.get_current_device_or_throw()
        compile_wrapper.writeline(f"''', device_str='{current_device.type}')")
        _, lineno = inspect.getsourcelines(kernel.fn)
        srcfile = inspect.getsourcefile(kernel.fn)
        metadata = f"# Original path: {srcfile}:{lineno}"
        self.define_kernel(
            name,
            compile_wrapper.getvalue(),
            metadata,
        )
        return name, triton_meta

    def generate_numel_expr(self, kernel_name: str, tree, suffix: Optional[str] = None):
        expr = f"{kernel_name}_{tree.prefix}numel"
        if suffix is not None:
            expr += f"_{suffix}"
        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(
                f"{self.declare}{expr} = {self.expr_printer(tree.numel)}{self.ending}"
            )
        else:
            self.writeline(f"{expr} = {self.expr_printer(tree.numel)}{self.ending}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, tree.numel)

    def generate_workspace_allocation(self, nbytes, device, zero_fill):
        if isinstance(nbytes, sympy.Expr):
            nbytes = V.graph.sizevars.size_hint(nbytes)
        line = self.make_allocation(
            "workspace", device, torch.uint8, shape=(nbytes,), stride=(1,)
        )
        self.writeline(line)
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(line)
        if zero_fill:
            self.writeline(f"workspace.zero_(){self.ending}")
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(f"workspace.zero_(){self.ending}")

    def wrap_kernel_call(self, name, call_args):
        return f"{name}({', '.join(call_args)}){self.ending}"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        stack.enter_context(self.wrapper_call.indent())

    def generate_start_graph(self):
        self.wrapper_call.writeline("start_graph()")

    def generate_end_graph(self):
        self.wrapper_call.writeline(f"end_graph({config.profile_bandwidth_output!r})")

    def generate_reset_kernel_saved_flags(self):
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    kernel.cuda_kernel_saved = False
            """
        )

    def generate_save_uncompiled_kernels(self):
        """
        Precompile and save the CUBINs of the Triton kernels that haven't
        been precompiled and saved as a side effect of running the generated
        JIT model (Python wrapper). This can happen when the model contains
        control flow: only one pass through the control flow operators covers
        the kernels that are saved, the remaining kernels are not launched,
        hence not saved. The main purpose of this codegen is to compile and
        save the Triton kernels outside the active control flow path for
        subsequent AOTInductor code generation and compilation.
        """
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    if not kernel.cuda_kernel_saved:
                        if len(kernel.launchers) == 0:
                            kernel.precompile()
                        kernel.save_gpu_kernel(
                            grid=(0, 0, 0),   # use dummy grid
                            stream="stream",  # use dummy stream
                            launcher=kernel.launchers[0],
                        )
            """
        )

    def generate_default_grid(
        self,
        kernel_name: str,
        grid: List[Any],
        gpu: bool = True,
        grid_callable: Optional[Callable[..., Any]] = None,
        **grid_extra_kwags,
    ):
        return grid

    def prepare_triton_kernel_call(self, device_index, call_args):
        def wrap_arg(arg):
            if isinstance(arg, str):
                # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                return arg + ".item()" if should_unwrap_unspec_arg(arg) else arg
            elif isinstance(arg, (int, float, bool, SymbolicCallArg)):
                return str(arg)
            else:
                return self.expr_printer(V.graph.sizevars.simplify(arg))

        call_args = [wrap_arg(arg) for arg in call_args]

        if device_index is None:
            current_device = V.graph.scheduler.get_current_device_or_throw()
            device_index = current_device.index

        return device_index, call_args

    def generate_example_arg_value(self, arg, arg_type, raw_arg=None, index=None):
        if isinstance(arg_type, torch_dtype):
            if V.graph.try_get_buffer(arg) is not None:
                buf_name = arg
                buf = V.graph.get_buffer(arg)
            else:
                assert (
                    raw_arg is not None
                ), "V.graph.get_buffer(arg) and raw_arg can't be None at the same time"
                buf_name = f"tmp_arg_{index}"
                buf = raw_arg

            size = V.graph.sizevars.size_hints(
                buf.get_size(),
                fallback=config.unbacked_symint_fallback,
            )
            stride = V.graph.sizevars.size_hints(
                buf.get_stride(),
                fallback=config.unbacked_symint_fallback,
            )
            device = buf.get_device()
            dtype = buf.get_dtype()
            offset = V.graph.sizevars.size_hint(
                buf.layout.offset,
                fallback=config.unbacked_symint_fallback,
            )
            value = f"generate_example_value({size}, {stride}, '{device}', {dtype}, {offset})"
            self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")
            return buf_name
        elif issubclass(arg_type, sympy.Basic) or isinstance(arg, SymbolicCallArg):
            # arg is a symbol or symbolic expression
            if isinstance(arg, str):
                if arg in self._meta_vars:
                    return arg
                if raw_arg is None:
                    return "None"
                arg = raw_arg
            if isinstance(arg, SymbolicCallArg):
                arg = arg.inner_expr
            if arg in V.graph.sizevars.inv_precomputed_replacements:
                arg = V.graph.sizevars.inv_precomputed_replacements[arg]
            return str(
                V.graph.sizevars.size_hint(
                    arg,
                    fallback=config.unbacked_symint_fallback,
                )
            )
        elif isinstance(arg, (str, int, float, bool)):
            return str(arg)
        elif isinstance(arg, list):
            return f"[{', '.join(self.generate_example_arg_value(a, type(a)) for a in arg)}]"
        else:
            raise NotImplementedError(f"Unsupported type {type(arg)}")

    def _grid_dim_str(self, grid_per_dim):
        if isinstance(grid_per_dim, list):
            return (
                "[" + ", ".join(self._grid_dim_str(item) for item in grid_per_dim) + "]"
            )
        else:
            return pexpr(grid_per_dim)

    def _ensure_numa_weight_cache_header(self) -> None:
        """Emit a module-level ``_numa_weight_cache = {}`` dict once per
        compiled graph. Shared across all forward calls to cache per-NUMA-node
        weight clones and pre-allocated intermediate buffers."""
        if not self._numa_weight_cache_header_written:
            self.header.writeline("_numa_weight_cache = {}")
            self._numa_weight_cache_header_written = True

    def generate_kernel_call(
        self,
        kernel_name,
        call_args,
        grid=None,
        device_index=None,
        gpu=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
        numa_shard_plan: Optional[Dict[str, Any]] = None,
        numa_role: Optional[str] = None,
    ):
        """
        Generates kernel call code.

        gpu: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the backend uses Triton for codegen. Otherwise it uses the CUDA language when gpu=True,
                and C++ when gpu=False.
        """
        if not (triton or gpu):
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            return

        device_index, call_args_str = self.prepare_triton_kernel_call(
            device_index, call_args
        )
        call_args_str = ", ".join(call_args_str)
        stream_name = self.write_get_raw_stream(device_index, V.graph)

        if not triton:
            stream_ptr = f"c_void_p({stream_name})"
            self.writeline(
                f"{kernel_name}.{kernel_name}({call_args_str}, {stream_ptr})"
            )
            return

        self.write_triton_header_once()
        if grid is None:
            grid_str = grid_fn
        else:
            grid_str = ", ".join(self._grid_dim_str(item) for item in grid)
            if grid_extra_kwargs:
                grid_str = f"{grid_str}, {grid_extra_kwargs}"
            grid_str = f"{grid_fn}({grid_str})"
        # Check if this kernel has a NUMA shard plan (set by numa_mlp_tp_pass).
        # If so, emit per-NUMA-node buffer clones and pass them as numa_args.
        numa_args_var = None
        numa_output_clones: Dict[str, List[str]] = {}
        numa_per_node_call_args: List[str] = []
        # Fallback: template kernels may lose direct origin-node metadata during
        # scheduling. If NUMA TP is enabled globally, still emit NUMA args for
        # tensor pointer arguments.
        if numa_shard_plan is None and (
            getattr(config, "_numa_mlp_tp", False)
            or getattr(config, "_numa_attention_tp", False)
        ):
            full_plan = getattr(config, "_numa_attention_tp_shard_plan", None)
            if not (isinstance(full_plan, dict) and full_plan.get("world_size", 0) >= 2):
                full_plan = getattr(config, "_numa_mlp_tp_shard_plan", None)
            if isinstance(full_plan, dict) and full_plan.get("world_size", 0) >= 2:
                numa_shard_plan = full_plan
            else:
                ws = int(getattr(config, "_numa_mlp_tp_world_size", 0))
                if ws <= 1:
                    ws = int(getattr(config, "_numa_attention_tp_world_size", 0))
                if ws <= 1:
                    cpu_mod = getattr(torch._C, "_cpu", None)
                    if (
                        cpu_mod is not None
                        and hasattr(cpu_mod, "_is_numa_enabled")
                        and hasattr(cpu_mod, "_num_numa_nodes")
                        and cpu_mod._is_numa_enabled()
                    ):
                        ws = int(cpu_mod._num_numa_nodes())
                if ws >= 2:
                    numa_shard_plan = {"world_size": ws, "plan_id": 0}

        if (
            numa_shard_plan is not None
            and numa_shard_plan.get("world_size", 0) >= 2
            and arg_types is not None
        ):
            (
                numa_args_var,
                numa_output_clones,
                numa_per_node_call_args,
            ) = self.generate_numa_shard_allocations(
                numa_shard_plan, call_args, arg_types, triton_meta, numa_role
            )

        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(call_args, kernel_name, arg_types, None)
        with debug_printer_manager:
            if numa_args_var is not None:
                self.writeline(
                    f"{kernel_name}.run({call_args_str}, grid={grid_str}, "
                    f"stream={stream_name}, numa_args={numa_args_var})"
                )
                # Row-parallel output (z_shard) needs a sum reduction across
                # per-node partial outputs. Then materialize the reduced result
                # back into every per-node output clone.
                should_allreduce_outputs = numa_role == "z_shard"
                if not should_allreduce_outputs and numa_shard_plan is not None:
                    # Fallback for paths that lose role metadata: row-parallel
                    # output kernels consume a row-parallel weight tensor and
                    # do NOT consume any column-parallel tensor. Check both the
                    # legacy ``a_node``/``b_node`` fields and the generalized
                    # ``column_parallel_nodes`` / ``row_parallel_nodes`` lists.
                    from torch._inductor.fx_passes.numa_tp_common import (
                        ensure_generalized_lists,
                    )

                    ngen = ensure_generalized_lists(numa_shard_plan)
                    # Entries are 2-tuples (name, dim) or 3-tuples (name, dim,
                    # shard_size). We only need the name here.
                    col_names = {e[0] for e in ngen.get("column_parallel_nodes", [])}
                    row_names = {e[0] for e in ngen.get("row_parallel_nodes", [])}
                    has_row = any(n in call_args for n in row_names)
                    has_col = any(n in call_args for n in col_names)
                    should_allreduce_outputs = has_row and not has_col
                if should_allreduce_outputs:
                    for out_arg, per_node_outs in numa_output_clones.items():
                        if len(per_node_outs) < 2:
                            continue
                        ws = len(per_node_outs)
                        # M-split reassembly: the Triton-CPU launcher splits the
                        # kernel grid linearly across sockets in row-major
                        # (M-major) tile order.  Socket ``i`` therefore owns the
                        # contiguous row shard [i * M//W : (i+1) * M//W] of every
                        # output buffer.  Rather than summing per-socket outputs
                        # (which would double-count rows), we concatenate the
                        # valid shard from each socket along dim-0 to reconstruct
                        # the full output tensor.
                        #
                        # The per-socket kernel outputs stay in _numa_tensor_cache
                        # so downstream NUMA kernels still access their own
                        # M-shard directly — no broadcast clone needed.
                        split_var = f"_numa_msplit_{out_arg}"
                        first_out = per_node_outs[0]
                        # In-place reassembly into the already-allocated
                        # out_arg buffer: each socket copies its M-shard
                        # into the corresponding slice. Avoids the per-
                        # forward allocation that ``torch.cat`` would do.
                        # Wrapped in ``_NumaReassembleLine`` so the whole
                        # reassembly is elided when ``out_arg`` has no
                        # non-NUMA consumer (downstream NUMA kernels read
                        # per-socket M-shards from ``_numa_tensor_cache``
                        # directly and never reference ``out_arg`` as a
                        # plain Python var).
                        self._numa_reassemble_candidates.add(out_arg)
                        consumed_set = self._numa_reassemble_consumed
                        self.writeline(
                            _NumaReassembleLine(
                                f"{split_var} = {first_out}.shape[0] // {ws}",
                                out_arg,
                                consumed_set,
                            )
                        )
                        for sid in range(ws):
                            start_s = f"{split_var}*{sid}" if sid > 0 else ""
                            end_s = f"{split_var}*{sid + 1}" if sid < ws - 1 else ""
                            self.writeline(
                                _NumaReassembleLine(
                                    f"{out_arg}[{start_s}:{end_s}].copy_("
                                    f"{per_node_outs[sid]}[{start_s}:{end_s}])",
                                    out_arg,
                                    consumed_set,
                                )
                            )
                        # Keep per-socket kernel outputs in cache so downstream
                        # NUMA kernels receive their correct M-shard without
                        # an extra clone or re-broadcast.
                        self._numa_tensor_cache[out_arg] = per_node_outs
            else:
                self.writeline(
                    f"{kernel_name}.run({call_args_str}, grid={grid_str}, stream={stream_name})"
                )
        if (
            config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Create example args for autotune in a separate epilogue
            assert arg_types is not None and len(call_args) == len(
                arg_types
            ), "call_args and arg_types do not match"

            tensor_args = {}
            all_args = []
            if raw_args is None:
                # create a dummy raw_args for uniform behavior in the following loop
                raw_args = [None] * len(call_args)
            else:
                assert len(raw_args) == len(
                    call_args
                ), "call_args and raw_args do not match"

            for i, (arg, arg_type, raw_arg) in enumerate(
                zip(call_args, arg_types, raw_args)
            ):
                key = None
                if isinstance(arg, str) and "=" in str(arg):
                    # arg may be passed in a kwarg style, and then we need to extract its value
                    key, arg = arg.split("=")

                if isinstance(arg_type, torch_dtype):
                    # workspace allocation is already generated by `generate_workspace_allocation()`
                    # in `TritonKernel.call_kernel()`.
                    if arg == "workspace":
                        arg_str = "workspace"
                        tensor_args[arg] = arg_str
                    elif arg not in tensor_args:
                        arg_str = self.generate_example_arg_value(
                            arg, arg_type, raw_arg, i
                        )
                        tensor_args[arg] = arg_str
                    else:
                        arg_str = tensor_args[arg]
                else:
                    arg_str = self.generate_example_arg_value(arg, arg_type, raw_arg, i)
                all_args.append(arg_str if key is None else f"{key}={arg_str}")

            if grid is None:
                grid_str = grid_fn
            else:
                grid_str = ", ".join(
                    self.generate_example_arg_value(g, type(g)) for g in grid
                )
                if grid_extra_kwargs:
                    grid_str = f"{grid_str}, {grid_extra_kwargs}"
                grid_str = f"{grid_fn}({grid_str})"

            self.kernel_autotune_calls.writeline(
                f"{kernel_name}.run({', '.join(all_args)}, grid={grid_str}, stream={stream_name})"
            )
            self.kernel_autotune_calls.writeline(
                f"del {', '.join(arg for arg in tensor_args.values())}\n",
            )
            self.kernel_autotune_names.add(kernel_name)

    def writeline(self, line):
        # Row-parallel reassembly elision: any str writeline that
        # references a candidate ``out_arg`` and is NOT a NUMA kernel
        # call (those contain ``numa_args=`` and are satisfied entirely
        # from ``_numa_tensor_cache`` per-socket names, not the plain
        # ``out_arg`` var) marks the candidate as consumed so the
        # deferred ``_NumaReassembleLine`` entries are kept at flush.
        candidates = self._numa_reassemble_candidates
        if candidates and isinstance(line, str) and "numa_args=" not in line:
            consumed = self._numa_reassemble_consumed
            for cand in candidates:
                if cand in consumed:
                    continue
                if re.search(rf"\b{re.escape(cand)}\b", line):
                    consumed.add(cand)
        self.lines.append(line)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    def val_to_arg_str(self, s, type_=None):
        from torch.utils._triton import dtype_to_string, has_triton_package

        if has_triton_package():
            import triton

        if isinstance(s, SymTypes):
            return pexpr(s.node.expr)
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                ref: Any

                def __repr__(self):
                    return self.ref

            return repr(type(s)(Shim(self.val_to_arg_str(a)) for a in s))
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        elif isinstance(s, (ir.Buffer, ReinterpretView)):
            return s.codegen_reference()
        elif has_triton_package() and isinstance(s, triton.language.dtype):  # type: ignore[possibly-undefined]
            return dtype_to_string(s)
        else:
            return repr(s)

    # ------------------------------------------------------------------
    # NUMA-aware buffer allocation helpers
    # ------------------------------------------------------------------
    def generate_numa_shard_allocations(
        self,
        shard_plan: Dict[str, Any],
        call_args: List[str],
        arg_types: List[Any],
        triton_meta: Optional[Dict[str, Any]] = None,
        numa_role: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, List[str]], List[str]]:
        """
        If the current kernel is part of a NUMA shard plan, emit per-NUMA-node
        clone calls for all input/output buffers and return the name of the
        ``numa_args`` list variable. Returns ``None`` if no plan applies.

        The generated code looks like::

            _numa_x_0 = torch.ops.aten.clone.default(arg1_1, numa_node=0)
            _numa_x_1 = torch.ops.aten.clone.default(arg1_1, numa_node=1)
            ...
            _numa_args = [
                (_numa_x_0, _numa_a_0, _numa_out_0),
                (_numa_x_1, _numa_a_1, _numa_out_1),
                ...
            ]
        """
        world_size = shard_plan.get("world_size", 0)
        if world_size < 2:
            return None, {}, []

        plan_id = shard_plan.get("plan_id", 0)
        prefix = f"_numa_p{plan_id}"

        per_node_arg_names: List[List[str]] = [[] for _ in range(world_size)]
        # Keep generated per-arg per-sid names so we can cache output tensors.
        generated_by_arg: Dict[str, List[str]] = {}
        sig_names: List[str] = []
        if triton_meta is not None:
            sig = triton_meta.get("signature", {})
            if isinstance(sig, dict):
                sig_names = list(sig.keys())
        tensor_sig_idx = 0

        # Normalize plan to generalized lists (column_parallel_nodes,
        # row_parallel_nodes, replicated_nodes, zero_on_nonzero_nodes). Plans
        # emitted by the legacy MLP pass (``a_node``/``b_node``/``b1_node``/
        # ``b2_node`` fields) are auto-converted by ``ensure_generalized_lists``.
        from torch._inductor.fx_passes.numa_tp_common import (
            ensure_generalized_lists,
        )

        normalized_plan = ensure_generalized_lists(shard_plan)
        x_node = normalized_plan.get("x_node")
        shard_d = int(normalized_plan.get("shard_d", 0))

        # column_parallel_nodes / row_parallel_nodes entries are 2-tuples
        # ``(name, dim)`` or 3-tuples ``(name, dim, shard_size)``. The 3-tuple
        # form lets a plan slice different weights with different per-node
        # chunk sizes (needed when Wq and Wk/Wv differ — Gemma GQA with
        # num_kv_heads < num_heads: Wq uses shard_q, Wk/Wv use shard_kv_size).
        # 2-tuples fall back to the plan-wide ``shard_d`` for backwards compat.
        def _parse_pp_entry(entry):
            if len(entry) == 3:
                n, d, s = entry
                return n, int(d), int(s)
            n, d = entry
            return n, int(d), int(shard_d)

        column_parallel_map: Dict[str, Tuple[int, int]] = {}
        for entry in normalized_plan.get("column_parallel_nodes", []):
            n, d, s = _parse_pp_entry(entry)
            column_parallel_map[n] = (d, s)
        row_parallel_map: Dict[str, Tuple[int, int]] = {}
        for entry in normalized_plan.get("row_parallel_nodes", []):
            n, d, s = _parse_pp_entry(entry)
            row_parallel_map[n] = (d, s)
        replicated_set = set(normalized_plan.get("replicated_nodes", []))
        zero_on_nonzero_set = set(normalized_plan.get("zero_on_nonzero_nodes", []))

        z_shard_like = numa_role == "z_shard"
        if not z_shard_like:
            has_row = any(n in call_args for n in row_parallel_map)
            has_col = any(n in call_args for n in column_parallel_map)
            z_shard_like = has_row and not has_col

        def _emit_replicate(var_name: str, sid: int, cache: bool = True) -> str:
            clone_name = f"{prefix}_{var_name}_n{sid}"
            init_expr = f"torch.ops.aten.clone.default({var_name}, numa_node={sid})"
            if cache:
                # Weight/bias: static between forwards — hoist behind a
                # module-level dict so the clone runs only once per compiled graph.
                self._ensure_numa_weight_cache_header()
                _emit_numa_weight_cache(
                    self, clone_name, init_expr, clone_name,
                    self._numa_suppressed_allocs,
                )
            else:
                # Activation input: changes every forward — must re-clone.
                self.writeline(
                    _NUMAFreshAllocLine(
                        f"{clone_name} = {init_expr}",
                        clone_name,
                        self._numa_suppressed_allocs,
                    )
                )
            return clone_name

        def _emit_slice_clone(var_name: str, dim: int, sid: int, shard_size: Optional[int] = None) -> str:
            # ``shard_size`` overrides the plan-wide ``shard_d`` so weights with
            # smaller per-node chunks (Wk/Wv under GQA: ``shard_kv_size``) get
            # their own start/end. Falls back to ``shard_d`` for legacy callers.
            chunk = int(shard_size) if shard_size is not None else shard_d
            start = sid * chunk
            end = (sid + 1) * chunk
            # Stride-safety rule: Inductor's MM template bakes the IR buffer's
            # strides as Python constants (``stride_bk = {{stride("B", 0)}}``
            # etc.). For a ``SliceView`` of a C-contiguous parent, the IR
            # layout keeps the parent's strides; only the size along ``dim``
            # shrinks. A compact clone of the view has a DIFFERENT stride
            # pattern unless ``dim == 0``, because slicing the leading dim
            # preserves inner strides while slicing any non-leading dim
            # changes the leading stride from ``parent_size[-1]`` down to
            # ``shard_size``. Handing a compact tensor to a kernel specialized
            # against the parent stride causes out-of-bounds reads.
            #
            # - dim == 0 (row slice of 2D, or any 1D slice): slice-then-clone
            #   yields a compact per-node buffer whose stride pattern matches
            #   the baked constants. The clone is hoisted behind _numa_weight_cache.
            # - dim != 0 (column slice of 2D): clone-full-then-slice; only the
            #   full clone is expensive and cached. The view re-creation per
            #   forward is O(1) (just pointer + strides, no memcpy).
            self._ensure_numa_weight_cache_header()
            if dim == 0:
                shard_name = f"{prefix}_{var_name}_n{sid}"
                key = shard_name
                # Inline slice into the cache initialiser to avoid a dangling
                # view_name variable in the generated module.
                init_expr = (
                    f"torch.ops.aten.clone.default("
                    f"torch.ops.aten.slice.Tensor({var_name}, {dim}, {start}, {end}, 1),"
                    f" numa_node={sid})"
                )
                _emit_numa_weight_cache(
                    self, key, init_expr, shard_name,
                    self._numa_suppressed_allocs,
                )
                return shard_name

            full_clone_name = f"{prefix}_{var_name}_full_n{sid}"
            slice_name = f"{prefix}_{var_name}_n{sid}"
            # Cache the full clone (malloc + memcpy) so it runs only once.
            key = full_clone_name
            init_expr = f"torch.ops.aten.clone.default({var_name}, numa_node={sid})"
            _emit_numa_weight_cache(
                self, key, init_expr, full_clone_name,
                self._numa_suppressed_allocs,
            )
            # The slice view is O(1) to re-create each forward (no memcpy).
            self.writeline(
                f"{slice_name} = torch.ops.aten.slice.Tensor({full_clone_name}, {dim}, {start}, {end}, 1)"
            )
            return slice_name

        def _emit_zero_bias_clone(var_name: str, sid: int) -> str:
            clone_name = f"{prefix}_{var_name}_n{sid}"
            # A zero bias is identical every forward — cache it like a weight clone.
            self._ensure_numa_weight_cache_header()
            key = clone_name
            # Inline zeros_like into the cache init to avoid a dangling intermediate.
            init_expr = (
                f"torch.ops.aten.clone.default("
                f"torch.ops.aten.zeros_like.default({var_name}), numa_node={sid})"
            )
            _emit_numa_weight_cache(
                self, key, init_expr, clone_name,
                self._numa_suppressed_allocs,
            )
            return clone_name

        def _emit_numa_empty_strided(var_name: str, sid: int) -> str:
            """Allocate an uninitialized per-node output buffer directly on the
            target NUMA socket. This replaces ``clone(var, numa_node=sid)`` for
            kernel output pointers where the data is undefined (will be
            overwritten by the kernel), avoiding a redundant ``copy_`` of
            garbage bytes and the associated first-touch page faults on the
            wrong socket.

            The buffer is cached in ``_numa_weight_cache`` so the malloc runs
            only on the first forward call. Subsequent calls reuse the same
            memory region (the kernel fully overwrites it each time, so
            stale data is never read).
            """
            out_name = f"{prefix}_{var_name}_n{sid}"
            self._ensure_numa_weight_cache_header()
            key = out_name
            init_expr = (
                f"empty_strided_cpu_numa("
                f"{var_name}.size(), {var_name}.stride(), {var_name}.dtype, "
                f"numa_node={sid})"
            )
            _emit_numa_weight_cache(
                self, key, init_expr, out_name,
                self._numa_suppressed_allocs,
            )
            return out_name

        # Only pointer/tensor arguments should be cloned and forwarded via
        # numa_args. Scalar/meta args (e.g. xnumel) are not pointers and should
        # remain in the regular launcher args.
        for arg_name, arg_type in zip(call_args, arg_types):
            if not isinstance(arg_name, str):
                continue
            if not isinstance(arg_type, torch_dtype):
                continue
            sig_name = (
                sig_names[tensor_sig_idx] if tensor_sig_idx < len(sig_names) else ""
            )
            tensor_sig_idx += 1
            is_out_only = sig_name.startswith("out_ptr")
            is_in_out = sig_name.startswith("in_out_ptr")
            for sid in range(world_size):
                # Reuse per-node outputs from a previous kernel if available.
                # Critical for multi-stage kernels like softmax where an
                # intermediate buffer is both READ and WRITTEN (``in_out_ptr``)
                # across successive kernel invocations — we must keep the
                # socket-local copy from the previous stage rather than
                # cloning the uninitialized shared Python buffer.
                if arg_name in self._numa_tensor_cache:
                    cached_name = self._numa_tensor_cache[arg_name][sid]
                    per_node_arg_names[sid].append(cached_name)
                    # Downstream consumption: mark the matching re-clone live
                    # so the deferred emission is kept at flush time.
                    self._numa_reclone_consumed.add(cached_name)
                    continue

                # Explicit output buffer policy: allocate per-node output
                # tensors directly on the target NUMA node (uninitialized).
                # The kernel fully overwrites these, so skip the wasteful
                # ``clone + copy`` of undefined bytes that _emit_replicate
                # would do. Applied to pure-output args and also to the
                # first occurrence of an in_out_ptr arg (no cache hit).
                if is_out_only or is_in_out:
                    out_name = _emit_numa_empty_strided(arg_name, sid)
                    per_node_arg_names[sid].append(out_name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = out_name
                    continue

                # Generalized plan semantics:
                # - replicate X and any tensor in ``replicated_nodes``
                #   (zero-on-nonzero-socket when in ``zero_on_nonzero_nodes``
                #   AND the kernel is row-parallel output ``z_shard``)
                # - shard-slice tensors in column_parallel_nodes / row_parallel_nodes
                if arg_name == x_node:
                    # Activation changes every forward: do NOT cache.
                    name = _emit_replicate(arg_name, sid, cache=False)
                    per_node_arg_names[sid].append(name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = name
                elif arg_name in replicated_set:
                    if (
                        arg_name in zero_on_nonzero_set
                        and z_shard_like
                        and sid != 0
                    ):
                        name = _emit_zero_bias_clone(arg_name, sid)
                    else:
                        name = _emit_replicate(arg_name, sid)  # cache=True (default)
                    per_node_arg_names[sid].append(name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = name
                elif arg_name in column_parallel_map and shard_d > 0:
                    cp_dim, cp_size = column_parallel_map[arg_name]
                    name = _emit_slice_clone(arg_name, cp_dim, sid, cp_size)
                    per_node_arg_names[sid].append(name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = name
                elif arg_name in row_parallel_map and shard_d > 0:
                    rp_dim, rp_size = row_parallel_map[arg_name]
                    name = _emit_slice_clone(arg_name, rp_dim, sid, rp_size)
                    per_node_arg_names[sid].append(name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = name
                else:
                    # Conservative fallback: replicate unknown tensor args.
                    name = _emit_replicate(arg_name, sid)
                    per_node_arg_names[sid].append(name)
                    generated_by_arg.setdefault(arg_name, [""] * world_size)[sid] = name

        numa_args_var = f"{prefix}_numa_args"
        # Trailing comma so length-1 entries become 1-tuples rather than
        # parenthesized expressions (the launcher requires a sequence per socket).
        inner = ", ".join(
            "(" + ", ".join(per_node_arg_names[sid]) + ",)"
            for sid in range(world_size)
        )
        self.writeline(f"{numa_args_var} = [{inner}]")

        # Cache output pointer args for downstream kernels (always override,
        # since the kernel just wrote new data to these buffers).
        tensor_sig_idx = 0
        for arg_name, arg_type in zip(call_args, arg_types):
            if not isinstance(arg_name, str) or not isinstance(arg_type, torch_dtype):
                continue
            sig_name = (
                sig_names[tensor_sig_idx] if tensor_sig_idx < len(sig_names) else ""
            )
            tensor_sig_idx += 1
            if (
                sig_name.startswith("out_ptr") or sig_name.startswith("in_out_ptr")
            ):
                if arg_name in generated_by_arg:
                    self._numa_tensor_cache[arg_name] = generated_by_arg[arg_name]

        # Cache input (replicated / sliced) args too, using setdefault so we
        # never override an existing output-derived entry.  This enables
        # ``_numa_propagate_buffer_alias`` to resolve buffer-reuse chains where
        # the old buffer was first touched as an *input* (e.g. bmm output
        # replicated for softmax, then reinterpreted for the next softmax).
        for arg_name, per_node in generated_by_arg.items():
            if arg_name not in self._numa_tensor_cache and all(per_node):
                self._numa_tensor_cache[arg_name] = per_node

        output_clones: Dict[str, List[str]] = {}
        tensor_sig_idx = 0
        for arg_name, arg_type in zip(call_args, arg_types):
            if not isinstance(arg_name, str) or not isinstance(arg_type, torch_dtype):
                continue
            sig_name = (
                sig_names[tensor_sig_idx] if tensor_sig_idx < len(sig_names) else ""
            )
            tensor_sig_idx += 1
            if (
                sig_name.startswith("out_ptr") or sig_name.startswith("in_out_ptr")
            ) and arg_name in generated_by_arg:
                output_clones[arg_name] = generated_by_arg[arg_name]

        pointer_positions = [
            idx
            for idx, (arg_name, arg_type) in enumerate(zip(call_args, arg_types))
            if isinstance(arg_name, str) and isinstance(arg_type, torch_dtype)
        ]
        per_node_call_args: List[str] = []
        for sid in range(world_size):
            ptr_i = 0
            args_for_sid: List[str] = []
            for pos, arg_name in enumerate(call_args):
                if pos in pointer_positions:
                    args_for_sid.append(per_node_arg_names[sid][ptr_i])
                    ptr_i += 1
                else:
                    args_for_sid.append(str(arg_name))
            per_node_call_args.append(", ".join(args_for_sid))

        return numa_args_var, output_clones, per_node_call_args

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return self.make_allocation(buffer.get_name(), device, dtype, shape, stride)

    def make_allocation(self, name, device, dtype, shape, stride):
        if device.type in ("cpu", "cuda", "xpu"):
            # optimized path for faster allocations, saving ~2us versus the stuff below
            return (
                f"{name} = empty_strided_{device.type}("
                f"{self.codegen_shape_tuple(shape)}, "
                f"{self.codegen_shape_tuple(stride)}, "
                f"{dtype})"
            )
        # all other devices:
        return (
            f"{name} = empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"device='{device.type}', dtype={dtype})"
        )

    def make_tensor_alias(self, new_name, old_name, comment=""):
        return f"{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}"

    def make_buffer_free(self, buffer):
        return f"del {buffer.get_name()}"

    def make_free_by_names(self, names_to_del: List[str]):
        return f"del {', '.join(name for name in names_to_del)}"

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"

    def _numa_propagate_buffer_alias(
        self,
        old_name: str,
        new_name: str,
        new_size: Any,
        new_stride: Any,
        exact: bool,
        code: "Optional[IndentedBuffer]" = None,
    ) -> None:
        """When a buffer is reused / reinterpreted, propagate the per-NUMA-node
        alias from ``old_name`` to ``new_name`` so that downstream kernels
        looking up ``new_name`` in the ``_numa_tensor_cache`` find the right
        socket-local storage (same underlying memory, new logical layout).

        ``code``: alias lines are written into this ``IndentedBuffer`` so they
        appear inline with the reuse line (before the kernel that reads the
        buffer). The alias variable names use the same flat convention as
        ``generate_numa_shard_allocations`` (``{prefix}_{new_name}_n{sid}``),
        and the corresponding ``_NUMAFreshAllocLine`` entries are added to
        ``_numa_suppressed_allocs`` so the stale ``empty_strided_cpu_numa``
        allocations are suppressed at flush time.
        """
        cache = getattr(self, "_numa_tensor_cache", None)
        if not cache or old_name not in cache:
            return
        per_node_old = cache[old_name]
        if not per_node_old:
            return

        # Extract the naming prefix (e.g. "_numa_p0") from the first cached
        # entry so alias names match what generate_numa_shard_allocations
        # already emitted as _NUMAFreshAllocLine placeholders.
        expected_suffix_0 = f"_{old_name}_n0"
        if per_node_old[0].endswith(expected_suffix_0):
            prefix = per_node_old[0][: -len(expected_suffix_0)]
        else:
            prefix = None  # unusual naming; fall back to _as_ style

        per_node_new: List[str] = []
        writer = code if code is not None else self
        suppressed = getattr(self, "_numa_suppressed_allocs", None)

        size_str = self.codegen_shape_tuple(new_size) if not exact else None
        stride_str = self.codegen_shape_tuple(new_stride) if not exact else None

        for sid, old_n in enumerate(per_node_old):
            if prefix is not None:
                # Flat naming: same variable name as the placeholder alloc.
                alias = f"{prefix}_{new_name}_n{sid}"
                if suppressed is not None:
                    suppressed.add(alias)
            else:
                # Fallback: compound _as_ naming (no suppression needed since
                # there is no matching placeholder allocation in this case).
                alias = f"{old_n}_as_{new_name}"

            if exact:
                writer.writeline(f"{alias} = {old_n}")
            else:
                writer.writeline(
                    f"{alias} = reinterpret_tensor({old_n}, {size_str}, {stride_str}, 0)"
                )
            per_node_new.append(alias)
        cache[new_name] = per_node_new

    def make_buffer_reuse(
        self,
        old: ir.Buffer,
        new: ir.Buffer,
        delete_old: bool,
        code: "Optional[IndentedBuffer]" = None,
    ):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        exact = (
            old.get_size() == new.get_size()
            and old.get_stride() == new.get_stride()
        )

        # Propagate NUMA aliases so ``new_name`` resolves to socket-local
        # storage for kernels that read ``in_out_ptr*`` / in-place args.
        # Pass ``code`` so that alias lines are emitted inline (before the
        # reuse line) rather than deferred to the end of ``call()``.
        try:
            self._numa_propagate_buffer_alias(
                old_name, new_name, new.get_size(), new.get_stride(), exact,
                code=code,
            )
        except Exception:
            # Defensive: never fail compilation for NUMA alias propagation.
            pass

        if exact:
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call
        )
        return f"{self.declare_maybe_reference}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare_maybe_reference}{name} = {layout.view.codegen_reference()}{self.ending}  "
                f"{self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer: ir.Buffer):
        name = buffer.get_name()

        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)
        if isinstance(
            buffer.get_defining_op(),
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            return
        if isinstance(layout, ir.NoneLayout):
            return
        if isinstance(layout, ir.NonOwningLayout):
            assert isinstance(
                layout.view, ir.ReinterpretView
            ), f"unexpected {type(layout.view)}: {layout.view}"
            assert isinstance(layout.view.data, ir.StorageBox), type(layout.view.data)
            assert isinstance(layout.view.data.data, ir.Buffer), type(layout.view.data)
            self.codegen_allocation(layout.view.data.data)
            self.codegen_deferred_allocation(name, layout)
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, input_buffer, output_buffer=None):
        name = input_buffer.get_name()
        return not (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in V.graph.torchbind_constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        )

    def did_reuse(self, buffer, reused_buffer):
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer: ir.Buffer, output_buffer: ir.Buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

    def codegen_unbacked_symbol_decl(self, symbol):
        name = str(symbol)
        if name in self.unbacked_symbol_decls:
            return name
        else:
            # When in CppWrapperCpu, we should only generate the declaration once
            self.unbacked_symbol_decls.add(name)
            return self.declare + name

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        for inner_input, outer_input in zip(subgraph.graph.graph_inputs, outer_inputs):
            self.writeline(f"{self.declare}{inner_input} = {outer_input}{self.ending}")

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            self.writeline(
                f"{outer_output} = {inner_output.codegen_reference()}{self.ending}"
            )

    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"{self.comment} subgraph: {subgraph.name}")
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            self.pop_codegened_graph()

    def codegen_conditional(self, conditional):
        name = conditional.get_name()

        self.writeline(f"{name} = [None] * {len(conditional.outputs)}")

        outer_inputs = [buf.codegen_reference() for buf in conditional.operands]
        outer_outputs = [f"{name}[{i}]" for i in range(len(conditional.outputs))]

        predicate = conditional.predicate.codegen_reference()
        if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
            # move the Tensor predicate to host
            predicate = f"{predicate}.item()"

        self.writeline(f"{name} = [None] * {len(conditional.outputs)}")
        self.writeline(f"if {predicate}:")
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("else:")
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))

    def codegen_while_loop(self, while_loop):
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]

        self.writeline(f"{name} = [None] * {len(outer_carried_inputs)}")
        for i, inp in enumerate(outer_carried_inputs):
            # set the initial state before the loop
            self.writeline(f"{name}[{i}] = {inp}")

        cond_outer_inputs = [
            *[f"{name}[{i}]" for i in range(len(outer_carried_inputs))],
            *outer_additional_inputs,
        ]
        cond_outer_outputs = [f"{name}_cond_result"]
        body_outer_inputs = list(
            cond_outer_inputs
        )  # same inputs for cond_fn and body_fn
        # Carry over the state from body_fn. Note: We only carry over
        # the carried_inputs part of the inputs, the additional ones
        # are passed in as they're before.
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        self.writeline("while True:")
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
        self.codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )
        self.writeline(
            f"if not {cond_outer_outputs[0]}.item(): break"
        )  # condition doesn't hold
        self.writeline(ExitSubgraphLine(self))
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        self.codegen_subgraph(
            while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
        )
        self.writeline(ExitSubgraphLine(self))

    @staticmethod
    def statically_known_int_or_none(x):
        try:
            if getattr(x, "free_symbols", None):
                # _maybe_evaluate_static will return (s0 // (2 // s0)) as 2, but
                # the actual codegen will still generate the full expression here.
                return None
            if isinstance(x, int):
                return x
            val = V.graph._shape_env._maybe_evaluate_static(x)
            return int(val)
        except Exception:
            return None

    @staticmethod
    def statically_known_list_of_ints_or_none(lst):
        result = []
        for x in lst:
            num = PythonWrapperCodegen.statically_known_int_or_none(x)
            if num is None:
                return None
            result.append(num)
        return result

    @staticmethod
    def is_statically_known_list_of_ints(lst):
        return (
            PythonWrapperCodegen.statically_known_list_of_ints_or_none(lst) is not None
        )

    @staticmethod
    def static_shape_for_buffer_or_none(buffer):
        return PythonWrapperCodegen.statically_known_list_of_ints_or_none(
            buffer.get_size()
        )

    @staticmethod
    def can_prove_buffer_has_static_shape(buffer):
        return PythonWrapperCodegen.static_shape_for_buffer_or_none(buffer) is not None
