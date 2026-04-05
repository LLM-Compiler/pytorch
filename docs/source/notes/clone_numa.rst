.. _clone-numa-note:

NUMA placement for :func:`torch.clone`
========================================

This note describes the optional **NUMA node** placement argument on
:func:`torch.clone` and :meth:`torch.Tensor.clone`.

When to use
-----------

On multi-socket Linux hosts, memory is attached to **NUMA nodes**. Placing a
newly allocated tensor on a specific node can reduce remote memory access when
subsequent compute is pinned to CPUs on that node.

The ``numa_node`` keyword requests that the **destination storage** of a clone
be associated with a given node (implementation uses migration after allocation;
see the C++ helper ``c10::NUMAMove``).

API summary
-----------

* **Signature:** ``clone(..., *, memory_format=..., numa_node=None)``
* **Default:** ``numa_node=None`` — behavior matches historical ``clone`` (no NUMA placement hint).
* **CPU, dense (strided) tensors:** When NUMA is enabled at build and runtime,
  ``numa_node`` may be set to an integer in ``[0, num_nodes-1]``.
* **Other layouts / devices:** Passing ``numa_node`` is rejected with a clear error
  (e.g. sparse, quantized, MKLDNN, nested, non-CPU devices), because those use
  different ``clone`` implementations.

Requirements
------------

#. **Linux** — NUMA helpers are not enabled for PyTorch CPU NUMA in the same way on all platforms.
#. **Build** — PyTorch must be built with NUMA support (``C10_USE_NUMA`` / ``USE_NUMA`` as configured in CMake).
#. **Runtime** — ``libnuma`` must be usable and the machine must expose NUMA topology
   (e.g. multiple nodes in ``/sys/devices/system/node/``).
#. **Effective enable flag** — Internally, ``c10::IsNUMAEnabled()`` requires both:

   * the runtime flag ``caffe2_cpu_numa_enabled`` (see below), and
   * ``numa_available() >= 0`` from libnuma.

If NUMA is not enabled, passing ``numa_node`` is an error (strict policy), so
callers do not silently ignore placement.

Enabling NUMA at runtime (for development / tests)
----------------------------------------------------

The global default for ``caffe2_cpu_numa_enabled`` may be off. For **this fork**,
internal helpers are exposed on ``torch._C._cpu`` (unstable API):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Call
     - Meaning
   * - ``torch._C._cpu._set_numa_enabled(bool)``
     - Sets the runtime flag (process-wide).
   * - ``torch._C._cpu._is_numa_enabled()``
     - ``True`` if NUMA placement is active (same as ``c10::IsNUMAEnabled()``).
   * - ``torch._C._cpu._num_numa_nodes()``
     - Configured node count when enabled; ``-1`` when disabled.

Example:

.. code-block:: python

    import torch

    torch._C._cpu._set_numa_enabled(True)
    if not torch._C._cpu._is_numa_enabled():
        raise RuntimeError("NUMA not available on this build/host")

    x = torch.randn(1024, 1024)
    y = x.clone(numa_node=0)

Do not rely on ``torch._C._cpu`` names for third-party libraries; they are for
debugging and tests unless promoted to a public API.

Autograd and ``torch.compile``
-------------------------------

* **Autograd:** ``clone`` remains differentiable in the usual way; NUMA placement
  does not change gradient semantics.
* **``torch.compile`` / refs:** Decompositions that route through Python refs may
  not carry ``numa_node``; treat NUMA placement as supported for **eager** ATen
  ``clone`` unless you extend the decomposition stack.

Further reading
---------------

* Implementation: ``aten/src/ATen/native/TensorFactories.cpp`` (dense ``clone``).
* Tests: ``test/test_torch.py`` (search for ``clone_numa_node``).
