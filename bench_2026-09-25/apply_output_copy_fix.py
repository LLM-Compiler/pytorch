#!/usr/bin/env python3
"""Reconcile replicated per-socket results that are also graph outputs.

The wrapper has two reconciliation paths (paper Fig. 6): terminal partial sums
get an all-reduce, intermediate activations stay socket-local and are forwarded
to the next local kernel. A replicated result that is itself RETURNED from the
graph fits neither -- there is no next local kernel -- so nothing writes the
buffer the caller receives and the model returns unwritten memory.

Exact-match and idempotent.
"""
import sys

PATH = "/home/shavin/llm-cpu-compiler/pytorch/torch/_inductor/codegen/wrapper.py"

ANCHOR_CLS = '''class _NUMAFreshAllocLine(DeferredLineBase):'''

NEW_CLS = '''class _NumaGraphOutputCopyLine(DeferredLineBase):
    """Reconcile a replicated per-socket result that is also a graph output.

    The two reconciliation paths cover terminal partial sums (all-reduce) and
    intermediate activations (stay socket-local, forwarded to the next local
    kernel). A replicated result that is returned from the graph fits neither:
    it has no downstream local kernel, so nothing would ever write the buffer
    the caller receives, and the model returns unwritten memory.

    Every socket holds a COMPLETE copy of a replicated result (Algorithm 3), so
    a single copy from node 0 reconciles it. Summing would scale the value by
    the socket count. Emitted only if ``name`` really is a graph output at
    flush time, when ``V.graph.graph_outputs`` is final.
    """

    def __init__(self, line: str, name: str) -> None:
        super().__init__(line)
        self.name = name

    def __call__(self) -> Optional[str]:
        return self.line if self.name in V.graph.get_output_names() else None

    def _new_line(self, line: str) -> "_NumaGraphOutputCopyLine":
        return _NumaGraphOutputCopyLine(line, self.name)


class _NUMAFreshAllocLine(DeferredLineBase):'''

ANCHOR_BLOCK = '''                        self._numa_tensor_cache[out_arg] = reclone_names
            else:'''

NEW_BLOCK = '''                        self._numa_tensor_cache[out_arg] = reclone_names
                else:
                    # Not a row-parallel kernel, so there is no partial sum to
                    # reduce. But if one of these per-socket results is the
                    # graph's own output (e.g. the final replicated LayerNorm
                    # of a Pre-LN stack) there is no downstream local kernel to
                    # forward it to either, and the returned buffer would never
                    # be written. Reconcile it with a single copy from node 0.
                    for out_arg, per_node_outs in numa_output_clones.items():
                        if not per_node_outs:
                            continue
                        self.writeline(
                            _NumaGraphOutputCopyLine(
                                f"{out_arg}.copy_({per_node_outs[0]})",
                                out_arg,
                            )
                        )
            else:'''


def main():
    with open(PATH, "r", encoding="utf-8", newline="") as fh:
        src = fh.read()

    if "_NumaGraphOutputCopyLine" in src:
        print("ALREADY PATCHED - no change")
        return 0

    for label, old in (("class anchor", ANCHOR_CLS), ("block anchor", ANCHOR_BLOCK)):
        n = src.count(old)
        if n != 1:
            print("ABORT: %s matched %d times (expected 1)" % (label, n))
            return 1

    src = src.replace(ANCHOR_CLS, NEW_CLS).replace(ANCHOR_BLOCK, NEW_BLOCK)

    with open(PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(src)
    print("PATCHED ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
