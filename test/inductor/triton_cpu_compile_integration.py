#!/usr/bin/env python3
"""
Post-build integration check: torch.compile on CPU with Inductor + Triton CPU backend.

Run from the PyTorch repo root after installing this tree and a Triton build with the
CPU backend (same env you use for Inductor CPU tests):

  conda activate triton_cpu_integrated
  python test/inductor/triton_cpu_compile_integration.py --fast --allow-extern-gemm

With TORCH_COMPILE_DEBUG, Inductor writes traces under
  <TORCH_COMPILE_DEBUG_DIR>/torch_compile_debug/.../output_code.py
(Default parent dir is this file's directory if TORCH_COMPILE_DEBUG_DIR is unset.)

Checks (unless --no-verify):
  - output_code.py exists under that tree
  - Triton kernels present (async_compile.triton / import triton)
  - By default: no extern_kernels.*(...) calls (strict GEMM path). Use --allow-extern-gemm
    if upstream CPU Inductor still emits extern_kernels.mm/addmm.

See also: test/inductor/test_triton_cpu_compile_integration.py (unittest wrapper).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

# ---------------------------------------------------------------------------
# Must run before importing torch._inductor / torch (trace.enabled & debug dir)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TORCH_COMPILE_DEBUG"] = "1"
# Parent directory for torch_compile_debug/ (see torch._dynamo.config.default_debug_dir_root)
os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", _SCRIPT_DIR)

import torch._inductor.config  # noqa: E402
import torch._dynamo.config  # noqa: E402

torch._inductor.config.max_autotune_gemm_backends = "TRITON"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _torch_compile_debug_root() -> str:
    """Directory that contains run_* trace folders (matches Dynamo debug_dir_root)."""
    parent = os.environ.get("TORCH_COMPILE_DEBUG_DIR", _SCRIPT_DIR)
    return os.path.join(parent, "torch_compile_debug")


class TransformerBlock(nn.Module):
    """Single transformer block: LN, MHA (QKV matmuls, softmax, AV), residual, FFN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        b, s, _ = h.shape
        q = self.w_q(h).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(h).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(h).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        scale = 1.0 / (self.d_head**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        mix = torch.matmul(attn, v)
        mix = mix.transpose(1, 2).contiguous().view(b, s, -1)
        h = x + self.w_o(mix)
        z = self.ln2(h)
        z = self.ff1(z)
        z = F.gelu(z)
        z = self.ff2(z)
        return h + z


class LLMStyleBundle(nn.Module):
    """Embeddings, transformer block, weight-tied logits, pooling."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block = TransformerBlock(d_model, n_heads)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, tok_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(tok_ids)
        x = self.block(x)
        x = self.proj(x)
        x = self.out_ln(x)
        logits = torch.matmul(x, self.embed.weight.T)
        pooled = x.mean(dim=1)
        pooled = F.silu(pooled)
        scores = pooled.sum(dim=-1, keepdim=True)
        return logits, scores


def _clear_local_compile_debug() -> None:
    root = _torch_compile_debug_root()
    if os.path.isdir(root):
        shutil.rmtree(root)


def _clear_inductor_cache() -> None:
    try:
        import torch._inductor.codecache as codecache

        if hasattr(codecache, "cache_dir"):
            d = codecache.cache_dir()
            if d and os.path.isdir(d):
                shutil.rmtree(d)
    except Exception:
        pass


def _find_output_code_files() -> list[str]:
    pattern = os.path.join(_torch_compile_debug_root(), "**", "output_code.py")
    return sorted(glob.glob(pattern, recursive=True))


def _read_files(paths: list[str]) -> str:
    parts: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            parts.append(f"# --- {p} ---\n")
            parts.append(f.read())
            parts.append("\n")
    return "".join(parts)


def verify_generated_code(text: str, *, allow_extern_gemm: bool) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True

    if not text.strip():
        return False, ["No output_code.py content to analyze."]

    triton_defs = len(re.findall(r"async_compile\.triton\(", text))
    has_triton = bool(
        re.search(r"\bimport\s+triton\b", text)
        or re.search(r"triton\.language", text)
        or re.search(r"@triton\.jit", text)
        or triton_defs > 0
    )
    if has_triton:
        msgs.append(
            f"Found Triton in output_code.py (async_compile.triton(...) blocks: {triton_defs})."
        )
    else:
        ok = False
        msgs.append(
            "FAIL: No Triton in output_code.py "
            "(expected import triton / async_compile.triton / @triton.jit)."
        )

    extern_lines = [
        ln.strip()
        for ln in text.splitlines()
        if "extern_kernels." in ln and not ln.strip().startswith("#")
    ]
    extern_calls = [ln for ln in extern_lines if re.search(r"extern_kernels\.\w+\s*\(", ln)]
    extern_kinds = sorted(set(re.findall(r"extern_kernels\.(\w+)\s*\(", text)))

    if extern_calls:
        if allow_extern_gemm:
            msgs.append(
                f"Note: extern_kernels call sites present ({len(extern_calls)} lines); "
                f"ops: {extern_kinds} (--allow-extern-gemm)."
            )
        else:
            ok = False
            msgs.append(
                f"FAIL: extern_kernels.* calls in output_code.py ({len(extern_calls)} lines); "
                f"ops: {extern_kinds}. For patched PyTorch that routes GEMM to Triton, expect zero. "
                f"Otherwise use --allow-extern-gemm."
            )
    else:
        msgs.append("No extern_kernels.(...) calls in output_code.py.")

    return ok, msgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Remove local torch_compile_debug and Inductor cache before running.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster compile: smaller problem and max_autotune=False.",
    )
    p.add_argument(
        "--allow-extern-gemm",
        action="store_true",
        help="Do not fail when extern_kernels.mm/addmm remain (common on upstream CPU).",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip output_code checks (forward only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.fresh:
        _clear_local_compile_debug()
        _clear_inductor_cache()

    torch._inductor.config.max_autotune = not args.fast

    device = torch.device("cpu")
    torch.manual_seed(0)

    if args.fast:
        vocab_size = 128
        d_model = 32
        n_heads = 4
        b, s = 1, 8
    else:
        vocab_size = 256
        d_model = 64
        n_heads = 4
        b, s = 2, 16

    model = LLMStyleBundle(vocab_size, d_model, n_heads).to(device).eval()
    tok = torch.randint(0, vocab_size, (b, s), device=device)

    compiled = torch.compile(model)
    with torch.no_grad():
        logits, scores = compiled(tok)

    assert logits.shape == (b, s, vocab_size)
    assert scores.shape == (b, 1)

    if args.no_verify:
        print("Forward OK (--no-verify: skipped output_code checks).")
        return 0

    paths = _find_output_code_files()
    if not paths:
        print(
            "FAIL: No output_code.py under "
            f"{_torch_compile_debug_root()}. "
            "Ensure TORCH_COMPILE_DEBUG=1 before first import and try --fresh.",
            file=sys.stderr,
        )
        return 2

    text = _read_files(paths)
    ok, msgs = verify_generated_code(text, allow_extern_gemm=args.allow_extern_gemm)
    print("output_code.py file(s):")
    for p in paths:
        print(f"  {p}")
    print()
    for m in msgs:
        print(m)
    print()
    if ok:
        msg = (
            "PASS: torch.compile forward succeeded; output_code.py contains Triton kernels"
        )
        if args.allow_extern_gemm:
            msg += " (--allow-extern-gemm: extern_kernels.mm/addmm not treated as failure)."
        else:
            msg += "; no extern_kernels.(...) calls."
        print(msg)
        return 0
    print("FAIL: See messages above.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
