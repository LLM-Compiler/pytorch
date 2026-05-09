#!/usr/bin/env python3
# mypy: ignore-errors
r"""
BERT encoder benchmark on **stock (unmodified) PyTorch**.

Runs the same BERT-Base / BERT-Large model architectures and
(batch, sequence) configurations as ``bert_multi_size_compile_debug_sweep.py``,
but uses stock ``torch.compile`` (inductor) **and** eager mode — no NUMA-TP
flags, no modified PyTorch internals.

For each (model, batch, seq) configuration the script measures:
  1. **eager** mode (no compile)
  2. **compile** mode (``torch.compile`` with default Inductor backend)

Results are written to a CSV (``--csv``), and optionally committed/pushed
to git after each row (``--git-commit-push``).

CSV columns::

    model, batch, seq_length, mode, mean_ms, median_ms, min_ms, warmup, iters

Default iteration space::

    models:     base, large
    batch:      1, 2, 4, 8
    seq:        128, 256, 512, 1024, 2048, 4096, 8192

Example::

    python scripts/inductor/bert_stock_torch_benchmark.py \
        --csv ./stock_bert_results.csv --warmup 5 --iters 50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# ── Constants ────────────────────────────────────────────────────────────
DEF_VOCAB = 30522
DEF_CONTEXT = 512


# ── Model presets ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BERTSizePreset:
    label: str
    num_layers: int
    d_model: int
    num_heads: int
    head_dim: int
    d_ffn: int


MODEL_PRESETS: Dict[str, BERTSizePreset] = {
    "base": BERTSizePreset(
        label="BERT-Base (~110M)",
        num_layers=12, d_model=768, num_heads=12, head_dim=64, d_ffn=3072,
    ),
    "large": BERTSizePreset(
        label="BERT-Large (~340M)",
        num_layers=24, d_model=1024, num_heads=16, head_dim=64, d_ffn=4096,
    ),
}


# ── Model ────────────────────────────────────────────────────────────────
class BertTransformerBlock(torch.nn.Module):
    """Post-LN BERT encoder block: ``LN(x + attn(x))`` then ``LN(x + mlp(x))``."""

    def __init__(self, h_in: int, num_heads: int, head_dim: int, d_ffn: int) -> None:
        super().__init__()
        self.H = num_heads
        self.E = head_dim
        self.H_in = h_in
        h_qkv = num_heads * head_dim

        self.ln1 = torch.nn.LayerNorm(h_in)
        self.Wq = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.Wk = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.Wv = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.bq = torch.nn.Parameter(torch.empty(h_qkv))
        self.bk = torch.nn.Parameter(torch.empty(h_qkv))
        self.bv = torch.nn.Parameter(torch.empty(h_qkv))
        self.Wo = torch.nn.Parameter(torch.empty(h_qkv, h_in))
        self.bo = torch.nn.Parameter(torch.empty(h_in))

        self.ln2 = torch.nn.LayerNorm(h_in)
        self.mlp_fc1_w = torch.nn.Parameter(torch.empty(h_in, d_ffn))
        self.mlp_fc1_b = torch.nn.Parameter(torch.empty(d_ffn))
        self.mlp_fc2_w = torch.nn.Parameter(torch.empty(d_ffn, h_in))
        self.mlp_fc2_b = torch.nn.Parameter(torch.empty(h_in))

    def _attn(self, x: torch.Tensor, B: int, S: int) -> torch.Tensor:
        H, E = self.H, self.E
        x_flat = x.reshape(B * S, self.H_in)
        q = torch.addmm(self.bq, x_flat, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x_flat, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x_flat, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(self.bo, y, self.Wo).view(B, S, self.H_in)

    def _mlp(self, x: torch.Tensor, B: int, S: int) -> torch.Tensor:
        x_flat = x.reshape(B * S, self.H_in)
        h = torch.addmm(self.mlp_fc1_b, x_flat, self.mlp_fc1_w)
        h = F.gelu(h)
        return torch.addmm(self.mlp_fc2_b, h, self.mlp_fc2_w).view(B, S, self.H_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        x = self.ln1(x + self._attn(x, B, S))
        x = self.ln2(x + self._mlp(x, B, S))
        return x


class BertEncoderModel(torch.nn.Module):
    """Minimal BERT encoder stack (random init; prefill through last block)."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        type_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(context_size, d_model)
        self.token_type_embedding = torch.nn.Embedding(type_vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            BertTransformerBlock(d_model, num_heads, head_dim, d_ffn)
            for _ in range(num_layers)
        )
        self._init_weights(num_residual_layers=2 * num_layers)

    def _init_weights(self, num_residual_layers: int) -> None:
        std = 0.02
        residual_std = std / math.sqrt(num_residual_layers)
        with torch.no_grad():
            torch.nn.init.normal_(self.token_embedding.weight, std=std)
            torch.nn.init.normal_(self.position_embedding.weight, std=std)
            torch.nn.init.normal_(self.token_type_embedding.weight, std=std)
            for blk in self.blocks:
                for p in (blk.Wq, blk.Wk, blk.Wv, blk.mlp_fc1_w):
                    torch.nn.init.normal_(p, std=std)
                for p in (blk.Wo, blk.mlp_fc2_w):
                    torch.nn.init.normal_(p, std=residual_std)
                for p in (
                    blk.bq, blk.bk, blk.bv, blk.bo,
                    blk.mlp_fc1_b, blk.mlp_fc2_b,
                ):
                    torch.nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device)
        token_type_ids = torch.zeros(
            (B, S), dtype=torch.long, device=input_ids.device
        )
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.token_type_embedding(token_type_ids)
        )
        for blk in self.blocks:
            x = blk(x)
        return x


# ── Timing ───────────────────────────────────────────────────────────────
@dataclass
class TimingSummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    iters: int


def _time_forward(
    fn,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> TimingSummary:
    with torch.no_grad():
        for _ in range(warmup):
            fn(input_ids)
        ordered: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(input_ids)
            ordered.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ordered)
    if not ordered:
        return TimingSummary(mean_ms=0.0, median_ms=0.0, min_ms=0.0, iters=0)
    return TimingSummary(
        mean_ms=sum(ordered) / len(ordered),
        median_ms=s[len(s) // 2],
        min_ms=s[0],
        iters=iters,
    )


# ── CSV helpers ──────────────────────────────────────────────────────────
_CSV_HEADER = [
    "model", "batch", "seq_length", "mode",
    "mean_ms", "median_ms", "min_ms",
    "warmup", "iters",
]


def _init_csv(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_HEADER)
        fh.flush()
        os.fsync(fh.fileno())


def _append_csv(
    path: str, *, model: str, batch: int, seq: int, mode: str,
    timing: TimingSummary, warmup: int, iters: int,
) -> None:
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            model, batch, seq, mode,
            f"{timing.mean_ms:.4f}",
            f"{timing.median_ms:.4f}",
            f"{timing.min_ms:.4f}",
            warmup, iters,
        ])
        fh.flush()
        os.fsync(fh.fileno())


# ── Git helpers ──────────────────────────────────────────────────────────
def _git_top_level(start_dir: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", start_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _git_commit_push(
    csv_abspath: str,
    *,
    repo_root: Optional[str] = None,
    remote: str = "origin",
    branch: Optional[str] = None,
    commit_message: str,
) -> None:
    csv_abspath = os.path.abspath(csv_abspath)
    if not os.path.isfile(csv_abspath):
        raise FileNotFoundError(f"CSV not found: {csv_abspath}")
    if repo_root is None:
        repo_root = _git_top_level(os.path.dirname(csv_abspath))
    else:
        repo_root = os.path.abspath(repo_root)
    if not repo_root:
        raise RuntimeError(f"No git repo containing {csv_abspath}")
    rel = os.path.relpath(csv_abspath, repo_root)
    subprocess.run(["git", "-C", repo_root, "add", "--", rel], check=True)
    diff = subprocess.run(["git", "-C", repo_root, "diff", "--staged", "--quiet"])
    if diff.returncode == 0:
        print(f"git: no staged changes for {rel} (skipped).", file=sys.stderr, flush=True)
        return
    subprocess.run(["git", "-C", repo_root, "commit", "-m", commit_message], check=True)
    if branch is None:
        br = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        branch = br.stdout.strip()
    subprocess.run(["git", "-C", repo_root, "push", remote, branch], check=True)


# ── Batch/seq filtering ─────────────────────────────────────────────────
def _batch_seq_pairs(
    batch_sizes: List[int],
    seq_lens: List[int],
    *,
    tiered_max_seq_by_batch: bool,
    tiered_seq_base: int,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for B in batch_sizes:
        cap = (tiered_seq_base // B) if tiered_max_seq_by_batch else (max(seq_lens) if seq_lens else 0)
        for S in seq_lens:
            if S <= cap:
                pairs.append((B, S))
    return pairs


# ── CLI ──────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BERT benchmark on stock (unmodified) PyTorch: eager + torch.compile"
    )
    p.add_argument(
        "--csv", type=str, default="./stock_bert_benchmark.csv",
        help="Path for results CSV (default: ./stock_bert_benchmark.csv).",
    )
    p.add_argument(
        "--models", nargs="+", default=["base", "large"],
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model presets (default: base large).",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
    )
    p.add_argument("--tiered-max-seq-by-batch", action="store_true")
    p.add_argument("--tiered-seq-base", type=int, default=8192)
    p.add_argument("--type-vocab-size", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=DEF_VOCAB)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--git-commit-push", action="store_true",
        help="Git add/commit/push CSV after each row.",
    )
    p.add_argument("--git-repo-root", type=str, default=None)
    p.add_argument("--git-remote", type=str, default="origin")
    p.add_argument("--git-branch", type=str, default=None)
    p.add_argument("--git-continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    csv_path = os.path.abspath(args.csv)
    _init_csv(csv_path)

    batch_seq_pairs = _batch_seq_pairs(
        list(args.batch_sizes), list(args.seq_lens),
        tiered_max_seq_by_batch=args.tiered_max_seq_by_batch,
        tiered_seq_base=args.tiered_seq_base,
    )
    if not batch_seq_pairs:
        print("No (batch, seq) pairs after filtering.", file=sys.stderr)
        return 2

    print(
        f"Stock PyTorch BERT benchmark\n"
        f"  torch={torch.__version__}\n"
        f"  hostname={platform.node()}\n"
        f"  models={args.models}\n"
        f"  batch_seq_pairs={len(batch_seq_pairs)} configs\n"
        f"  warmup={args.warmup} iters={args.iters}\n"
        f"  csv={csv_path}",
        file=sys.stderr, flush=True,
    )

    torch.manual_seed(args.seed)
    t0_all = time.perf_counter()
    ok = 0
    fail = 0

    for model_key in args.models:
        preset = MODEL_PRESETS[model_key]
        arch = {
            "num_layers": preset.num_layers,
            "d_model": preset.d_model,
            "num_heads": preset.num_heads,
            "head_dim": preset.head_dim,
            "d_ffn": preset.d_ffn,
        }

        for B, S in batch_seq_pairs:
            context_size = max(S, DEF_CONTEXT)
            model = BertEncoderModel(
                vocab_size=args.vocab_size,
                context_size=context_size,
                type_vocab_size=args.type_vocab_size,
                d_model=arch["d_model"],
                num_layers=arch["num_layers"],
                num_heads=arch["num_heads"],
                head_dim=arch["head_dim"],
                d_ffn=arch["d_ffn"],
            ).eval()
            input_ids = torch.randint(0, args.vocab_size, (B, S), dtype=torch.long)

            # ── Eager mode ───────────────────────────────────────────
            mode = "eager"
            label = f"BERT {model_key} {mode} B={B} S={S}"
            print(f"[run] {label}", file=sys.stderr, flush=True)
            try:
                timing = _time_forward(
                    model, input_ids,
                    warmup=args.warmup, iters=args.iters,
                )
                ok += 1
                _append_csv(
                    csv_path, model=model_key, batch=B, seq=S, mode=mode,
                    timing=timing, warmup=args.warmup, iters=args.iters,
                )
                print(
                    f"  OK mean_ms={timing.mean_ms:.3f}",
                    file=sys.stderr, flush=True,
                )
                if args.git_commit_push:
                    try:
                        _git_commit_push(
                            csv_path,
                            repo_root=args.git_repo_root,
                            remote=args.git_remote,
                            branch=args.git_branch,
                            commit_message=(
                                f"Stock BERT {mode}: {model_key} B={B} S={S} "
                                f"mean={timing.mean_ms:.3f}ms"
                            ),
                        )
                    except Exception as ge:
                        if args.git_continue_on_error:
                            print(f"  WARN git: {ge}", file=sys.stderr, flush=True)
                        else:
                            raise
            except Exception as e:
                fail += 1
                print(f"  FAIL {e}", file=sys.stderr, flush=True)

            # ── Compile mode ─────────────────────────────────────────
            mode = "compile"
            label = f"BERT {model_key} {mode} B={B} S={S}"
            print(f"[run] {label}", file=sys.stderr, flush=True)
            try:
                torch._dynamo.reset()
                compiled = torch.compile(model, mode="default", fullgraph=False)
                # Initial compile pass (counted as first warmup)
                with torch.no_grad():
                    compiled(input_ids)
                remaining_warmup = max(0, args.warmup - 1)
                timing = _time_forward(
                    compiled, input_ids,
                    warmup=remaining_warmup, iters=args.iters,
                )
                ok += 1
                _append_csv(
                    csv_path, model=model_key, batch=B, seq=S, mode=mode,
                    timing=timing, warmup=args.warmup, iters=args.iters,
                )
                print(
                    f"  OK mean_ms={timing.mean_ms:.3f}",
                    file=sys.stderr, flush=True,
                )
                if args.git_commit_push:
                    try:
                        _git_commit_push(
                            csv_path,
                            repo_root=args.git_repo_root,
                            remote=args.git_remote,
                            branch=args.git_branch,
                            commit_message=(
                                f"Stock BERT {mode}: {model_key} B={B} S={S} "
                                f"mean={timing.mean_ms:.3f}ms"
                            ),
                        )
                    except Exception as ge:
                        if args.git_continue_on_error:
                            print(f"  WARN git: {ge}", file=sys.stderr, flush=True)
                        else:
                            raise
            except Exception as e:
                fail += 1
                print(f"  FAIL {e}", file=sys.stderr, flush=True)

    total_s = time.perf_counter() - t0_all
    print(
        f"\nDone. ok={ok} fail={fail} wall_s={total_s:.1f}\n"
        f"CSV: {csv_path}",
        file=sys.stderr,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
