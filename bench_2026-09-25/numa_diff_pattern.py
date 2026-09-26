#!/usr/bin/env python
"""Characterise HOW the NUMA-TP output differs from the reference.

max_rel == 1.0 exactly is the signature of "NUMA output is zero where the
reference is not". This confirms or refutes that, and localises the damage to
specific rows/columns of the [B, S, D] output so the culprit buffer is
identifiable.
"""
import argparse
import os
import sys
import tempfile

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))

import torch  # noqa: E402
from torch._inductor import config as inductor_config  # noqa: E402

import gpt2_multi_size_speedup_cli as G  # noqa: E402


def flags_for(combo, world_size):
    f = {
        "_numa_layer_norm_tp": False, "_numa_attention_tp": False,
        "_numa_mlp_tp": False, "_numa_residual_tp": False,
        "_numa_lm_head_tp": False,
        "_numa_layer_norm_tp_world_size": 0,
        "_numa_attention_tp_world_size": 0,
        "_numa_mlp_tp_world_size": 0, "_numa_lm_head_tp_world_size": 0,
    }
    for name in combo:
        f["_numa_%s_tp" % name] = True
        key = "_numa_%s_tp_world_size" % name
        if key in f:
            f[key] = world_size
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default="mlp")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--ref", required=True, help="ref.pt from the isolation run")
    args = ap.parse_args()

    G._enable_numa()
    torch.manual_seed(0)
    preset = G.MODEL_PRESETS["117m"]
    model = G.GPT2PreLN(
        vocab_size=G.DEF_VOCAB,
        context_size=max(args.seq_length, G.DEF_CONTEXT),
        d_model=preset.d_model, num_layers=args.num_layers,
        num_heads=preset.num_heads, head_dim=preset.head_dim,
        d_ffn=preset.d_ffn,
    ).eval()
    input_ids = torch.randint(
        0, G.DEF_VOCAB, (args.batch_size, args.seq_length), dtype=torch.long
    )

    combo = tuple(x for x in args.combo.split(",") if x)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="numa_diff_")
    torch._dynamo.reset()
    with inductor_config.patch(flags_for(combo, args.world_size)):
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.no_grad():
            got = compiled(input_ids).detach().clone()

    ref = torch.load(args.ref).float()
    got = got.float()
    d = (got - ref).abs()
    mism = d > (1e-3 + 1e-3 * ref.abs())

    print("combo=%s world_size=%d shape=%s"
          % (args.combo, args.world_size, list(ref.shape)))
    print("mismatched elements: %d / %d  (%.2f%%)"
          % (int(mism.sum()), mism.numel(), 100.0 * float(mism.float().mean())))
    print("of the mismatched, got==0 exactly: %d"
          % int((got[mism] == 0).sum()))
    print("of the mismatched, got non-finite: %d"
          % int((~torch.isfinite(got[mism])).sum()))
    print("got: zeros overall = %d / %d" % (int((got == 0).sum()), got.numel()))
    print("ref: zeros overall = %d / %d" % (int((ref == 0).sum()), ref.numel()))

    # Localise along the last (feature) dim and the sequence dim.
    D = ref.shape[-1]
    per_feat = mism.reshape(-1, D).any(dim=0)
    bad_feats = torch.nonzero(per_feat).flatten().tolist()
    print("\nfeature columns with any mismatch: %d / %d" % (len(bad_feats), D))
    if bad_feats:
        print("  first..last: %d .. %d" % (bad_feats[0], bad_feats[-1]))
        shard = D // args.world_size
        print("  shard width D/W = %d" % shard)
        for sid in range(args.world_size):
            lo, hi = sid * shard, (sid + 1) * shard
            n = int(per_feat[lo:hi].sum())
            print("    shard %d cols [%4d:%4d]: %4d/%4d mismatched"
                  % (sid, lo, hi, n, shard))

    S = ref.shape[1] if ref.dim() == 3 else 1
    per_tok = mism.reshape(-1, S, D).any(dim=2).reshape(-1)
    bad_tok = torch.nonzero(per_tok).flatten().tolist()
    print("\nsequence positions with any mismatch: %d / %d" % (len(bad_tok), S))
    if bad_tok:
        print("  first..last: %d .. %d" % (bad_tok[0], bad_tok[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
