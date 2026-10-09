"""Run ONE benchmark arm (standard or numa) in its own process.

Why this exists: the stock CLI runs both arms in a single process. The NUMA
arm then has to find room for its socket-local clones on a SPECIFIC NUMA node
(~16 GB of the 62 GB), and the standard arm having run first leaves pages
placed (Linux first-touch) such that the needed node cannot satisfy the
request. Measured: gpt2-762m B=1 S=2048 FAILS in the combined process at
50.0 GB with "Could not move memory to a NUMA node", but PASSES standalone at
48.9 GB -- near-identical memory, opposite outcome.

Model construction is replicated EXACTLY as the CLI does it (enable numa,
manual_seed(seed), build, randint) so both processes produce bit-identical
weights and inputs. That is what makes the cross-process correctness
comparison meaningful -- the stock Gemma CLI gets this wrong by building a
fresh unseeded model per arm.

Timing uses the CLI's own _run_path, so numbers stay comparable with stage3.
"""
import argparse, importlib, json, os, resource, sys

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))

CLI = {
    "gpt2-117m": ("gpt2_multi_size_speedup_cli", "117m"),
    "gpt2-345m": ("gpt2_multi_size_speedup_cli", "345m"),
    "gpt2-762m": ("gpt2_multi_size_speedup_cli", "762m"),
    "bert-base": ("bert_multi_size_speedup_cli", "base"),
    "bert-large": ("bert_multi_size_speedup_cli", "large"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CLI))
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--arm", required=True, choices=["standard", "numa"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--out-prefix", required=True,
                    help="writes <prefix>.timing.json and <prefix>.out.pt")
    a = ap.parse_args()

    modname, preset_key = CLI[a.model]
    M = importlib.import_module(modname)
    import torch

    preset = M.MODEL_PRESETS[preset_key]
    vocab = a.vocab_size if a.vocab_size is not None else M.DEF_VOCAB

    # --- replicate the CLI's construction exactly ---
    M._enable_numa()
    torch.manual_seed(a.seed)
    ctx = max(a.seq_length, M.DEF_CONTEXT)
    common = dict(vocab_size=vocab, context_size=ctx, d_model=preset.d_model,
                  num_layers=preset.num_layers, num_heads=preset.num_heads,
                  head_dim=preset.head_dim, d_ffn=preset.d_ffn)
    if modname.startswith("bert"):
        model = M.BertEncoderModel(type_vocab_size=2, **common).eval()
    else:
        model = M.GPT2PreLN(**common).eval()
    input_ids = torch.randint(0, vocab, (a.batch_size, a.seq_length), dtype=torch.long)
    # --- end replication ---

    timing, out, info = M._run_path(
        model, input_ids,
        numa=(a.arm == "numa"), world_size=a.world_size,
        warmup=a.warmup, iters=a.iters,
    )

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    torch.save(out.detach().float().contiguous(), a.out_prefix + ".out.pt")
    rec = {"model": a.model, "batch": a.batch_size, "seq": a.seq_length,
           "world_size": a.world_size, "arm": a.arm,
           "mean_ms": timing.mean_ms, "median_ms": timing.median_ms,
           "min_ms": timing.min_ms, "iters": timing.iters,
           "peak_rss_gb": round(peak_gb, 2),
           "numa_args_cache_dir": info.get("cache_dir", "")}
    with open(a.out_prefix + ".timing.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
