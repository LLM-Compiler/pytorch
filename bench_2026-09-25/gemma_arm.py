"""Run ONE Gemma benchmark arm (standard or numa) in its own process.

Split-process for the same reason as split_arm.py: the NUMA arm needs room on a
SPECIFIC NUMA node, and a standard arm having run first in the same process
leaves pages placed so the needed node cannot satisfy it.

The model is built from a fixed seed, identically in both arms, so the
cross-process output comparison is meaningful. The stock Gemma CLI does NOT do
this -- it builds a fresh unseeded model per arm.
"""
import argparse, json, os, resource, sys, time, statistics

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--arm", required=True, choices=["standard", "numa"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=18)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        f"/tmp/gsw_{a.arm}_b{a.batch_size}_s{a.seq_length}")

    REPO = "/home/shavin/llm-cpu-compiler/pytorch"
    sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
    import torch, torch.nn as nn
    from torch._inductor import config as ic
    from simple_gemma_decoder_block_torch_compile_demo import (
        SimpleGemmaStack, _build_prefill_inputs)

    ic.max_autotune_gemm_backends = "TRITON"
    torch._C._cpu._set_numa_enabled(True)
    H_IN, NH, NKV, HD, MLP = 1152, 4, 1, 256, 6912

    torch.manual_seed(a.seed)
    model = SimpleGemmaStack(
        num_layers=a.num_layers, hidden_size=H_IN, num_heads=NH,
        num_kv_heads=NKV, head_dim=HD, intermediate_size=MLP,
        attn_logit_softcapping=None, use_kv_cache=False).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    torch.manual_seed(a.seed + 1)
    inputs = _build_prefill_inputs(
        a.batch_size, a.seq_length, H_IN, NKV, HD, a.num_layers,
        torch.device("cpu"), use_kv_cache=False)

    flags = {"_numa_attention_tp": False, "_numa_mlp_tp": False,
             "fx_graph_cache": False}
    if a.arm == "numa":
        flags.update({
            "_numa_attention_tp": True,
            "_numa_attention_tp_world_size": a.world_size,
            "_numa_attention_tp_compute_shard": 0,
            "_numa_mlp_tp": True,
            "_numa_mlp_tp_world_size": a.world_size,
            "_numa_mlp_tp_compute_shard": 0})

    with ic.patch(flags), torch.no_grad():
        torch._dynamo.reset()
        c = torch.compile(model)
        for _ in range(a.warmup):
            out = c(*inputs)
        lat = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            out = c(*inputs)
            lat.append((time.perf_counter() - t0) * 1e3)

    out = (out[0] if isinstance(out, (tuple, list)) else out)
    torch.save(out.detach().float().contiguous(), a.out_prefix + ".out.pt")
    rec = {"model": "gemma-270m", "batch": a.batch_size, "seq": a.seq_length,
           "world_size": a.world_size, "arm": a.arm,
           "mean_ms": statistics.fmean(lat), "median_ms": statistics.median(lat),
           "min_ms": min(lat), "iters": len(lat),
           "peak_rss_gb": round(
               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2)}
    with open(a.out_prefix + ".timing.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
