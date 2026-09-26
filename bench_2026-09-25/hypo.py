import os, sys
REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch, torch.nn.functional as F
import gpt2_multi_size_speedup_cli as G

torch.manual_seed(0)
p = G.MODEL_PRESETS["117m"]
m = G.GPT2PreLN(vocab_size=G.DEF_VOCAB, context_size=1024, d_model=p.d_model,
                num_layers=1, num_heads=p.num_heads, head_dim=p.head_dim,
                d_ffn=p.d_ffn).eval()
x = torch.randint(0, G.DEF_VOCAB, (1, 128), dtype=torch.long)

numa = torch.load("/tmp/mg_numa_w4.pt", weights_only=True).float().flatten()
std  = torch.load("/tmp/mg_std_w4.pt",  weights_only=True).float().flatten()

W = 4
blk = m.blocks[0]
A, B_ = blk.mlp_fc1_w, blk.mlp_fc2_w        # (768,3072), (3072,768)
b1, b2 = blk.mlp_fc1_b, blk.mlp_fc2_b
D = A.shape[1]; SD = D // W
sl = lambda i: slice(i*SD, (i+1)*SD)

def make_mlp(fn):
    def _mlp(self, xn, Bsz, S):
        xf = xn.reshape(Bsz*S, self.H_in)
        return fn(xf).view(Bsz, S, self.H_in)
    return _mlp

HYP = {}
HYP["H0_correct"]    = lambda xf: sum(F.gelu(torch.addmm(b1[sl(i)], xf, A[:, sl(i)])) @ B_[sl(i), :] for i in range(W)) + b2
HYP["H1_shard0only"] = lambda xf: F.gelu(torch.addmm(b1[sl(0)], xf, A[:, sl(0)])) @ B_[sl(0), :] + b2
HYP["H2_gelu0_allB"] = lambda xf: sum(F.gelu(torch.addmm(b1[sl(0)], xf, A[:, sl(0)])) @ B_[sl(i), :] for i in range(W)) + b2
HYP["H3_allgelu_B0"] = lambda xf: sum(F.gelu(torch.addmm(b1[sl(i)], xf, A[:, sl(i)])) @ B_[sl(0), :] for i in range(W)) + b2
HYP["H4_4x_shard0"]  = lambda xf: W * (F.gelu(torch.addmm(b1[sl(0)], xf, A[:, sl(0)])) @ B_[sl(0), :]) + b2

orig = G.GPT2TransformerBlock._mlp
def corr(a, b):
    ac = a - a.mean(); bc = b - b.mean()
    return float((ac*bc).sum() / (ac.norm()*bc.norm()))

for name, fn in HYP.items():
    G.GPT2TransformerBlock._mlp = make_mlp(fn)
    with torch.no_grad():
        out = m(x).detach().float().flatten()
    print("%-16s corr_vs_NUMA=%+.6f  corr_vs_STD=%+.6f  maxabs_vs_NUMA=%.3e"
          % (name, corr(out, numa), corr(out, std), (out-numa).abs().max()))
G.GPT2TransformerBlock._mlp = orig
