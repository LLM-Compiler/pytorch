import sys, torch
a = torch.load(sys.argv[1], weights_only=True).float().flatten()
b = torch.load(sys.argv[2], weights_only=True).float().flatten()
ac = a - a.mean(); bc = b - b.mean()
print("%-46s vs %-46s  max_abs=%.4e  corr=%+.6f"
      % (sys.argv[1].split("/")[-1], sys.argv[2].split("/")[-1],
         (a-b).abs().max(), (ac*bc).sum()/(ac.norm()*bc.norm())))
