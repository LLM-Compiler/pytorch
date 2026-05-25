"""
Cache-Aware Roofline Model (CARM) Comparison
Standard PyTorch Inductor + Triton CPU  vs.  NUMA Tensor-Parallel Compilation
GPT-2 762M | B=1 | S=2048 | Dual-Socket CPU

Data source: Intel Advisor HTML roofline reports
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (extracted from Intel Advisor HTML roofline files)
# ─────────────────────────────────────────────────────────────────────────────

# ── Hardware bandwidth ceilings (derived from Intel Advisor program-level data)
# Standard: effective bandwidth = total memory traffic / elapsed time
#   L1  = 147785.43 GB / 1012.57 s  = 145.95 GB/s
#   L3  =   5105.39 GB / 1012.57 s  =   5.04 GB/s
#   DRAM =  1935.04 GB / 1012.57 s  =   1.91 GB/s   (hardware max = 2.02 GB/s)
# NUMA:  elapsed = 680.33 s
#   L1  = 148077.65 GB / 680.33 s   = 217.65 GB/s
#   L3  =   9183.08 GB / 680.33 s   =  13.50 GB/s
#   DRAM =  3078.26 GB / 680.33 s   =   4.52 GB/s   (2× sockets × 2.02 GB/s)

BW = {
    'std_dram':  1.91,
    'std_dram_hw': 2.02,
    'std_l3':    5.04,
    'std_l1':  145.95,
    'numa_dram':  4.52,
    'numa_dram_hw': 4.04,   # theoretical 2 × 2.02
    'numa_l3':   13.50,
    'numa_l1':  217.65,
}

# ── Compute peaks (highest observed GFLOPS for compute-bound intrinsic loops)
COMPUTE_PEAK_STD  = 240.0   # _mm256_fmadd peak seen at AI=4.0
COMPUTE_PEAK_NUMA = 550.0   # _mm256_fmadd peak seen at AI=4.0 (NUMA path)

# ── Program totals (programTotal from rooflineData)
prog_std_carm  = (0.2568, 37.48)   # (AI_L1,  GFLOPS)
prog_std_dram  = (19.615, 37.48)   # (AI_DRAM, GFLOPS)
prog_numa_carm = (0.2567, 55.87)
prog_numa_dram = (12.347, 55.87)

# ── Per-kernel hot loops  (AI=FLOP/Byte [L1 CARM], GFLOPS, selfTime[s], label)
# Standard
std_kernels = [
    # MLP kernels
    (0.2548, 131.47, 816.4,  "addmm_gelu\n(body)"),
    (0.2546, 106.60, 1002.3, "addmm_gelu\n(func)"),
    (0.2548, 137.57, 788.2,  "addmm_1\n(body)"),
    # Attention kernels
    (0.2538, 114.36, 364.9,  "bmm_qkv"),
    (0.2475, 140.07, 308.8,  "bmm_attn"),
    # Softmax (multiple AI points from different loop levels)
    (0.0640,  10.69, 122.2,  "softmax\n(outer)"),
    (0.7574,  68.95,   8.3,  "softmax\n(mid)"),
    (12.850,  84.22,   2.7,  "softmax\n(inner)"),
    # Layer-norm
    (0.0892,  11.45,   3.1,  "layer_norm"),
]

# NUMA
numa_kernels = [
    # MLP kernels (NUMA-split)
    (0.2548, 170.25, 1339.8, "addmm_gelu\n(body)"),
    (0.2544, 158.43, 1439.8, "addmm_gelu\n(func)"),
    (0.2548, 175.07,  971.0, "addmm_1\n(body)"),
    (0.2548, 190.55,  297.4, "addmm_8\n(body)"),
    # Attention kernels
    (0.2538, 136.54,  667.3, "bmm_qkv"),
    (0.2475, 207.60,  440.7, "bmm_attn"),
    # Softmax
    (0.0640,  14.52,  188.4, "softmax\n(outer)"),
    (0.7574, 116.85,   10.0, "softmax\n(mid)"),
    (12.850, 178.29,    2.9, "softmax\n(inner)"),
    # Layer-norm
    (0.0892,  18.31,    8.5, "layer_norm"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw a bandwidth roofline + horizontal compute ceiling
# ─────────────────────────────────────────────────────────────────────────────
def roofline_y(ai, bw, comp_peak):
    """Roofline model: min(bw*AI, comp_peak)"""
    return np.minimum(bw * ai, comp_peak)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: COMPREHENSIVE CARM COMPARISON  (main figure for the paper)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 9))

AI = np.logspace(-2.5, 3.5, 2000)

# ── Roofline ceilings ──────────────────────────────────────────────────────

# Standard DRAM (hardware measured maximum)
y_std_dram_hw = roofline_y(AI, BW['std_dram_hw'], COMPUTE_PEAK_STD)
ax.plot(AI, y_std_dram_hw, '--', color='#D62728', lw=2.2, zorder=3,
        label=f"Std DRAM ceiling  ({BW['std_dram_hw']:.2f} GB/s)")

# NUMA DRAM (2× sockets, hardware maximum)
y_numa_dram_hw = roofline_y(AI, BW['numa_dram_hw'], COMPUTE_PEAK_NUMA)
ax.plot(AI, y_numa_dram_hw, '--', color='#1F77B4', lw=2.2, zorder=3,
        label=f"NUMA DRAM ceiling ({BW['numa_dram_hw']:.2f} GB/s, 2× sockets)")

# Standard L3
y_std_l3 = roofline_y(AI, BW['std_l3'], COMPUTE_PEAK_STD)
ax.plot(AI, y_std_l3, ':', color='#FF7F0E', lw=1.8, zorder=3,
        label=f"Std L3 ceiling    ({BW['std_l3']:.2f} GB/s)")

# NUMA L3
y_numa_l3 = roofline_y(AI, BW['numa_l3'], COMPUTE_PEAK_NUMA)
ax.plot(AI, y_numa_l3, ':', color='#2CA02C', lw=1.8, zorder=3,
        label=f"NUMA L3 ceiling   ({BW['numa_l3']:.2f} GB/s)")

# Standard L1 (light)
y_std_l1 = roofline_y(AI, BW['std_l1'], COMPUTE_PEAK_STD)
ax.plot(AI, y_std_l1, '-', color='#D62728', lw=1.0, alpha=0.35, zorder=2,
        label=f"Std L1  ceiling   ({BW['std_l1']:.1f} GB/s)")

# NUMA L1 (light)
y_numa_l1 = roofline_y(AI, BW['numa_l1'], COMPUTE_PEAK_NUMA)
ax.plot(AI, y_numa_l1, '-', color='#1F77B4', lw=1.0, alpha=0.35, zorder=2,
        label=f"NUMA L1 ceiling   ({BW['numa_l1']:.1f} GB/s)")

# ── Shaded DRAM-bound region (left of standard DRAM ridge) ────────────────
dram_ridge_std = COMPUTE_PEAK_STD / BW['std_dram_hw']  # ~118.8 FLOP/B
# shade the region where the program is DRAM-bound (relative to std)
ax.axvspan(0.001, 5.0, alpha=0.06, color='#D62728', zorder=0, label='_nolegend_')
ax.text(0.013, 0.45, 'DRAM-\nBound\nRegion\n(Std)', fontsize=8, color='#D62728',
        alpha=0.7, ha='left', va='center', style='italic', transform=ax.transData,
        zorder=5)

# ── Program total operating points ──────────────────────────────────────────

# Standard CARM point (L1-view AI)
ax.scatter(*prog_std_carm, s=260, marker='*', color='#D62728', zorder=10,
           edgecolors='black', linewidths=0.8,
           label='Std Program Total (CARM/L1 view)')

# Standard DRAM point (DRAM-view AI)
ax.scatter(*prog_std_dram, s=200, marker='D', color='#D62728', zorder=10,
           edgecolors='black', linewidths=0.8,
           label='Std Program Total (DRAM view)')

# Connect the two standard points (same GFLOPS, different AI views)
ax.annotate('', xy=prog_std_dram, xytext=prog_std_carm,
            arrowprops=dict(arrowstyle='->', color='#D62728', lw=1.3, linestyle='dashed'))

# NUMA CARM point
ax.scatter(*prog_numa_carm, s=260, marker='*', color='#1F77B4', zorder=10,
           edgecolors='black', linewidths=0.8,
           label='NUMA Program Total (CARM/L1 view)')

# NUMA DRAM point
ax.scatter(*prog_numa_dram, s=200, marker='D', color='#1F77B4', zorder=10,
           edgecolors='black', linewidths=0.8,
           label='NUMA Program Total (DRAM view)')

ax.annotate('', xy=prog_numa_dram, xytext=prog_numa_carm,
            arrowprops=dict(arrowstyle='->', color='#1F77B4', lw=1.3, linestyle='dashed'))

# ── Per-kernel data points ─────────────────────────────────────────────────

max_time_std  = max(k[2] for k in std_kernels)
max_time_numa = max(k[2] for k in numa_kernels)

# Scale bubble size 30–300 proportional to sqrt(selfTime)
def bubble_size(t, max_t):
    return 35 + 280 * (np.sqrt(t) / np.sqrt(max_t))

# Standard kernels
for (ai, gflops, stime, lbl) in std_kernels:
    sz = bubble_size(stime, max_time_std)
    ax.scatter(ai, gflops, s=sz, color='#D62728', alpha=0.70,
               edgecolors='#8B0000', linewidths=0.7, zorder=8)

# NUMA kernels
for (ai, gflops, stime, lbl) in numa_kernels:
    sz = bubble_size(stime, max_time_numa)
    ax.scatter(ai, gflops, s=sz, color='#1F77B4', alpha=0.70,
               edgecolors='#003366', linewidths=0.7, zorder=8)

# Annotate selected key kernels
annotation_pairs = [
    # (std_idx, numa_idx, x_text, y_text_std, y_text_numa, label)
    (0, 0, 0.135, 105, 195, "addmm_gelu\n(MLP FC)"),
    (4, 5, 0.165, 115, 240, "bmm_attn\n(Attn)"),
    (5, 6, 0.043, 7.5, 18,  "softmax"),
]
for si, ni, xt, yst, ynt, lbl in annotation_pairs:
    xs, gs = std_kernels[si][0], std_kernels[si][1]
    xn, gn = numa_kernels[ni][0], numa_kernels[ni][1]
    # vertical arrow between std and numa
    ax.annotate('', xy=(xn, gn), xytext=(xs, gs),
                arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=1.8,
                                connectionstyle='arc3,rad=-0.3'))
    pct = (gn/gs - 1) * 100
    ax.text(xt, (gs*gn)**0.5, f'+{pct:.0f}%', fontsize=8.5, color='#2CA02C',
            ha='right', va='center', fontweight='bold', zorder=11)
    ax.text(xt * 0.82, (gs*gn)**0.5 * 0.55, lbl, fontsize=7.5, color='#555',
            ha='right', va='center', zorder=11)

# ── Vertical line at AI = 0.25 (ridge of main kernels) ─────────────────────
ax.axvline(x=0.255, color='gray', lw=1.0, linestyle=':', alpha=0.6, zorder=2)
ax.text(0.255, 1.2, 'Main kernel\nAI ≈ 0.25\nFLOP/B', fontsize=8, color='gray',
        ha='center', va='bottom', zorder=5)

# ── Speedup annotation ──────────────────────────────────────────────────────
ax.annotate('', xy=(0.257, prog_numa_carm[1]), xytext=(0.257, prog_std_carm[1]),
            arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
ax.text(0.38, (prog_std_carm[1] * prog_numa_carm[1])**0.5,
        f'1.49× speedup\n(program total)',
        fontsize=9, ha='left', va='center', color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.9))

# ── DRAM bandwidth annotation ───────────────────────────────────────────────
ax.text(22, 1.91*22 * 0.6, f"Std DRAM\n{BW['std_dram_hw']:.2f} GB/s", fontsize=8,
        color='#D62728', ha='left', va='top', style='italic')
ax.text(22, BW['numa_dram_hw']*22 * 0.7, f"NUMA DRAM\n{BW['numa_dram_hw']:.2f} GB/s\n(2× local)",
        fontsize=8, color='#1F77B4', ha='left', va='top', style='italic')

# ── Axis styling ────────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.008, 600)
ax.set_ylim(0.5, 8000)
ax.set_xlabel('Arithmetic Intensity  (FLOP / Byte)', fontsize=13, labelpad=8)
ax.set_ylabel('Performance  (GFLOPS)', fontsize=13, labelpad=8)
ax.set_title(
    'Cache-Aware Roofline Model Comparison\n'
    'Standard vs. NUMA Tensor-Parallel Compilation — GPT-2 762M (B=1, S=2048)',
    fontsize=13, fontweight='bold', pad=12
)
ax.grid(True, which='both', alpha=0.25, linestyle='-', linewidth=0.5)
ax.tick_params(labelsize=10)

# ── Legend ──────────────────────────────────────────────────────────────────
# Custom legend handles
legend_elements = [
    Line2D([0],[0], color='#D62728', lw=2.2, linestyle='--',
           label=f"Std DRAM ceiling ({BW['std_dram_hw']:.2f} GB/s, 1 socket)"),
    Line2D([0],[0], color='#1F77B4', lw=2.2, linestyle='--',
           label=f"NUMA DRAM ceiling ({BW['numa_dram_hw']:.2f} GB/s, 2 sockets)"),
    Line2D([0],[0], color='#FF7F0E', lw=1.8, linestyle=':',
           label=f"Std L3 ceiling ({BW['std_l3']:.2f} GB/s)"),
    Line2D([0],[0], color='#2CA02C', lw=1.8, linestyle=':',
           label=f"NUMA L3 ceiling ({BW['numa_l3']:.2f} GB/s)"),
    Line2D([0],[0], color='#D62728', lw=1.0, linestyle='-', alpha=0.4,
           label=f"Std L1 ceiling ({BW['std_l1']:.0f} GB/s)"),
    Line2D([0],[0], color='#1F77B4', lw=1.0, linestyle='-', alpha=0.4,
           label=f"NUMA L1 ceiling ({BW['numa_l1']:.0f} GB/s)"),
    plt.scatter([],[],s=180, marker='*', color='#D62728', edgecolors='black', lw=0.8,
                label='Std program total (★ CARM / ◆ DRAM view)'),
    plt.scatter([],[],s=150, marker='D', color='#D62728', edgecolors='black', lw=0.8,
                label='_nolegend_'),
    plt.scatter([],[],s=180, marker='*', color='#1F77B4', edgecolors='black', lw=0.8,
                label='NUMA program total (★ CARM / ◆ DRAM view)'),
    plt.scatter([],[],s=150, marker='D', color='#1F77B4', edgecolors='black', lw=0.8,
                label='_nolegend_'),
    plt.scatter([],[],s=100, color='#D62728', alpha=0.7, edgecolors='#8B0000', lw=0.7,
                label='Standard kernels (● size ∝ runtime)'),
    plt.scatter([],[],s=100, color='#1F77B4', alpha=0.7, edgecolors='#003366', lw=0.7,
                label='NUMA kernels (● size ∝ runtime)'),
    Line2D([0],[0], color='#2CA02C', lw=1.8, linestyle='-',
           marker='>', markersize=6, label='NUMA kernel GFLOPS improvement'),
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=8.5,
          framealpha=0.93, ncol=1, handlelength=2.5,
          columnspacing=1.0, labelspacing=0.5)

plt.tight_layout()
out1 = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_carm_comparison.pdf'
out1_png = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_carm_comparison.png'
plt.savefig(out1,     dpi=200, bbox_inches='tight')
plt.savefig(out1_png, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")
print(f"Saved: {out1_png}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: FOCUSED KERNEL COMPARISON  (zoomed on the key compute kernels)
# Highlights the DRAM-bound → higher-bandwidth DRAM transition for the
# dominant MLP/Attention kernels
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# ── Left panel: zoomed roofline around AI ≈ 0.25 ───────────────────────────
ax2 = axes[0]

AI2 = np.logspace(-1.5, 1.5, 2000)

# Draw rooflines
for bw_val, color, ls, lbl in [
    (BW['std_dram_hw'],  '#D62728', '--',  f"Std DRAM  ({BW['std_dram_hw']:.2f} GB/s)"),
    (BW['numa_dram_hw'], '#1F77B4', '--',  f"NUMA DRAM ({BW['numa_dram_hw']:.2f} GB/s)"),
    (BW['std_l3'],       '#FF7F0E', ':',   f"Std L3    ({BW['std_l3']:.2f} GB/s)"),
    (BW['numa_l3'],      '#2CA02C', ':',   f"NUMA L3   ({BW['numa_l3']:.2f} GB/s)"),
]:
    comp = COMPUTE_PEAK_STD if 'Std' in lbl else COMPUTE_PEAK_NUMA
    ax2.plot(AI2, roofline_y(AI2, bw_val, comp), ls, color=color, lw=2.2,
             label=lbl, zorder=4)

# Shade region below standard DRAM ceiling — the "inaccessible zone" for std
# At any AI, std is limited by DRAM = 2.02 * AI
y_std_dram_line = BW['std_dram_hw'] * AI2
y_numa_dram_line = BW['numa_dram_hw'] * AI2
ax2.fill_between(AI2, y_std_dram_line, y_numa_dram_line,
                 where=(AI2 < 15), alpha=0.13, color='#1F77B4',
                 label='NUMA bandwidth advantage\n(inaccessible for Standard)')

# Kernel-type groups for focused plot (only MLP/attention compute)
focus_std = [
    (0.2548, 131.47, 816.4,  "addmm_gelu_9\n(body)",   'v'),
    (0.2546, 106.60, 1002.3, "addmm_gelu_10\n(func)",  's'),
    (0.2548, 137.57, 788.2,  "addmm_1\n(body)",        '^'),
    (0.2538, 114.36, 364.9,  "bmm_qkv",               'o'),
    (0.2475, 140.07, 308.8,  "bmm_attn",              'D'),
    (0.0640,  10.69, 122.2,  "softmax (low-AI)",       'p'),
    (0.7574,  68.95,   8.3,  "softmax (mid-AI)",       'h'),
]
focus_numa = [
    (0.2548, 170.25, 1339.8, "addmm_gelu_10\n(body)",  'v'),
    (0.2544, 158.43, 1439.8, "addmm_gelu_11\n(func)",  's'),
    (0.2548, 175.07,  971.0, "addmm_1\n(body)",        '^'),
    (0.2548, 190.55,  297.4, "addmm_8\n(body)",        'P'),
    (0.2538, 136.54,  667.3, "bmm_qkv",               'o'),
    (0.2475, 207.60,  440.7, "bmm_attn",              'D'),
    (0.0640,  14.52,  188.4, "softmax (low-AI)",       'p'),
    (0.7574, 116.85,   10.0, "softmax (mid-AI)",       'h'),
]

kernel_colors = {
    'v': '#9B2335', 's': '#C0392B', '^': '#E74C3C', 'P': '#CB4335',
    'o': '#884EA0', 'D': '#6C3483', 'p': '#1A5276', 'h': '#154360',
}
kernel_colors_numa = {
    'v': '#1A5276', 's': '#154360', '^': '#2E86C1', 'P': '#1F618D',
    'o': '#117A65', 'D': '#0E6655', 'p': '#784212', 'h': '#6E2F1A',
}

max_t_std2  = max(k[2] for k in focus_std)
max_t_numa2 = max(k[2] for k in focus_numa)

for ai, gflops, stime, lbl, mk in focus_std:
    sz = 70 + 320*(stime/max_t_std2)**0.5
    ax2.scatter(ai, gflops, s=sz, marker=mk,
                color='#D62728', alpha=0.80, edgecolors='#8B0000', linewidths=0.9,
                zorder=8)

for ai, gflops, stime, lbl, mk in focus_numa:
    sz = 70 + 320*(stime/max_t_numa2)**0.5
    ax2.scatter(ai, gflops, s=sz, marker=mk,
                color='#1F77B4', alpha=0.80, edgecolors='#003366', linewidths=0.9,
                zorder=8)

# Label each kernel type (at median AI position)
kernel_label_map = {
    ('v', 'MLP FC2 (addmm_gelu)')  : (0.20, 83,  0.21, 230),
    ('s', 'MLP FC1 (addmm_gelu)')  : (0.20, 75,  0.21, 210),
    ('^', 'MLP FC (addmm_1)')      : (0.20, 90,  0.21, 240),
    ('o', 'Attn bmm QKV')          : (0.19, 70,  0.20, 190),
    ('D', 'Attn bmm Out')          : (0.19, 95,  0.20, 260),
    ('p', 'Softmax (low AI)')      : (0.050, 7,  0.052, 19),
    ('h', 'Softmax (mid AI)')      : (0.55, 47,  0.57, 140),
}
for (mk, lbl), (xs, ys, xn, yn) in kernel_label_map.items():
    ax2.annotate(lbl, xy=(xs, ys), fontsize=7.5, color='#D62728',
                 ha='right', va='center', zorder=11,
                 xytext=(xs*0.5, ys),
                 arrowprops=dict(arrowstyle='-', color='#D62728', lw=0.5, alpha=0.5))

# Vertical line at ridge AI
ax2.axvline(0.255, color='gray', lw=1.0, ls=':', alpha=0.6)
ax2.text(0.22, 3.5, "AI = 0.25\n(main kernels)", fontsize=8, color='gray',
         ha='center', va='bottom')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.03, 5.0)
ax2.set_ylim(3, 1500)
ax2.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
ax2.set_ylabel('Performance (GFLOPS)', fontsize=12)
ax2.set_title('Kernel-Level Roofline (Zoomed)\nMLP & Attention Compute Kernels', fontsize=12)
ax2.grid(True, which='both', alpha=0.25)
ax2.tick_params(labelsize=10)

leg_handles2 = [
    Line2D([0],[0], color='#D62728', lw=2, ls='--', label=f"Std DRAM ({BW['std_dram_hw']:.2f} GB/s)"),
    Line2D([0],[0], color='#1F77B4', lw=2, ls='--', label=f"NUMA DRAM ({BW['numa_dram_hw']:.2f} GB/s)"),
    Line2D([0],[0], color='#FF7F0E', lw=2, ls=':', label=f"Std L3 ({BW['std_l3']:.2f} GB/s)"),
    Line2D([0],[0], color='#2CA02C', lw=2, ls=':', label=f"NUMA L3 ({BW['numa_l3']:.2f} GB/s)"),
    mpatches.Patch(color='#1F77B4', alpha=0.2, label='NUMA bandwidth gain'),
    plt.scatter([],[],s=80, color='#D62728', alpha=0.8, edgecolors='#8B0000', lw=0.9,
                label='Standard kernels'),
    plt.scatter([],[],s=80, color='#1F77B4', alpha=0.8, edgecolors='#003366', lw=0.9,
                label='NUMA kernels'),
]
ax2.legend(handles=leg_handles2, loc='lower right', fontsize=9, framealpha=0.92)

# ── Right panel: bar chart of effective memory bandwidth per kernel ──────────
ax3 = axes[1]

# Pair up matching kernels (standard vs NUMA) by type
bw_pairs = [
    ("addmm_gelu\n(body)", 131.47/0.2548, 170.25/0.2548),
    ("addmm_gelu\n(func)", 106.60/0.2546, 158.43/0.2544),
    ("addmm_1\n(body)",    137.57/0.2548, 175.07/0.2548),
    ("bmm_qkv",            114.36/0.2538, 136.54/0.2538),
    ("bmm_attn",           140.07/0.2475, 207.60/0.2475),
    ("softmax\n(outer)",    10.69/0.0640,  14.52/0.0640),
    ("softmax\n(mid)",      68.95/0.7574, 116.85/0.7574),
    ("layer_norm",          11.45/0.0892,  18.31/0.0892),
]

labels_bw = [p[0] for p in bw_pairs]
std_bw    = [p[1] for p in bw_pairs]
numa_bw   = [p[2] for p in bw_pairs]

x = np.arange(len(labels_bw))
width = 0.38

bars1 = ax3.bar(x - width/2, std_bw,  width, color='#D62728', alpha=0.85,
                edgecolor='#8B0000', linewidth=0.8, label='Standard')
bars2 = ax3.bar(x + width/2, numa_bw, width, color='#1F77B4', alpha=0.85,
                edgecolor='#003366', linewidth=0.8, label='NUMA')

# Add horizontal lines for key hardware thresholds
ax3.axhline(y=BW['std_dram_hw'],  color='#D62728', lw=1.8, ls='--', alpha=0.8,
            label=f"Std DRAM HW max ({BW['std_dram_hw']:.2f} GB/s)")
ax3.axhline(y=BW['numa_dram_hw'], color='#1F77B4', lw=1.8, ls='--', alpha=0.8,
            label=f"NUMA DRAM HW max ({BW['numa_dram_hw']:.2f} GB/s)")
ax3.axhline(y=BW['std_l3'],  color='#FF7F0E', lw=1.6, ls=':', alpha=0.8,
            label=f"Std L3 ({BW['std_l3']:.2f} GB/s)")
ax3.axhline(y=BW['numa_l3'], color='#2CA02C', lw=1.6, ls=':', alpha=0.8,
            label=f"NUMA L3 ({BW['numa_l3']:.2f} GB/s)")

# Annotate % improvement
for i, (s, n) in enumerate(zip(std_bw, numa_bw)):
    pct = (n/s - 1)*100
    ax3.text(i + width/2, n + 6, f'+{pct:.0f}%', ha='center', va='bottom',
             fontsize=8.5, color='#1F77B4', fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(labels_bw, fontsize=9, rotation=30, ha='right')
ax3.set_ylabel('Effective Memory Bandwidth (GB/s)', fontsize=12)
ax3.set_title('Effective Bandwidth per Kernel Type\n(GFLOPS / Arithmetic Intensity)',
              fontsize=12)
ax3.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim(0, max(max(std_bw), max(numa_bw)) * 1.22)
ax3.tick_params(labelsize=9)

# Add region annotations
ax3.axhspan(0, BW['std_dram_hw'], alpha=0.06, color='#D62728', zorder=0)
ax3.text(len(labels_bw)-0.5, BW['std_dram_hw']*0.45,
         'DRAM-\nbound\n(Std)', fontsize=8, color='#D62728',
         ha='right', va='center', style='italic', alpha=0.9)

plt.suptitle(
    'NUMA Tensor-Parallel Compilation: Memory-Bandwidth Analysis\n'
    'GPT-2 762M | Batch=1 | Seq=2048 | Dual-Socket CPU (Intel Advisor)',
    fontsize=13, fontweight='bold', y=1.01
)

plt.tight_layout()
out2     = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_kernel_comparison.pdf'
out2_png = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_kernel_comparison.png'
plt.savefig(out2,     dpi=200, bbox_inches='tight')
plt.savefig(out2_png, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")
print(f"Saved: {out2_png}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: CLEAN SINGLE-PANEL (the "hero" figure for the paper)
# Shows the DRAM-bound → higher BW transition most clearly
# ─────────────────────────────────────────────────────────────────────────────

fig3, ax4 = plt.subplots(figsize=(11, 7.5))

AI3 = np.logspace(-2, 3, 3000)

# Hardware ceilings
ceiling_specs = [
    (BW['std_l1'],       '#C0392B', '-',  0.30, f"Std L1 ({BW['std_l1']:.0f} GB/s)"),
    (BW['numa_l1'],      '#2980B9', '-',  0.30, f"NUMA L1 ({BW['numa_l1']:.0f} GB/s)"),
    (BW['std_l3'],       '#E74C3C', ':',  1.6,  f"Std L3 ({BW['std_l3']:.2f} GB/s)"),
    (BW['numa_l3'],      '#2E86C1', ':',  1.6,  f"NUMA L3 ({BW['numa_l3']:.2f} GB/s)"),
    (BW['std_dram_hw'],  '#922B21', '--', 2.4,  f"Std DRAM ({BW['std_dram_hw']:.2f} GB/s / 1-socket)"),
    (BW['numa_dram_hw'], '#1A5276', '--', 2.4,  f"NUMA DRAM ({BW['numa_dram_hw']:.2f} GB/s / 2-socket local)"),
]

for bw_val, col, ls, lw, lbl in ceiling_specs:
    comp = COMPUTE_PEAK_NUMA if 'NUMA' in lbl else COMPUTE_PEAK_STD
    y = roofline_y(AI3, bw_val, comp)
    ax4.plot(AI3, y, ls, color=col, lw=lw, label=lbl, zorder=5,
             alpha=(1.0 if lw > 1 else 0.35))

# ── Shaded "DRAM-bound bandwidth gap" ─────────────────────────────────────
# The region between the standard and NUMA DRAM ceiling is "unlocked" by NUMA
y_std_dram_fill = BW['std_dram_hw'] * AI3
y_numa_dram_fill = BW['numa_dram_hw'] * AI3
mask = (AI3 >= 0.01) & (AI3 <= 100)
ax4.fill_between(AI3[mask], y_std_dram_fill[mask], y_numa_dram_fill[mask],
                 alpha=0.18, color='#27AE60', zorder=1,
                 label='Bandwidth gain (NUMA local DRAM)')

# ── Program total operating points ─────────────────────────────────────────
# Standard CARM and DRAM points
ax4.scatter(*prog_std_carm, s=350, marker='*', color='#C0392B', zorder=12,
            edgecolors='black', linewidths=1.0, label='Std: prog total (CARM view)')
ax4.scatter(*prog_std_dram, s=250, marker='D', color='#C0392B', zorder=12,
            edgecolors='black', linewidths=1.0, label='Std: prog total (DRAM view)')
ax4.annotate('', xy=prog_std_dram, xytext=prog_std_carm,
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, ls='dashed'))

# NUMA CARM and DRAM points
ax4.scatter(*prog_numa_carm, s=350, marker='*', color='#1A5276', zorder=12,
            edgecolors='black', linewidths=1.0, label='NUMA: prog total (CARM view)')
ax4.scatter(*prog_numa_dram, s=250, marker='D', color='#1A5276', zorder=12,
            edgecolors='black', linewidths=1.0, label='NUMA: prog total (DRAM view)')
ax4.annotate('', xy=prog_numa_dram, xytext=prog_numa_carm,
            arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.5, ls='dashed'))

# Annotate DRAM operating point labels
ax4.annotate(f"Std\n(DRAM: {BW['std_dram']:0.2f} GB/s used\n94.6% of peak)",
             xy=prog_std_dram, xytext=(prog_std_dram[0]*0.4, prog_std_dram[1]*0.28),
             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2),
             fontsize=8.5, color='#C0392B',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FADBD8', ec='#E74C3C', alpha=0.9))

ax4.annotate(f"NUMA\n(DRAM: {BW['numa_dram']:0.2f} GB/s used\n2.37× higher BW)",
             xy=prog_numa_dram, xytext=(prog_numa_dram[0]*0.4, prog_numa_dram[1]*3.0),
             arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.2),
             fontsize=8.5, color='#1A5276',
             bbox=dict(boxstyle='round,pad=0.35', fc='#D6EAF8', ec='#2E86C1', alpha=0.9))

# ── Hot kernel scatter ─────────────────────────────────────────────────────
# Only plot the highest-time kernels for clarity
dominant_std = [
    (0.2548, 131.47, 816.4),
    (0.2546, 106.60, 1002.3),
    (0.2548, 137.57, 788.2),
    (0.2538, 114.36, 364.9),
    (0.2475, 140.07, 308.8),
    (0.0640,  10.69, 122.2),
    (0.7574,  68.95,   8.3),
    (12.850,  84.22,   2.7),
]
dominant_numa = [
    (0.2548, 170.25, 1339.8),
    (0.2544, 158.43, 1439.8),
    (0.2548, 175.07,  971.0),
    (0.2548, 190.55,  297.4),
    (0.2538, 136.54,  667.3),
    (0.2475, 207.60,  440.7),
    (0.0640,  14.52,  188.4),
    (0.7574, 116.85,   10.0),
    (12.850, 178.29,    2.9),
]

mt_s = max(k[2] for k in dominant_std)
mt_n = max(k[2] for k in dominant_numa)

for ai, gflops, st in dominant_std:
    sz = 50 + 260*(st/mt_s)**0.5
    ax4.scatter(ai, gflops, s=sz, color='#E74C3C', alpha=0.75,
                edgecolors='#922B21', linewidths=0.9, zorder=9)

for ai, gflops, st in dominant_numa:
    sz = 50 + 260*(st/mt_n)**0.5
    ax4.scatter(ai, gflops, s=sz, color='#2E86C1', alpha=0.75,
                edgecolors='#1A5276', linewidths=0.9, zorder=9)

# Arrow between standard and NUMA clusters at AI=0.25
mid_std  = np.median([k[1] for k in dominant_std  if 0.20 < k[0] < 0.30])
mid_numa = np.median([k[1] for k in dominant_numa if 0.20 < k[0] < 0.30])
ax4.annotate('', xy=(0.28, mid_numa*1.05), xytext=(0.28, mid_std*0.95),
             arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2.5,
                             connectionstyle='arc3,rad=0.3'))
pct_k = (mid_numa/mid_std - 1)*100
ax4.text(0.45, (mid_std*mid_numa)**0.5,
         f'+{pct_k:.0f}%\nGFLOPS\n(kernel)', fontsize=9.5, color='#27AE60',
         ha='left', va='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', fc='#EAFAF1', ec='#27AE60', alpha=0.9))

# Vertical dashed line + region labels
ax4.axvline(0.255, color='gray', lw=1.0, ls=':', alpha=0.55, zorder=3)
ax4.text(0.255, 1.5, 'Main kernels\nAI ≈ 0.25 FLOP/B', fontsize=8, color='gray',
         ha='center', va='bottom')

# "DRAM-bound" annotation for the standard program total
ax4.text(0.015, 45, 'Standard: DRAM-bound\n(94.6% peak DRAM util.)',
         fontsize=9, color='#922B21', ha='left', va='center',
         bbox=dict(boxstyle='round', fc='#FADBD8', ec='#E74C3C', alpha=0.85))

# "Higher BW" annotation for NUMA
ax4.text(0.015, 75, 'NUMA: 2× local DRAM\n(2.37× bandwidth)',
         fontsize=9, color='#1A5276', ha='left', va='center',
         bbox=dict(boxstyle='round', fc='#D6EAF8', ec='#2E86C1', alpha=0.85))

# Overall speedup box
ax4.text(2.5, 1000,
         "Overall Speedup\n"
         f"  Elapsed: 1012.6s → 680.3s\n"
         f"  = 1.49× faster\n"
         f"  Program GFLOPS: {prog_std_carm[1]:.1f} → {prog_numa_carm[1]:.1f}",
         fontsize=9.5, ha='left', va='center',
         bbox=dict(boxstyle='round,pad=0.5', fc='#EAFAF1', ec='#27AE60', lw=1.5, alpha=0.95))

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(0.008, 500)
ax4.set_ylim(0.8, 5000)
ax4.set_xlabel('Arithmetic Intensity  (FLOP / Byte)', fontsize=13)
ax4.set_ylabel('Performance  (GFLOPS)', fontsize=13)
ax4.set_title(
    'Cache-Aware Roofline Model: Standard vs. NUMA Tensor-Parallel\n'
    'GPT-2 762M | B=1, S=2048 | Dual-Socket CPU | Intel Advisor Measurements',
    fontsize=12.5, fontweight='bold', pad=10
)
ax4.grid(True, which='both', alpha=0.22, linewidth=0.5)
ax4.tick_params(labelsize=10)

# Legend
handles3 = [
    Line2D([0],[0], color='#922B21', lw=2.4, ls='--',
           label=f"Std DRAM ({BW['std_dram_hw']:.2f} GB/s, 1-socket)"),
    Line2D([0],[0], color='#1A5276', lw=2.4, ls='--',
           label=f"NUMA DRAM ({BW['numa_dram_hw']:.2f} GB/s, 2-socket local)"),
    Line2D([0],[0], color='#E74C3C', lw=1.6, ls=':',
           label=f"Std L3 ({BW['std_l3']:.2f} GB/s)"),
    Line2D([0],[0], color='#2E86C1', lw=1.6, ls=':',
           label=f"NUMA L3 ({BW['numa_l3']:.2f} GB/s)"),
    mpatches.Patch(color='#27AE60', alpha=0.25,
                   label='NUMA bandwidth gain region'),
    plt.scatter([],[],s=120, color='#E74C3C', alpha=0.8, edgecolors='#922B21',
                label='Standard kernel hotspots'),
    plt.scatter([],[],s=120, color='#2E86C1', alpha=0.8, edgecolors='#1A5276',
                label='NUMA kernel hotspots'),
    plt.scatter([],[],s=200, marker='*', color='#C0392B', edgecolors='k',
                label='Std prog total ★(CARM) ◆(DRAM)'),
    plt.scatter([],[],s=200, marker='*', color='#1A5276', edgecolors='k',
                label='NUMA prog total ★(CARM) ◆(DRAM)'),
]
ax4.legend(handles=handles3, loc='upper left', fontsize=8.8,
           framealpha=0.93, labelspacing=0.45, handlelength=2.3)

plt.tight_layout()
out3     = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_hero.pdf'
out3_png = '/home/ubuntu/pytorch-private/scripts/inductor/advisor_b1_s2048_762M/reports/roofline_hero.png'
plt.savefig(out3,     dpi=200, bbox_inches='tight')
plt.savefig(out3_png, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")
print(f"Saved: {out3_png}")
print("\nAll figures generated successfully.")
