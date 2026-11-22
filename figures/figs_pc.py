"""
figs_pc.py -- Fig. 1 (workflow) and Fig. 3 (raw and corrected scores) as vector EPS.

Requirements: pip install matplotlib pandas numpy
Input:  results/stats_main.csv (output of code/04_statistics.py)
Usage:  python figures/figs_pc.py        (run from the repository folder)
Output: fig_workflow.eps, fig_scores.eps
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "ps.fonttype": 42, "pdf.fonttype": 42})

# ----------------------------------------------------------------------------- Fig. 1 workflow
W, H = 3.45, 1.0
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
boxes = [("1. Data", "#1f3a5f", "FF++ DF c23\ntest split\n500 crops,\nbox masks"),
         ("2. Maps", "#1f3a5f", "frozen\nXception;\nattribution\nmethods"),
         ("3. Controls", "#2e6b4a", "uniform,\nblob, mean\nmask, positive\ncontrol"),
         ("4. Scores", "#a8322d", "soft IoU,\ntop-k; own\nand other\nmasks"),
         ("5. Correction", "#a8322d", "own minus\nprior; video\nbootstrap;\nHolm")]
n = len(boxes); gap = 0.07; bw = (W - 0.06 - gap * (n - 1)) / n; y0 = 0.06; bh = H - 0.12
for i, (t, c, b) in enumerate(boxes):
    x = 0.03 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=c, lw=1.0))
    ax.text(x + bw / 2, y0 + bh - 0.1, t, ha="center", va="center", fontsize=5.6, fontweight="bold", color=c)
    ax.text(x + bw / 2, y0 + bh / 2 - 0.08, b, ha="center", va="center", fontsize=5.2, linespacing=1.15)
    if i < n - 1:
        ax.annotate("", xy=(x + bw + gap - 0.005, y0 + bh / 2), xytext=(x + bw + 0.005, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.8, mutation_scale=6))
fig.savefig("fig_workflow.eps"); plt.close(fig)
print("fig_workflow.eps saved")

# ----------------------------------------------------------------------------- Fig. 3 scores
T = pd.read_csv("results/stats_main.csv")
T = T[(T.metric == "soft") & (T.scheme == "primary")].set_index("map")
NAMES = {"cam": "CAM / Grad-CAM", "gradcam++": "Grad-CAM++", "scorecam": "Score-CAM", "lrp": "LRP",
         "deeptaylor": "DeepTaylor", "guided_bp": "Guided BP", "occlusion": "Occlusion",
         "vanilla_grad": "Gradient", "lime": "LIME", "excitationbp": "Excitation BP",
         "input_x_grad": "Input×Grad", "integrated_grad": "Integrated Grad",
         "smoothgrad": "SmoothGrad", "kernelshap": "KernelSHAP", "deconvolution": "Deconvolution"}
M = T.loc[list(NAMES)].sort_values("raw", ascending=False)        # 15 methods (Grad-CAM = CAM row)
uni, blob, mean_m = (T.loc[k, "raw"] for k in ["_uniform", "_centre_blob_s55", "_mean_mask_train"])
print(M[["raw", "corrected", "ci_lo", "ci_hi", "p_holm"]].round(4).to_string())

y = np.arange(len(M))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw=dict(width_ratios=[1.05, 1]))
a1.barh(y, M.raw, color="#1f3a5f", height=0.65)
a1.axvline(uni, color="#555", ls=":", lw=1.2, label=f"Uniform noise ({uni:.3f})")
a1.axvline(blob, color="#c0392b", ls="--", lw=1.2, label=f"Centre blob, $\\sigma$=55 ({blob:.3f})")
a1.axvline(mean_m, color="#2e6b4a", ls="-.", lw=1.2, label=f"Mean training mask ({mean_m:.3f})")
a1.set_yticks(y); a1.set_yticklabels([NAMES[k] for k in M.index], fontsize=7)
a1.set_xlim(0, 1); a1.set_xlabel("raw soft IoU (published protocol)", fontsize=7)
a1.set_title("(a) Raw score and model-free reference maps", fontsize=8)
a1.legend(fontsize=6, loc="lower right", framealpha=1.0); a1.tick_params(labelsize=6.5)
sig = (M.p_holm < 0.05).values
for yi, (k, r), s in zip(y, M.iterrows(), sig):
    col = "#a8322d" if s else "#555"
    a2.errorbar(r.corrected, yi, xerr=[[r.corrected - r.ci_lo], [r.ci_hi - r.corrected]], fmt="o",
                ms=4, color=col, mfc=col if s else "white", mec=col, capsize=2, lw=1)
a2.axvline(0, color="k", ls=":", lw=0.7)
a2.set_yticks(y); a2.set_yticklabels([]); a2.set_ylim(a1.get_ylim())
a2.set_xlabel("corrected soft IoU (own minus shuffled prior), 95% CI", fontsize=7)
a2.set_title("(b) Corrected score", fontsize=8); a2.tick_params(labelsize=6.5)
for a in (a1, a2):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("fig_scores.eps", bbox_inches="tight"); plt.close(fig)
print("fig_scores.eps saved")
