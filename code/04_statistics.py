"""04_statistics.py -- statistics from per_image_scores.csv.gz (output of 03_scores.py).

Usage:  python 04_statistics.py results/per_image_scores.csv.gz results

Inference is on the mean corrected score (the reported quantity):
  cluster bootstrap over videos (B = 10,000) -> 95% CI and two-sided bootstrap p,
  Holm correction over the 15 methods (CAM and Grad-CAM reported as one row);
  sensitivity: the same bootstrap over source-target pairs.
Check: image-independent reference maps should come out non-significant.
Outputs: stats_main.csv, stats_pairwise.csv, cam_gradcam.json"""
import sys, numpy as np, pandas as pd, itertools, json
X = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
B = 10000
METHODS = ["cam","gradcam","gradcam++","scorecam","vanilla_grad","integrated_grad","smoothgrad","input_x_grad",
           "guided_bp","deconvolution","lrp","deeptaylor","excitationbp","occlusion","lime","kernelshap"]
TESTED = [m for m in METHODS if m != "gradcam"]        # CAM and Grad-CAM reported as one row
REFS = ["_positive_control", "_mean_mask_train", "_mean_mask_self", "_centre_blob_s55", "_uniform"]

def boot(d, clusters, seed=0):
    keys, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(keys)); c = np.bincount(inv, minlength=len(keys))
    r = np.random.default_rng(seed); idx = r.integers(0, len(keys), size=(B, len(keys)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())); p = max(p, 1.0 / B)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

rows = []
for (mp, metric), g in X.groupby(["map", "metric"]):
    g = g[g.valid]
    for scheme, col in [("primary", "prior_primary"), ("k10", "prior_k10"), ("k20", "prior_k20")]:
        d = (g.own - g[col]).values
        lo, hi, p = boot(d, g.video.values); plo, phi, pp = boot(d, g.pair.values)
        rows.append(dict(map=mp, metric=metric, scheme=scheme, n=len(g), n_videos=g.video.nunique(),
                         raw=g.own.mean(), prior=g[col].mean(), corrected=d.mean(),
                         share_pct=100 * g[col].mean() / g.own.mean(), ci_lo=lo, ci_hi=hi, p=p,
                         ci_pair_lo=plo, ci_pair_hi=phi, p_pair=pp))
T = pd.DataFrame(rows)
for (metric, scheme), idx in T.groupby(["metric", "scheme"]).groups.items():
    k = T.index.isin(idx) & T["map"].isin(TESTED)
    T.loc[k, "p_holm"] = holm(T.loc[k, "p"]); T.loc[k, "p_pair_holm"] = holm(T.loc[k, "p_pair"])
T.to_csv(f"{OUT}/stats_main.csv", index=False)

# pairwise (soft, primary), paired per image, video-cluster bootstrap, Holm over pairs
S = X[(X.metric == "soft")].assign(cd=lambda d: d.own - d.prior_primary)
P = []
for a, b in itertools.combinations(TESTED, 2):
    A = S[(S["map"] == a)].set_index("image_id"); Bm = S[(S["map"] == b)].set_index("image_id")
    v = A.valid & Bm.valid; d = (A["cd"] - Bm["cd"])[v]
    lo, hi, p = boot(d.values, A.video[v].values)
    P.append(dict(a=a, b=b, n=int(v.sum()), diff=d.mean(), ci_lo=lo, ci_hi=hi, p=p))
P = pd.DataFrame(P); P["p_holm"] = holm(P.p); P.to_csv(f"{OUT}/stats_pairwise.csv", index=False)

# CAM vs Grad-CAM equivalence per image
c = X[X["map"] == "cam"].set_index(["metric", "image_id"]); gc = X[X["map"] == "gradcam"].set_index(["metric", "image_id"])
eq = dict(max_abs_own_diff=float((c.own - gc.own).abs().max()),
          max_abs_corrected_diff=float(((c.own - c.prior_primary) - (gc.own - gc.prior_primary)).abs().max()))
json.dump(eq, open(f"{OUT}/cam_gradcam.json", "w"), indent=1)
print("CAM vs Grad-CAM:", eq)

pd.set_option("display.width", 250)
for metric in ["soft", "hit10", "hit30"]:
    t = T[(T.metric == metric) & (T.scheme == "primary") & T["map"].isin(REFS + TESTED)].sort_values("corrected", ascending=False)
    print(f"\n=== {metric} | primary ===")
    print(t[["map", "n", "raw", "prior", "share_pct", "corrected", "ci_lo", "ci_hi", "p", "p_holm", "ci_pair_lo", "ci_pair_hi", "p_pair_holm"]].round(4).to_string(index=False))
    r = t[t["map"].str.startswith("_") & (t["map"] != "_positive_control")]
    print("  VALIDITY (image-independent maps must be n.s.):", "PASS" if (r.p > 0.05).all() else "FAIL", list(r.p.round(3)))
    m = t[t["map"].isin(TESTED)]
    print(f"  significant after Holm: {int((m.p_holm < .05).sum())}/{len(m)} -> {list(m[m.p_holm < .05]['map'])}")
print(f"\nPairwise (soft): {int((P.p_holm < .05).sum())}/{len(P)} pairs differ after Holm")
