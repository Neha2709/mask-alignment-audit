"""
03_scores.py -- mask-alignment scores, reference maps and the shuffled correction.

CPU only. Inputs (edit the paths below):
    MAPS      output folder of 02_attribution_maps.py (MAPS/<method>/<image_id>.npy)
    FACES     output folder of 01_extract_faces.py (masks/ and manifest.csv)
Outputs in OUT:
    per_image_scores.csv.gz  one row per map x measure x image: own score, prior
                             (primary shuffle set, random 10, random 20), valid flag
    geometry.csv, blob_sigma.csv, pc_noise.csv, meta.json
Statistics (bootstrap, Holm, pairwise) are computed from per_image_scores.csv.gz
by 04_statistics.py.

Measures: mass-matched soft IoU (= c / (2 - c), c = share of the map's mass inside
the dilated mask) and top-k containment (k = 10, 30; exact rank among strictly
positive values). Masks are dilated once with a 3 x 3 kernel.
Primary shuffle set S_i: every other evaluation image that shares no identity
with image i.
"""
import os, json, time, hashlib
import numpy as np, pandas as pd, cv2
from pathlib import Path

MAPS = Path(os.environ.get("MAPS", "/kaggle/input/saliency"))        # edit
FACES = Path(os.environ.get("FACES", "/kaggle/input/ffdf-faces-v4"))  # edit
OUT = Path(os.environ.get("OUT", "results")); OUT.mkdir(parents=True, exist_ok=True)
DF_MASK, DF_MANIFEST = FACES / "masks", FACES / "manifest.csv"
SEED = 0
KERNEL = np.ones((3, 3), np.uint8)
METHODS = ["cam", "gradcam", "gradcam++", "scorecam", "vanilla_grad", "integrated_grad",
           "smoothgrad", "input_x_grad", "guided_bp", "deconvolution", "lrp",
           "deeptaylor", "excitationbp", "occlusion", "lime", "kernelshap"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ----------------------------------------------------------------------------- helpers
def load_mask_raw(mask_dir, sid):
    m = cv2.imread(str(mask_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    if m.shape != (256, 256): m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)

def n01(E):                                     # absolute value + min-max to [0, 1]
    E = np.abs(np.asarray(E, np.float32))
    if E.shape != (256, 256): E = cv2.resize(E, (256, 256))
    return (E - E.min()) / (E.max() - E.min() + 1e-8)

def topk_positive(En, k):                       # top-k% pixels among strictly positive values
    f = En.ravel(); pos = np.flatnonzero(f > 1e-6); n = max(1, int(k / 100.0 * f.size))
    if pos.size == 0: return np.array([], int)
    return pos if pos.size <= n else pos[np.argpartition(f[pos], -n)[-n:]]

def manifest_names(mf_path, split, label=1):
    df = pd.read_csv(mf_path)
    df = df[(df.split == split) & (df.label == label)]
    df = df[df.mask_path.fillna("").astype(str).str.strip() != ""]
    return sorted(str(p).replace("\\", "/").split("/")[-1][:-4] for p in df.path)

yy, xx = np.mgrid[0:256, 0:256]
def gauss(sig): return np.exp(-((yy-128)**2 + (xx-128)**2) / (2.0*sig*sig)).astype(np.float32)

class DS:
    """Mask-side data of the evaluation images."""
    def __init__(self, name, mask_dir, ids):
        self.name, self.ids = name, list(ids)
        self.N = len(self.ids); self.IX = {s: i for i, s in enumerate(self.ids)}
        self.RAW = np.stack([load_mask_raw(mask_dir, s) for s in self.ids])
        self.DIL = np.stack([cv2.dilate(m, KERNEL, iterations=1) for m in self.RAW])
        self.MF = self.DIL.reshape(self.N, -1).astype(np.float32)
        self.VID = np.array([s.rsplit("_f", 1)[0] for s in self.ids])
        self.IDA = np.array([v.split("_")[0] for v in self.VID])
        self.IDB = np.array([v.split("_")[1] for v in self.VID])
        self.PAIR = np.array(["_".join(sorted(v.split("_"))) for v in self.VID])
        oth = ~np.eye(self.N, dtype=bool)
        share = ((self.IDA[:, None] == self.IDA[None, :]) | (self.IDA[:, None] == self.IDB[None, :]) |
                 (self.IDB[:, None] == self.IDA[None, :]) | (self.IDB[:, None] == self.IDB[None, :]))
        self.ALLOWED = oth & ~share                       # primary shuffle set
        print(f"[{name}] images {self.N} | videos {len(set(self.VID))} | identity pairs "
              f"{len(set(self.PAIR))} | identities {len(set(self.IDA) | set(self.IDB))} | "
              f"shuffle set per image min {int(self.ALLOWED.sum(1).min())} max {int(self.ALLOWED.sum(1).max())}")

def score_maps(ds, Emat):
    """Emat: N x 65536 normalised maps (row i = map of image i).
    Returns soft / hit10 / hit30 as N x N matrices ([i, j] = map i vs mask j) and valid (N,)."""
    tot = Emat.sum(1)
    valid = (Emat > 1e-6).any(1)
    c = (Emat @ ds.MF.T) / np.maximum(tot, 1e-12)[:, None]
    out = dict(soft=c / (2.0 - c + 1e-12), valid=valid)
    for k in (10, 30):
        T = np.zeros_like(Emat)
        for i in range(Emat.shape[0]):
            idx = topk_positive(Emat[i], k)
            if idx.size: T[i, idx] = 1.0
        out[f"hit{k}"] = (T @ ds.MF.T) / np.maximum(T.sum(1), 1)[:, None]
    return out

def fixed_rows(ds, E): return np.repeat(n01(E).reshape(1, -1), ds.N, axis=0)

def corrected(ds, M, scheme="primary", sets=None):
    own = M[np.arange(ds.N), np.arange(ds.N)]
    if scheme == "primary":
        shf = (M * ds.ALLOWED).sum(1) / ds.ALLOWED.sum(1)
    else:                                               # explicit index sets
        shf = np.array([M[i, sets[i]].mean() for i in range(ds.N)])
    return own, shf, own - shf

# ----------------------------------------------------------------------------- the 500 evaluation images
def get_map(m, sid):
    f = MAPS / m / f"{sid}.npy"
    return np.load(f).astype(np.float32) if f.exists() else None

ALL = sorted(p.stem for p in (MAPS / "gradcam").glob("*.npy"))
good = [s for s in ALL if (DF_MASK / f"{s}.png").exists()]
good = [s for s in good if (lambda m: m is not None and m.sum() > 0)(load_mask_raw(DF_MASK, s))]
IDS = sorted(np.random.default_rng(SEED).choice(good, size=min(500, len(good)), replace=False).tolist())
h = hashlib.sha256("\n".join(IDS).encode()).hexdigest()[:16]
print(f"pool (non-empty masks) {len(good)} of {len(ALL)} | sample {len(IDS)} | hash {h} (paper: db2df1bb59227d5b)")
if os.environ.get("SMOKE_TEST") != "1":
    assert h == "db2df1bb59227d5b", "these are not the 500 evaluation images of the paper"
DF = DS("deepfakes", DF_MASK, IDS)

def load_method_rows(get, ds):
    rows, miss = [], []
    for s in ds.ids:
        E = get(s)
        if E is None: miss.append(s); E = np.zeros((256, 256), np.float32)
        rows.append(n01(E).ravel())
    return np.stack(rows), miss

DF_RES, MISSING = {}, {}
for m in METHODS:
    Em, miss = load_method_rows(lambda s: get_map(m, s), DF)
    DF_RES[m] = score_maps(DF, Em); MISSING[m] = miss
    log(f"  {m:16s} valid {int(DF_RES[m]['valid'].sum())}  missing files {len(miss)}")

# ----------------------------------------------------------------------------- reference maps
# The order of the random draws below is kept as in our run (uniform map, random
# shuffle sets, positive-control noise), so that the same values are obtained.
rng = np.random.default_rng(SEED)
DMEAN_SELF = DF.DIL.astype(np.float32).mean(0)
UNIF = rng.random((256, 256)).astype(np.float32)
SHUF = {s: rng.choice([j for j in IDS if j != s], size=20, replace=False).tolist() for s in IDS}
PC05 = np.stack([(DF.DIL[i].astype(np.float32) + 0.05 * rng.random((256, 256)).astype(np.float32))
                 for i in range(DF.N)])
SH10 = [np.array([DF.IX[j] for j in SHUF[s][:10]]) for s in IDS]      # random 10 (sensitivity)
SH20 = [np.array([DF.IX[j] for j in SHUF[s]]) for s in IDS]           # random 20 (sensitivity)

DF_RES["_positive_control"] = score_maps(DF, np.stack([n01(x).ravel() for x in PC05]))
DF_RES["_centre_blob_s55"] = score_maps(DF, fixed_rows(DF, gauss(55)))
DF_RES["_mean_mask_self"] = score_maps(DF, fixed_rows(DF, DMEAN_SELF))
DF_RES["_uniform"] = score_maps(DF, fixed_rows(DF, UNIF))

def train_mean(mask_dir, manifest, label):
    """Mean of the dilated non-empty masks of the training split."""
    names = manifest_names(manifest, "train")
    acc, n, empty = np.zeros((256, 256), np.float64), 0, 0
    for s in names:
        m = load_mask_raw(mask_dir, s)
        if m is None: continue
        if m.sum() == 0: empty += 1; continue
        acc += cv2.dilate(m, KERNEL, iterations=1); n += 1
    vids = len(set(s.rsplit("_f", 1)[0] for s in names))
    print(f"[{label}] training masks: {n} non-empty used, {empty} empty skipped, {vids} training videos")
    assert n > 0
    return (acc / n).astype(np.float32), n, vids

DF_TRAIN_MEAN, DF_TRAIN_N, DF_TRAIN_V = train_mean(DF_MASK, DF_MANIFEST, "deepfakes")
assert not set(manifest_names(DF_MANIFEST, "train")) & set(IDS), "train and test overlap"
DF_RES["_mean_mask_train"] = score_maps(DF, fixed_rows(DF, DF_TRAIN_MEAN))

a, b = DF_RES["cam"], DF_RES["gradcam"]
same_valid = bool((a["valid"] == b["valid"]).all())
dmax = max(float(np.abs(a[k] - b[k])[np.ix_(a["valid"], np.ones(DF.N, bool))].max()) for k in ("soft", "hit10", "hit30"))
CAM_EQ = same_valid and dmax < 1e-4
print(f"CAM vs Grad-CAM: same valid images {same_valid} | max score difference over all image x mask pairs {dmax:.2e}")
TESTED = [m for m in METHODS if not (CAM_EQ and m == "gradcam")]

# ----------------------------------------------------------------------------- export per-image scores
def pc_rows(amp):
    rr = np.random.default_rng(100)
    return np.stack([n01(DF.DIL[i].astype(np.float32) + amp * rr.random((256, 256)).astype(np.float32)).ravel()
                     for i in range(DF.N)])
for amp in [0.0, 0.01, 0.10, 0.25, 0.50, 1.00]:
    DF_RES[f"_pc_noise_{amp:.2f}"] = score_maps(DF, pc_rows(amp))
for sig in [10, 20, 30, 45, 70, 90, 128]:
    DF_RES[f"_centre_blob_s{sig}"] = score_maps(DF, fixed_rows(DF, gauss(sig)))
rows = []
for name, sc in DF_RES.items():
    for metric in ("soft", "hit10", "hit30"):
        M = sc[metric]
        own, pri, _ = corrected(DF, M)
        _, p10, _ = corrected(DF, M, "sets", SH10)
        _, p20, _ = corrected(DF, M, "sets", SH20)
        rows.append(pd.DataFrame(dict(map=name, metric=metric, image_id=IDS, video=DF.VID, pair=DF.PAIR,
                                      valid=sc["valid"], own=own, prior_primary=pri, prior_k10=p10, prior_k20=p20)))
X = pd.concat(rows, ignore_index=True)
X.to_csv(OUT / "per_image_scores.csv.gz", index=False, compression="gzip")
print("rows:", len(X), "| maps:", X["map"].nunique())

# ----------------------------------------------------------------------------- geometry, blob width, positive-control noise
def geometry(ds):
    R_ = ds.RAW.reshape(ds.N, -1)
    fills, cy, cx = [], [], []
    for m in ds.RAW:
        ys, xs = np.where(m)
        fills.append(m.sum() / ((ys.max()-ys.min()+1) * (xs.max()-xs.min()+1))); cy.append(ys.mean()); cx.append(xs.mean())
    cy, cx = np.array(cy), np.array(cx); d = np.hypot(cy - 128, cx - 128)
    return dict(dataset=ds.name, n=ds.N, area_mean=R_.mean(1).mean(), area_median=np.median(R_.mean(1)),
                area_sd=R_.mean(1).std(), dil_area_mean=ds.MF.mean(1).mean(),
                rect_frac=float(np.mean(np.array(fills) > 0.999)), fill_mean=float(np.mean(fills)),
                centroid_y=cy.mean(), centroid_y_sd=cy.std(), centroid_x=cx.mean(), centroid_x_sd=cx.std(),
                dist_centre_median=np.median(d))

def blob_sweep(ds, methods_raw):
    rows = []
    for sig in [10, 20, 30, 45, 55, 70, 90, 128]:
        sc = score_maps(ds, fixed_rows(ds, gauss(sig))); o = sc["soft"][np.arange(ds.N), np.arange(ds.N)].mean()
        rows.append(dict(dataset=ds.name, sigma=sig, soft_raw=o,
                         hit10_raw=sc["hit10"][np.arange(ds.N), np.arange(ds.N)].mean(),
                         n_methods_beaten=int((methods_raw < o).sum()), n_methods=len(methods_raw)))
    return rows

def pc_noise(ds):
    rows = []
    for amp in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        rr = np.random.default_rng(100)
        sc = score_maps(ds, np.stack([n01(ds.DIL[i].astype(np.float32) + amp * rr.random((256, 256)).astype(np.float32)).ravel()
                                      for i in range(ds.N)]))
        o, sh, d = corrected(ds, sc["soft"]); rows.append(dict(dataset=ds.name, noise_amp=amp, raw=o.mean(),
                                                               prior=sh.mean(), corrected=d.mean()))
    return rows

raw_m = pd.Series({m: corrected(DF, DF_RES[m]["soft"])[0][DF_RES[m]["valid"]].mean() for m in TESTED})
pd.DataFrame([geometry(DF)]).to_csv(OUT / "geometry.csv", index=False)
pd.DataFrame(blob_sweep(DF, raw_m.values)).to_csv(OUT / "blob_sigma.csv", index=False)
pd.DataFrame(pc_noise(DF)).to_csv(OUT / "pc_noise.csv", index=False)
json.dump(dict(df_train_masks=DF_TRAIN_N, df_train_videos=DF_TRAIN_V, cam_gradcam_maxdiff=dmax,
               ids_hash=h, pool=len(good)), open(OUT / "meta.json", "w"), indent=1)
log("done ->", OUT)
