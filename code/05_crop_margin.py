"""
05_crop_margin.py -- crop-margin check (masks only).

Crops the same Deepfakes test videos again with margins 0.15, 0.30 and 0.50,
using the functions of 01_extract_faces.py (only the margin changes), and
measures the mask prior on the same 500 evaluation frames. Attribution maps
exist only for the 0.30 margin, so no method scores are computed here.

Check: the new 0.30 crops and masks must be identical to the original ones.

Usage:  python 05_crop_margin.py --root /path/to/data
Needs:  <root>/FFpp/... and <root>/splits/ (as for 01_extract_faces.py),
        <root>/ffdf_faces_v4/ (original crops), data/eval_image_ids.csv
Output: <root>/crop_m15, crop_m30, crop_m50 and <root>/crop_margin_results.csv
"""
import argparse, importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd, cv2

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v4", HERE / "01_extract_faces.py")
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
ORIG_BOX = v4.detect_face_box
MARGINS = [0.15, 0.30, 0.50]


def extract(R, fake_dir, mask_dir, out, margin, splits_wanted, img_dir_for):
    """process_video of 01_extract_faces.py on every video; only the crop margin changes."""
    v4.detect_face_box = lambda det, frame, margin=margin: ORIG_BOX(det, frame, margin=margin)
    split_of = v4.load_official_splits(R / "splits")
    det = v4.build_detector()
    (out / "masks").mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(fake_dir).glob("*.mp4"))
    rows, t0 = [], time.time()
    todo = [vp for vp in vids if split_of.get(vp.stem) in splits_wanted]
    print(f"  margin {margin:.2f}: {len(todo)} videos")
    for k, vp in enumerate(todo):
        sp = split_of[vp.stem]; img_dir = img_dir_for(sp); img_dir.mkdir(parents=True, exist_ok=True)
        mp = Path(mask_dir) / vp.name
        r = v4.process_video(vp, mp if mp.exists() else None, det, img_dir, out / "masks",
                             vp.stem, 1, 10, 256)
        for x in r: x["split"] = sp
        rows += r
        if (k + 1) % 50 == 0: print(f"    {k+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
    v4.detect_face_box = ORIG_BOX
    df = pd.DataFrame(rows); df.to_csv(out / "manifest.csv", index=False)
    print(f"  done: {len(df)} crops, {int((df.mask_path != '').sum())} masks  ({time.time()-t0:.0f}s)")
    return df


def load_mask(folder, s):
    m = cv2.imread(str(folder / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    if m.shape != (256, 256): m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)

def fill(m):
    ys, xs = np.where(m)
    return m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))

yy, xx = np.mgrid[0:256, 0:256]
def gauss(s): return np.exp(-((yy-128)**2 + (xx-128)**2) / (2.0*s*s)).astype(np.float32)
def n01(E): E = np.abs(E.astype(np.float32)); return (E - E.min()) / (E.max() - E.min() + 1e-8)


def prior_stats(ids, folder, label):
    """From the masks only: area, rectangle check, centroid, raw score of the
    model-free maps, and the positive control (own mask) before and after the
    shuffled correction (primary set: all other images that share no identity)."""
    raw = [load_mask(folder, s) for s in ids]
    miss = sum(m is None for m in raw)
    keep = [i for i, m in enumerate(raw) if m is not None and m.sum() > 0]
    empty = len(ids) - miss - len(keep)
    ids_k = [ids[i] for i in keep]; RAW = np.stack([raw[i] for i in keep])
    N = len(ids_k)
    DIL = np.stack([cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1) for m in RAW])
    MF = DIL.reshape(N, -1).astype(np.float32); A = MF.sum(1)
    vid = np.array([s.rsplit("_f", 1)[0] for s in ids_k])
    a_ = np.array([v.split("_")[0] for v in vid]); b_ = np.array([v.split("_")[1] for v in vid])
    share = ((a_[:, None] == a_[None, :]) | (a_[:, None] == b_[None, :]) |
             (b_[:, None] == a_[None, :]) | (b_[:, None] == b_[None, :]))
    ALW = ~share                                         # the diagonal is also excluded
    def fixed_soft(E):                                   # one fixed map vs every mask
        En = n01(E).ravel(); c = (MF @ En) / En.sum(); return c / (2 - c)
    def corr_matrix(S):                                  # S[i, j] = map i vs mask j
        own = np.diag(S); prior = (S * ALW).sum(1) / ALW.sum(1); return own.mean(), prior.mean()
    out = dict(set=label, n_eval=N, n_missing=miss, n_empty=empty,
               area_mean=RAW.reshape(N, -1).mean(1).mean(), area_median=float(np.median(RAW.reshape(N, -1).mean(1))),
               rect_frac=float(np.mean([fill(m) > 0.999 for m in RAW])))
    cy = np.array([np.where(m)[0].mean() for m in RAW]); cx = np.array([np.where(m)[1].mean() for m in RAW])
    out.update(centroid_y=cy.mean(), centroid_x=cx.mean(), centroid_sd=float(np.hypot(cy.std(), cx.std())))
    best = (None, -1)
    for sg in [10, 20, 30, 55, 90]:
        v = fixed_soft(gauss(sg)).mean(); out[f"blob{sg}_soft_raw"] = v
        if v > best[1]: best = (sg, v)
    out["blob_best_sigma"] = best[0]
    idx = np.argpartition(gauss(55).ravel(), -int(0.1 * 65536))[-int(0.1 * 65536):]
    out["blob55_top10_containment"] = MF[:, idx].mean()
    S_ = DIL.astype(np.float32).sum(0)
    loo = np.stack([n01((S_ - DIL[i]) / (N - 1)).ravel() for i in range(N)])        # leave-one-out mean mask
    c = (loo @ MF.T) / loo.sum(1)[:, None]; out["meanmask_LOO_soft_raw"] = np.diag(c / (2 - c)).mean()
    out["uniform_soft_raw"] = fixed_soft(np.random.default_rng(0).random((256, 256))).mean()
    inter = MF @ MF.T; c = inter / A[:, None]; S = c / (2 - c)      # map = own dilated mask, no noise
    o, p = corr_matrix(S); out.update(pc_noisefree_raw=o, pc_noisefree_prior=p, pc_noisefree_corrected=o - p)
    rr = np.random.default_rng(100)
    E = np.stack([n01(DIL[i] + 0.05 * rr.random((256, 256)).astype(np.float32)).ravel() for i in range(N)])
    c = (E @ MF.T) / E.sum(1)[:, None]; o, p = corr_matrix(c / (2 - c))
    out.update(pc05_raw=o, pc05_prior=p, pc05_corrected=o - p)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); args = ap.parse_args()
    R = Path(args.root); FFPP = R / "FFpp"
    ids = pd.read_csv(HERE.parent / "data" / "eval_image_ids.csv").image_id.astype(str).tolist()
    assert len(ids) == 500
    fake_dir = FFPP / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    mask_dir = FFPP / "manipulated_sequences" / "Deepfakes" / "masks" / "videos"
    for m in MARGINS:
        out = R / f"crop_m{int(round(m*100)):02d}"
        if (out / "manifest.csv").exists(): print(f"  {out.name}: already exists, skipped"); continue
        extract(R, fake_dir, mask_dir, out, m, {"test"}, lambda sp, out=out: out / "fake")
    old, new = R / "ffdf_faces_v4", R / "crop_m30"          # check: margin 0.30 = original crops
    same = diff = missing = 0
    for sub in ["fake", "masks"]:
        for f in sorted((new / sub).glob("*.png")):
            b = cv2.imread(str(old / sub / f.name))
            if b is None: missing += 1; continue
            a = cv2.imread(str(f)); ok = a.shape == b.shape and (a == b).all()
            same += int(ok); diff += int(not ok)
    print(f"\nmargin 0.30 vs original crops: same {same} | different {diff} | missing {missing}")
    assert diff == 0 and missing == 0 and same > 0
    res = []
    for m in MARGINS:
        d = prior_stats(ids, R / f"crop_m{int(round(m*100)):02d}" / "masks", f"margin_{m:.2f}"); d["margin"] = m; res.append(d)
    T = pd.DataFrame(res); T.to_csv(R / "crop_margin_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.set_index("set").T.round(4).to_string())


if __name__ == "__main__":
    main()
