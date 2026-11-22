"""
figs_kaggle.py -- Fig. 2 (sample data) and Fig. 4 (qualitative examples) as EPS.

CPU only (a few minutes). Inputs (edit the paths below):
    MAPS    output folder of code/02_attribution_maps.py
    FACES   output folder of code/01_extract_faces.py (fake/, masks/)
Uses the same 500 evaluation images (hash check), the same random pick of
example images (seed 1) and the same per-image soft IoU and corrected score as
code/03_scores.py. Fig. 4 shows the first two of the four picked images.
Images are embedded at their native 256 x 256 resolution; image data in the
EPS is compressed losslessly (Flate), text and lines stay vector.
Output: figs/fig_sample_data.eps, figs/fig_qualitative.eps
"""
import os, hashlib, json, re, zlib, base64
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("OUT", "figs")); OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
KERNEL = np.ones((3, 3), np.uint8)

# ----------------------------------------------------------------------------- EPS compression
_HEAD = re.compile(
    rb"(\d+) (\d+) 8 \[[^\]]*\]\s*\{\s*currentfile DataString readhexstring pop\s*\}"
    rb"\s*bind false 3 colorimage[ \t]*\r?\n")
_HEX = set(b"0123456789abcdefABCDEF")

def compress_eps(path):
    """Compress the image data of a matplotlib EPS losslessly (Flate); text and lines are unchanged."""
    data = open(path, "rb").read()
    out, pos, n = [], 0, 0
    for m in _HEAD.finditer(data):
        if m.start() < pos: continue
        w, h = int(m.group(1)), int(m.group(2))
        need, i, got, buf = 2 * w * h * 3, m.end(), 0, bytearray()
        while got < need:
            c = data[i]
            if c in _HEX: buf.append(c); got += 1
            elif c not in b" \t\r\n": raise ValueError(f"unexpected byte at {i}")
            i += 1
        a85 = base64.a85encode(zlib.compress(bytes.fromhex(buf.decode()), 9), wrapcol=76) + b"~>"
        head = re.sub(rb"\{\s*currentfile DataString readhexstring pop\s*\}\s*bind",
                      b"currentfile /ASCII85Decode filter /FlateDecode filter", data[m.start():m.end()])
        out += [data[pos:m.start()], head, a85, b"\n"]; pos, n = i, n + 1
    out.append(data[pos:]); new = b"".join(out)
    if n: open(path, "wb").write(new.replace(b"%%LanguageLevel: 2", b"%%LanguageLevel: 3"))
    print(f"  {path.name}: {n} images compressed, {len(data)/1e6:.2f} MB -> {len(new)/1e6:.2f} MB")

# ----------------------------------------------------------------------------- data
MAPS = Path(os.environ.get("MAPS", "/kaggle/input/saliency"))        # edit
FACES = Path(os.environ.get("FACES", "/kaggle/input/ffdf-faces-v4"))  # edit
DF_MASK, DF_FACE = FACES / "masks", FACES / "fake"

def load_mask_raw(mask_dir, sid):
    m = cv2.imread(str(mask_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    if m.shape != (256, 256): m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)

def n01(E):
    E = np.abs(np.asarray(E, np.float32))
    if E.shape != (256, 256): E = cv2.resize(E, (256, 256))
    return (E - E.min()) / (E.max() - E.min() + 1e-8)

yy, xx = np.mgrid[0:256, 0:256]
def gauss(sig): return np.exp(-((yy-128)**2 + (xx-128)**2) / (2.0*sig*sig)).astype(np.float32)

def get_df_map(m, sid):
    f = MAPS / m / f"{sid}.npy"
    return np.load(f).astype(np.float32) if f.exists() else None

ALL = sorted(p.stem for p in (MAPS / "gradcam").glob("*.npy"))
good = [s for s in ALL if (DF_MASK / f"{s}.png").exists()]
good = [s for s in good if (lambda m: m is not None and m.sum() > 0)(load_mask_raw(DF_MASK, s))]
IDS = sorted(np.random.default_rng(SEED).choice(good, size=min(500, len(good)), replace=False).tolist())
h = hashlib.sha256("\n".join(IDS).encode()).hexdigest()[:16]
print(f"500 images: hash {h}  (paper: db2df1bb59227d5b)")
if os.environ.get("SMOKE_TEST") != "1":
    assert h == "db2df1bb59227d5b", "these are not the 500 evaluation images of the paper"
N = len(IDS)
RAW = np.stack([load_mask_raw(DF_MASK, s) for s in IDS])
DIL = np.stack([cv2.dilate(m, KERNEL, iterations=1) for m in RAW])
MF = DIL.reshape(N, -1).astype(np.float32)
VID = np.array([s.rsplit("_f", 1)[0] for s in IDS])
IDA = np.array([v.split("_")[0] for v in VID]); IDB = np.array([v.split("_")[1] for v in VID])
share = ((IDA[:, None] == IDA[None, :]) | (IDA[:, None] == IDB[None, :]) |
         (IDB[:, None] == IDA[None, :]) | (IDB[:, None] == IDB[None, :]))
ALLOWED = ~np.eye(N, dtype=bool) & ~share                        # primary shuffle set

def soft_matrix(Emat):                                           # mass-matched soft IoU = c/(2-c)
    c = (Emat @ MF.T) / np.maximum(Emat.sum(1), 1e-12)[:, None]
    return c / (2.0 - c + 1e-12)

def raw_and_corrected(get_row):
    Emat = np.stack([n01(get_row(s)).ravel() for s in IDS])
    M = soft_matrix(Emat)
    own = M[np.arange(N), np.arange(N)]
    shf = (M * ALLOWED).sum(1) / ALLOWED.sum(1)
    return own, own - shf

# ----------------------------------------------------------------------------- figures (CELL 10 code)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "ps.fonttype": 42, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
def save(fig, name):
    p = OUT / f"{name}.eps"; fig.savefig(p); plt.close(fig); compress_eps(p)
def contour(ax, m, color="white"):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        c = c[:, 0, :]; ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=0.9)
def face(folder, s):
    im = cv2.imread(str(folder / f"{s}.png"))
    return None if im is None else cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
def blend(img, E):
    heat = plt.get_cmap("jet")(n01(E))[..., :3]
    return np.clip(0.5 * img + 0.5 * heat, 0, 1)

pick = sorted(np.random.default_rng(1).choice(N, 4, replace=False).tolist())   # same random pick as before
print("picked images:", [IDS[i] for i in pick], "| Fig. 4 uses the first 2")

# Fig. 2 -- sample data + mean of the 500 masks
fig, axs = plt.subplots(1, 5, figsize=(7.1, 1.6))
for a, i in zip(axs[:4], pick):
    img = face(DF_FACE, IDS[i]); a.imshow(img if img is not None else np.zeros((256, 256, 3)), interpolation="none")
    contour(a, RAW[i], "yellow"); a.set_title(f"mask area {RAW[i].mean():.2f}", fontsize=7); a.axis("off")
axs[4].imshow(RAW.astype(np.float32).mean(0), cmap="viridis", vmin=0, vmax=1, interpolation="none")
axs[4].plot(128, 128, "+", color="white", ms=6); axs[4].set_title(f"mean of {N} masks", fontsize=7); axs[4].axis("off")
save(fig, "fig_sample_data")

# Fig. 4 -- qualitative, first 2 of the 4 picked images
rows = pick[:2]
cols = [("Input + annotation", None), ("Grad-CAM", "gradcam"), ("SmoothGrad", "smoothgrad"),
        ("Fixed centre blob", "_centre_blob_s55")]
SC = {m: raw_and_corrected((lambda s: gauss(55)) if m.startswith("_") else (lambda s, m=m: get_df_map(m, s)))
      for _, m in cols if m}
fig, axs = plt.subplots(len(rows), 4, figsize=(7.1, 1.8 * len(rows)))
for r_, i in enumerate(rows):
    s = IDS[i]; img = face(DF_FACE, s)
    for c_, (title, m) in enumerate(cols):
        a = axs[r_, c_]
        if m is None: a.imshow(img, interpolation="none")
        else:
            E = gauss(55) if m.startswith("_") else get_df_map(m, s)
            a.imshow(blend(img, E), interpolation="none")
            o, d = SC[m]
            a.text(4, 250, f"raw {o[i]:.2f} | corrected {d[i]:+.3f}", color="white", fontsize=6,
                   va="bottom", bbox=dict(facecolor="black", edgecolor="none", pad=1))
            print(f"  {s} {m:18s} raw {o[i]:.2f} corrected {d[i]:+.3f}")
        contour(a, RAW[i], "white"); a.axis("off")
        if r_ == 0: a.set_title(title, fontsize=8)
save(fig, "fig_qualitative")
print("\nDone ->", OUT)
print("Values in the paper: 0.83/-0.093, 0.19/-0.083, 0.38/-0.091 | 0.89/-0.026, 0.21/-0.076, 0.40/-0.074")
