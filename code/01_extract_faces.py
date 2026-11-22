"""
01_extract_faces.py -- face crops and manipulation masks from FaceForensics++ (Deepfakes, c23).

Uses the official FaceForensics++ train/val/test split files
(https://github.com/ondyari/FaceForensics/tree/master/dataset/splits:
train.json, val.json, test.json), so the test videos were never seen by the
DeepfakeBench detector, which was trained on the official training split.

For every video: ten evenly spaced frames; face found with the OpenCV Haar
cascade (largest detection); square crop of side 1.3 x the larger side of the
detected box, clipped at the frame border; frames without a detection use the
median box of the video; crop resized to 256 x 256 (INTER_AREA). The mask video
is cut with exactly the same box, resized with nearest-neighbour interpolation
and binarised at 127.

Usage:
    python 01_extract_faces.py --root /path/to/data --limit 5     (quick test)
    python 01_extract_faces.py --root /path/to/data               (full run)

Expected layout under --root:
    FFpp/original_sequences/youtube/c23/videos/*.mp4
    FFpp/manipulated_sequences/Deepfakes/c23/videos/*.mp4
    FFpp/manipulated_sequences/Deepfakes/masks/videos/*.mp4
    splits/train.json, splits/val.json, splits/test.json
Output: <root>/ffdf_faces_v4/{real,fake,masks}/*.png and manifest.csv
"""

import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Official splits
# -----------------------------------------------------------------------------
def load_official_splits(splits_dir):
    """Each json file is a list of pairs [["000", "003"], ...]. A pair gives two
    fake videos ("000_003", "003_000") and two real videos ("000", "003"),
    all in the same split."""
    split_of = {}
    for name in ["train", "val", "test"]:
        f = Path(splits_dir) / f"{name}.json"
        if not f.exists():
            raise SystemExit(f"{f} not found. Download the split files from "
                             "github.com/ondyari/FaceForensics/tree/master/dataset/splits")
        pairs = json.loads(f.read_text())
        for a, b in pairs:
            split_of[a] = name
            split_of[b] = name
            split_of[f"{a}_{b}"] = name
            split_of[f"{b}_{a}"] = name
        print(f"  {name:5s} : {len(pairs)} pairs")
    return split_of


# -----------------------------------------------------------------------------
# Face detection and cropping
# -----------------------------------------------------------------------------
def build_detector():
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    d = cv2.CascadeClassifier(xml)
    if d.empty():
        raise SystemExit("Haar cascade could not be loaded")
    return d


def detect_face_box(det, frame, margin=0.30):
    """Square box around the largest detection, side = (1 + margin) x max(w, h)."""
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + margin)
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(cx - side/2)), max(0, int(cy - side/2))
    x2, y2 = min(W, int(cx + side/2)), min(H, int(cy + side/2))
    return None if (x2-x1 < 48 or y2-y1 < 48) else (x1, y1, x2, y2)


def process_video(vp, mp, det, img_dir, mask_dir, vid, label, n_frames, size):
    """Two passes: (1) read the selected frames and detect faces; (2) frames
    without a detection use the median box. The mask is cut with the same box."""
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return []

    want = set(np.linspace(0, total-1, min(n_frames, total)).astype(int).tolist())
    mcap = cv2.VideoCapture(str(mp)) if (mp and Path(mp).exists()) else None

    frames, masks, boxes, i = {}, {}, [], 0
    while True:
        ok, fr = cap.read()
        mf = None
        if mcap is not None:
            okm, mt = mcap.read(); mf = mt if okm else None
        if not ok:
            break
        if i in want:
            frames[i] = fr
            if mf is not None:
                masks[i] = mf
            b = detect_face_box(det, fr)
            if b: boxes.append(b)
        i += 1
    cap.release()
    if mcap: mcap.release()
    if not boxes:
        return []

    fb = tuple(int(v) for v in np.median(np.array(boxes), axis=0))

    rows = []
    for i in sorted(frames):
        x1, y1, x2, y2 = detect_face_box(det, frames[i]) or fb
        face = frames[i][y1:y2, x1:x2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        name = f"{vid}_f{i:04d}.png"
        out_img = Path(img_dir) / name
        cv2.imwrite(str(out_img), face)

        out_mask = ""
        if i in masks:
            m = masks[i][y1:y2, x1:x2]
            if m.size > 0:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                m = (m > 127).astype(np.uint8) * 255
                out_mask = str(Path(mask_dir) / name)
                cv2.imwrite(out_mask, m)

        rows.append({"path": str(out_img), "label": label, "video_id": vid,
                     "frame_idx": i, "mask_path": out_mask})
    return rows


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--frames_per_video", type=int, default=10)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    R = Path(args.root)
    real_dir = R / "FFpp/original_sequences/youtube/c23/videos"
    fake_dir = R / "FFpp/manipulated_sequences/Deepfakes/c23/videos"
    mask_dir = R / "FFpp/manipulated_sequences/Deepfakes/masks/videos"

    print("[1] Official splits:")
    split_of = load_official_splits(R / "splits")

    det = build_detector()
    out = R / "ffdf_faces_v4"
    for s in ["real", "fake", "masks"]:
        (out / s).mkdir(parents=True, exist_ok=True)

    real = sorted(real_dir.glob("*.mp4"))
    fake = sorted(fake_dir.glob("*.mp4"))
    print(f"[2] Found {len(real)} real and {len(fake)} fake videos")
    if args.limit:
        real, fake = real[:args.limit], fake[:args.limit]

    rows, skipped = [], 0
    for vp in tqdm(real, desc="real"):
        sp = split_of.get(vp.stem)
        if sp is None:
            skipped += 1; continue
        r = process_video(vp, None, det, out/"real", out/"masks",
                          vp.stem, 0, args.frames_per_video, args.size)
        for x in r: x["split"] = sp
        rows += r

    for vp in tqdm(fake, desc="fake"):
        sp = split_of.get(vp.stem)
        if sp is None:
            skipped += 1; continue
        mp = mask_dir / vp.name
        r = process_video(vp, mp if mp.exists() else None, det,
                          out/"fake", out/"masks",
                          vp.stem, 1, args.frames_per_video, args.size)
        for x in r: x["split"] = sp
        rows += r

    if skipped:
        print(f"[!] {skipped} videos are not in the official split files (skipped)")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nothing was extracted. Check the paths.")
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        assert not (set(df[df.split == a].video_id) & set(df[df.split == b].video_id))
    print("\n[OK] no video appears in more than one split")

    df.to_csv(out / "manifest.csv", index=False)
    print(f"\n[done] {len(df)} images -> {out/'manifest.csv'}")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
