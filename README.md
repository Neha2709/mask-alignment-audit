# mask-alignment-audit

Code, evaluation image identifiers and per-image scores for the paper

> KM Neha Yadav, Shailesh Kamble, Nisha and Nikhil Mangrulkar,
> "Evaluating the Reliability of Mask-Alignment Metrics for Deepfake Explanations,"
> IEEE AIEI 2027 (to appear).

The paper asks how much of the mask-alignment ("coherence") score of a saliency
map on FaceForensics++ Deepfakes comes from the image itself, and how much a map
gets only because the annotated region is a large box near the centre of the
face crop. Every map is scored against its own mask and against the masks of
other images (shuffled baseline), and the difference is reported.

## Contents

| Folder | File | What it is |
|---|---|---|
| `code/` | `01_extract_faces.py` | Face crops and masks from the videos (Haar cascade, margin 0.30, 10 frames per video, official splits) |
| | `02_attribution_maps.py` | Maps of the 16 attribution methods for the manipulated test crops (GPU) |
| | `03_scores.py` | Soft IoU and top-k containment, reference maps, shuffled correction; writes the per-image scores |
| | `04_statistics.py` | Video-level cluster bootstrap, Holm correction, pairwise comparisons |
| | `05_crop_margin.py` | Crop-margin check (margins 0.15 / 0.30 / 0.50, masks only) |
| `figures/` | `figs_pc.py` | Fig. 1 and Fig. 3 (vector EPS) from `results/stats_main.csv` |
| | `figs_kaggle.py` | Fig. 2 and Fig. 4 (needs the face crops and the maps) |
| `data/` | `eval_image_ids.csv` | The 500 evaluation images (image id, video, target id, source id) |
| `results/` | `per_image_scores.csv.gz` | One row per map, measure and image (see below) |
| | `stats_main.csv`, `stats_pairwise.csv`, `cam_gradcam.json` | Output of `04_statistics.py` |
| | `geometry.csv`, `blob_sigma.csv`, `pc_noise.csv` | Mask geometry, centre-blob width, positive-control noise |
| | `crop_margin_results.csv` | Output of `05_crop_margin.py` |

No face images, masks, attribution maps or detector weights are included.

## Data and model

* **FaceForensics++** (Deepfakes, c23, videos and masks): available from its
  maintainers under their access agreement, <https://github.com/ondyari/FaceForensics>.
  The official split files (`train.json`, `val.json`, `test.json`) are in
  `dataset/splits` of that repository.
* **Detector**: the Xception checkpoint released with DeepfakeBench,
  <https://github.com/SCLBD/DeepfakeBench>. It is used in inference mode only.

## How to run

```
pip install -r requirements.txt
python code/01_extract_faces.py --root /path/to/data          # face crops + masks
python code/02_attribution_maps.py                            # GPU; edit the paths at the top
MAPS=/path/to/saliency FACES=/path/to/ffdf_faces_v4 OUT=results python code/03_scores.py
python code/04_statistics.py results/per_image_scores.csv.gz results
python code/05_crop_margin.py --root /path/to/data            # optional
python figures/figs_pc.py
```

`03_scores.py` checks that the 500 evaluation images are the ones used in the
paper (hash `db2df1bb59227d5b`). `04_statistics.py` run on the
`per_image_scores.csv.gz` in this repository gives the `stats_main.csv` and
`stats_pairwise.csv` in this repository.

## `per_image_scores.csv.gz`

| Column | Meaning |
|---|---|
| `map` | attribution method (e.g. `gradcam`) or reference map (names start with `_`) |
| `metric` | `soft` (mass-matched soft IoU), `hit10` / `hit30` (top-10% / top-30% containment) |
| `image_id`, `video`, `pair` | evaluation image, its video and its source-target pair |
| `valid` | `False` if the map has no positive value and cannot be scored |
| `own` | score against the image's own mask |
| `prior_primary` | mean score against the masks of all other evaluation images that share no identity with the image |
| `prior_k10`, `prior_k20` | the same against 10 / 20 randomly drawn images (sensitivity check) |

Corrected score = `own - prior_primary`.

## Notes

* CAM and Grad-CAM give per-image scores that differ by less than 1e-4; the paper
  reports them as one row, so 15 methods are tested (`04_statistics.py` does the same).
* SmoothGrad, LIME and KernelSHAP use a fixed seed (42). The other methods are
  deterministic.
* The attribution maps are not included because of their size; `02_attribution_maps.py`
  produces them.

## License

The code is released under the MIT License (see `LICENSE`).
