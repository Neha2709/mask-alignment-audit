"""
02_attribution_maps.py -- attribution maps of 16 methods for the manipulated test crops.

Run on a GPU (we used a Kaggle notebook). Requirements:
    pip install timm grad-cam captum zennit scikit-image

Inputs (edit the paths below):
    FACES     folder from 01_extract_faces.py (contains fake/, masks/, manifest.csv)
    WEIGHTS   DeepfakeBench Xception checkpoint (from the DeepfakeBench release;
              not redistributed here)
Output: OUT/<method>/<image_id>.npy  (256 x 256 map in [0, 1], float16)

Detector: single Xception classifier trained under the DeepfakeBench protocol,
used in inference mode (no weight is updated). Inputs scaled to [-1, 1]; every
map explains class 1 (fake). Target layer for the CAM-type methods: act4
(2048 x 8 x 8, last activation before global average pooling).

Note on our run: the twelve deterministic methods were produced in one run and
DeepTaylor, SmoothGrad, LIME and KernelSHAP in a second run with the
EpsilonGammaBox composite for DeepTaylor and a fixed seed (42) before every call.
This script uses the code of the second run for all sixteen methods; for the
deterministic methods the seed has no effect.
"""
import numpy as np, pandas as pd, cv2, timm, torch, gc
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

FACES = Path("/kaggle/input/ffdf-faces-v4")          # edit
WEIGHTS = [p for p in Path("/kaggle/input").rglob("*.pth") if "deepfakebench" in str(p).lower()][0]   # edit if needed
OUT = Path("/kaggle/working/saliency")
DEVICE = "cuda"
SEED = 42

# ----------------------------------------------------------------------------- model
sd = torch.load(WEIGHTS, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
new_sd = {k.replace("backbone.", "").replace("last_linear", "fc"): v
          for k, v in sd.items() if "adjust_channel" not in k}
model = timm.create_model("xception", pretrained=False, num_classes=2)
miss, unexp = model.load_state_dict(new_sd, strict=False)
assert len(miss) == 0 and len(unexp) == 0, f"weights do not match: {miss[:3]}"
model = model.to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(True)
TARGET_LAYER = model.act4

# ----------------------------------------------------------------------------- helpers
def load_image(path, size=256):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.
    img = (img - 0.5) / 0.5
    return torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def to_heatmap(attr, size=256):
    """Any attribution -> one 256 x 256 map: absolute value, sum over colour
    channels, resize, min-max to [0, 1]."""
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    attr = np.squeeze(attr)
    if attr.ndim == 3:
        attr = np.abs(attr).sum(axis=0)
    attr = np.abs(attr).astype(np.float32)
    if attr.shape != (size, size):
        attr = cv2.resize(attr, (size, size))
    return (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)


class ClassicCAM:
    """CAM: classifier weights of the target class x last activation, ReLU."""
    def __init__(self, model, layer):
        self.model, self.feat = model, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "feat", o.detach()))
    @torch.no_grad()
    def __call__(self, x, c=1):
        _ = self.model(x)
        W = self.model.get_classifier().weight[c].detach()
        return F.relu((W.view(-1, 1, 1) * self.feat[0]).sum(0)).cpu().numpy()


class ScoreCAMTopK:
    """Score-CAM on the top-32 channels (by mean activation), batch 16."""
    def __init__(self, model, layer, top_k=32, batch=16):
        self.model, self.top_k, self.batch, self.feat = model, top_k, batch, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "feat", o.detach()))
    @torch.no_grad()
    def __call__(self, x, c=1):
        _ = self.model(x)
        A = self.feat[0]
        k = min(self.top_k, A.shape[0])
        A_sel = A[torch.topk(A.mean(dim=(1, 2)), k).indices].unsqueeze(1)
        M = F.relu(F.interpolate(A_sel, size=x.shape[-2:], mode="bilinear", align_corners=False))
        flat = M.view(k, -1)
        mn, mx = flat.min(1, True)[0], flat.max(1, True)[0]
        M = ((flat - mn) / (mx - mn + 1e-8)).view(k, 1, *x.shape[-2:])
        w = torch.cat([torch.softmax(self.model(x * M[i:i+self.batch]), 1)[:, c]
                       for i in range(0, k, self.batch)])
        return F.relu((w.view(-1, 1, 1) * M.squeeze(1)).sum(0)).cpu().numpy()


def patch_mask(size=256, patch=32):
    """KernelSHAP features: 32 x 32 non-overlapping patches."""
    n = size // patch
    m = np.kron(np.arange(n*n).reshape(n, n), np.ones((patch, patch), dtype=int))
    return torch.tensor(m).unsqueeze(0).unsqueeze(0).to(DEVICE)


def slic_mask(x, n_seg=50):
    """LIME features: SLIC superpixels (n_segments = 50, compactness = 10)."""
    from skimage.segmentation import slic
    img = (x[0].detach().cpu().numpy().transpose(1, 2, 0)*0.5+0.5).clip(0, 1)
    seg = slic(img, n_segments=n_seg, compactness=10, start_label=0)
    return torch.tensor(seg.astype(np.int64)).unsqueeze(0).unsqueeze(0).to(DEVICE)


PATCH_MASK = patch_mask()

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import (Saliency, GuidedBackprop, Deconvolution, Occlusion, NoiseTunnel,
                         IntegratedGradients, InputXGradient, Lime, KernelShap)
from zennit.attribution import Gradient as ZGradient
from zennit.composites import EpsilonPlus, ExcitationBackprop, EpsilonGammaBox


class XAIToolbox16:
    def __init__(self, model, layer):
        self.model = model
        self.gradcam   = GradCAM(model=model, target_layers=[layer])
        self.gradcampp = GradCAMPlusPlus(model=model, target_layers=[layer])
        self.scorecam  = ScoreCAMTopK(model, layer)
        self.classic   = ClassicCAM(model, layer)
        self.sal       = Saliency(model)
        self.gbp       = GuidedBackprop(model)
        self.deconv    = Deconvolution(model)
        self.smooth    = NoiseTunnel(Saliency(model))
        self.ig        = IntegratedGradients(model)
        self.ixg       = InputXGradient(model)
        self.occ       = Occlusion(model)
        self.lime      = Lime(model)
        self.kshap     = KernelShap(model)

    def _flush_cam_buffers(self):
        """pytorch-grad-cam keeps the activations of every forward pass in a list
        that is only cleared when the CAM object itself is called; clear it after
        every call to avoid running out of memory."""
        for c in (self.gradcam, self.gradcampp):
            a = getattr(c, "activations_and_grads", None)
            if a is not None:
                a.activations = []
                a.gradients = []

    def explain(self, x, m, c=1):
        try:
            return self._explain(x, m, c)
        finally:
            self._flush_cam_buffers()

    def _explain(self, x, m, c=1):
        torch.manual_seed(SEED); np.random.seed(SEED)      # fixed seed before every call
        t = [ClassifierOutputTarget(c)]
        if m == "cam":            return to_heatmap(self.classic(x, c))
        if m == "gradcam":        return to_heatmap(self.gradcam(input_tensor=x, targets=t))
        if m == "gradcam++":      return to_heatmap(self.gradcampp(input_tensor=x, targets=t))
        if m == "scorecam":       return to_heatmap(self.scorecam(x, c))
        if m == "vanilla_grad":   return to_heatmap(self.sal.attribute(x, target=c, abs=True))
        if m == "guided_bp":      return to_heatmap(self.gbp.attribute(x, target=c))
        if m == "deconvolution":  return to_heatmap(self.deconv.attribute(x, target=c))
        if m == "input_x_grad":   return to_heatmap(self.ixg.attribute(x, target=c))
        if m == "integrated_grad":
            return to_heatmap(self.ig.attribute(x, target=c, baselines=x*0,
                                                n_steps=50, internal_batch_size=50))
        if m == "smoothgrad":
            return to_heatmap(self.smooth.attribute(x, target=c, nt_type="smoothgrad",
                                                    nt_samples=25, stdevs=0.15))
        if m == "lrp":
            with ZGradient(model=self.model, composite=EpsilonPlus()) as att:
                _, rel = att(x, torch.eye(2, device=x.device)[[c]])
            return to_heatmap(rel)
        if m == "deeptaylor":
            # input bounds in the normalised space: (0 - 0.5)/0.5 = -1, (1 - 0.5)/0.5 = 1
            with ZGradient(model=self.model, composite=EpsilonGammaBox(low=-1.0, high=1.0)) as att:
                _, rel = att(x, torch.eye(2, device=x.device)[[c]])
            return to_heatmap(rel)
        if m == "excitationbp":
            with ZGradient(model=self.model, composite=ExcitationBackprop()) as att:
                _, rel = att(x, torch.eye(2, device=x.device)[[c]])
            return to_heatmap(rel)
        if m == "occlusion":
            return to_heatmap(self.occ.attribute(x, target=c, sliding_window_shapes=(3, 32, 32),
                                                 strides=(3, 16, 16), baselines=x.mean().item(),
                                                 perturbations_per_eval=16))
        if m == "lime":
            return to_heatmap(self.lime.attribute(x, target=c, feature_mask=slic_mask(x),
                                                  n_samples=400, perturbations_per_eval=16,
                                                  show_progress=False))
        if m == "kernelshap":
            return to_heatmap(self.kshap.attribute(x, target=c, feature_mask=PATCH_MASK,
                                                   n_samples=200, perturbations_per_eval=16,
                                                   show_progress=False))
        raise ValueError(m)


METHODS = ["cam", "gradcam", "gradcam++", "scorecam", "vanilla_grad", "integrated_grad",
           "smoothgrad", "input_x_grad", "guided_bp", "deconvolution", "lrp", "deeptaylor",
           "excitationbp", "occlusion", "lime", "kernelshap"]

# ----------------------------------------------------------------------------- images
df = pd.read_csv(FACES / "manifest.csv")
df = df[(df.split == "test") & (df.label == 1)]
df = df[df.mask_path.fillna("").astype(str).str.strip() != ""]
IDS = sorted(str(p).replace("\\", "/").split("/")[-1][:-4] for p in df.path)
print(f"manipulated test crops with a mask file: {len(IDS)}")

toolbox = XAIToolbox16(model, TARGET_LAYER)
for m in METHODS:
    (OUT / m).mkdir(parents=True, exist_ok=True)
for sid in tqdm(IDS):
    x = load_image(FACES / "fake" / f"{sid}.png")
    for m in METHODS:
        f = OUT / m / f"{sid}.npy"
        if f.exists():
            continue
        np.save(f, toolbox.explain(x, m).astype(np.float16))
    gc.collect()
print("done ->", OUT)
