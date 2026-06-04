# =============================================================================
# XAI MODULE: Grad-CAM, Grad-CAM++, Integrated Gradients for 3-class CXR
# =============================================================================
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# If coming from your training script:
# DEVICE, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, test_loader, test_dataset must exist.

# -----------------------------------------------------------------------------
# Utility: de/normalization and visualization
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406] if 'IMAGENET_MEAN' not in globals() else IMAGENET_MEAN
IMAGENET_STD  = [0.229, 0.224, 0.225] if 'IMAGENET_STD'  not in globals() else IMAGENET_STD

def denormalize(img_tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img_tensor: (3,H,W) or (N,3,H,W) normalized.
    Returns float tensor in [0,1].
    """
    if img_tensor.dim() == 3:
        t = img_tensor.clone().cpu()
        for c in range(3):
            t[c] = t[c] * std[c] + mean[c]
        return t.clamp(0, 1)
    else:
        t = img_tensor.clone().cpu()
        for c in range(3):
            t[:, c] = t[:, c] * std[c] + mean[c]
        return t.clamp(0, 1)

def to_numpy_uint8(img_0_1: torch.Tensor):
    """
    img_0_1: (3,H,W) in [0,1] -> np.uint8 (H,W,3) BGR for OpenCV overlay convenience.
    """
    img = (img_0_1.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    # Keep RGB for matplotlib; for cv2.applyColorMap we convert later
    return img

def make_cam_overlay(rgb_img_uint8: np.ndarray, cam_0_1: np.ndarray, alpha: float = 0.35, colormap: int = cv2.COLORMAP_JET):
    """
    rgb_img_uint8: (H,W,3) RGB uint8
    cam_0_1: (H,W) float in [0,1]
    Returns RGB heatmap overlay (uint8).
    """
    heatmap = (cam_0_1 * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_img_uint8, 1.0, heatmap, alpha, 0)
    return overlay

def vis_triptych(rgb, heat, overlay, titles=('Input', 'Heatmap', 'Overlay'), figsize=(12,4)):
    plt.figure(figsize=figsize)
    plt.subplot(1,3,1); plt.imshow(rgb); plt.axis('off'); plt.title(titles[0])
    plt.subplot(1,3,2); plt.imshow(heat, cmap='jet'); plt.axis('off'); plt.title(titles[1])
    plt.subplot(1,3,3); plt.imshow(overlay); plt.axis('off'); plt.title(titles[2])
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Hook helper: find deepest Conv2d for CAM by default
# -----------------------------------------------------------------------------
def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model for CAM.")
    return last_conv

# -----------------------------------------------------------------------------
# Grad-CAM / Grad-CAM++
# -----------------------------------------------------------------------------
@dataclass
class CamOutputs:
    cam: np.ndarray        # (H,W) in [0,1]
    pred_class: int
    pred_score: float

class CamBase:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None, device=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.target_layer = target_layer if target_layer is not None else find_last_conv_layer(model)

        self._activations = None
        self._grads = None
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_activation(self, module, inp, out):
        # out: (N,C,H,W)
        self._activations = out.detach()

    def _save_grad(self, module, grad_input, grad_output):
        # grad_output[0]: (N,C,H,W)
        self._grads = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @torch.no_grad()
    def _upsample_cam(self, cam: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # (H,W)
        cam = cam.clamp(min=0)
        cam = cam / (cam.max() + 1e-8)
        return cam

    def _prepare_input(self, img_or_batch: torch.Tensor) -> torch.Tensor:
        """
        Accepts either (3,H,W) or (N,3,H,W) normalized to ImageNet.
        """
        if img_or_batch.dim() == 3:
            x = img_or_batch.unsqueeze(0)
        else:
            x = img_or_batch
        return x.to(self.device)

    def __call__(self, img: torch.Tensor, target_class: Optional[int] = None) -> CamOutputs:
        raise NotImplementedError


class GradCAM(CamBase):
    def __call__(self, img: torch.Tensor, target_class: Optional[int] = None) -> CamOutputs:
        x = self._prepare_input(img)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # (N,C)
        probs = F.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1)[0])
        score = float(probs[0, pred_class])
        tc = pred_class if target_class is None else int(target_class)

        # Backprop for the target class score
        loss = logits[:, tc].sum()
        loss.backward(retain_graph=True)

        # activations, grads: (N,C,H,W)
        A = self._activations
        dYdA = self._grads
        # weights: global-average-pool over spatial dims
        weights = dYdA.mean(dim=(2,3), keepdim=True)  # (N,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (N,1,H,W)

        cam_up = self._upsample_cam(cam, size=img.shape[-2:])
        return CamOutputs(cam=cam_up.cpu().numpy(), pred_class=pred_class, pred_score=score)


class GradCAMPlusPlus(CamBase):
    def __call__(self, img: torch.Tensor, target_class: Optional[int] = None) -> CamOutputs:
        x = self._prepare_input(img)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1)[0])
        score = float(probs[0, pred_class])
        tc = pred_class if target_class is None else int(target_class)

        # Backprop target
        loss = logits[:, tc].sum()
        loss.backward(retain_graph=True)

        A = self._activations           # (N,C,H,W)
        dYdA = self._grads              # (N,C,H,W)
        d2YdA2 = dYdA.pow(2)            # second-order approx
        d3YdA3 = dYdA.pow(3)            # third-order approx

        # alpha_ij as in Grad-CAM++ paper
        eps = 1e-6
        # sum over H,W
        sum_d2 = d2YdA2.sum(dim=(2,3), keepdim=True)  # (N,C,1,1)
        sum_A_d3 = (A * d3YdA3).sum(dim=(2,3), keepdim=True)

        alpha = d2YdA2 / (2.0 * d2YdA2 + sum_A_d3 + eps)  # (N,C,H,W)

        # weights: sum over spatial of alpha_ij * relu(grad_ij)
        weights = (alpha * F.relu(dYdA)).sum(dim=(2,3), keepdim=True)  # (N,C,1,1)

        cam = (weights * A).sum(dim=1, keepdim=True)  # (N,1,H,W)

        cam_up = self._upsample_cam(cam, size=img.shape[-2:])
        return CamOutputs(cam=cam_up.cpu().numpy(), pred_class=pred_class, pred_score=score)

# -----------------------------------------------------------------------------
# Integrated Gradients (+ optional SmoothIG)
# -----------------------------------------------------------------------------
@dataclass
class IGOutputs:
    attributions: np.ndarray  # (3,H,W) in [-?, ?]
    pred_class: int
    pred_score: float

class IntegratedGradients:
    def __init__(self, model: torch.nn.Module, device=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device

    def _prepare_input(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            x = img.unsqueeze(0)
        else:
            x = img
        return x.to(self.device)

    def attribute(self,
                  img: torch.Tensor,
                  target_class: Optional[int] = None,
                  baseline: Optional[torch.Tensor] = None,
                  steps: int = 50) -> IGOutputs:
        """
        img: (3,H,W) normalized.
        baseline: (3,H,W) normalized; default = zero image (black, normalized)
        steps: number of interpolation steps
        """
        x = self._prepare_input(img)  # (1,3,H,W)
        if baseline is None:
            baseline = torch.zeros_like(x)

        # Prediction
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1)[0])
            score = float(probs[0, pred_class])
        tc = pred_class if target_class is None else int(target_class)

        # Integrated gradients
        scaled_inputs = [baseline + (float(i)/steps) * (x - baseline) for i in range(1, steps+1)]
        grads_sum = torch.zeros_like(x)
        for s_in in scaled_inputs:
            s_in.requires_grad_(True)
            logits_s = self.model(s_in)
            loss = logits_s[:, tc].sum()
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            grads = s_in.grad.detach()
            grads_sum += grads
        avg_grads = grads_sum / steps
        attributions = (x - baseline) * avg_grads   # (1,3,H,W)

        return IGOutputs(
            attributions=attributions.squeeze(0).cpu().numpy(),
            pred_class=pred_class,
            pred_score=score
        )

    def smooth_ig(self,
                  img: torch.Tensor,
                  target_class: Optional[int] = None,
                  steps: int = 50,
                  noise_sigma: float = 0.2,
                  samples: int = 10) -> IGOutputs:
        """
        Smooth Integrated Gradients: average IG over noisy baselines around zero.
        noise_sigma is in normalized space.
        """
        x = self._prepare_input(img)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1)[0])
            score = float(probs[0, pred_class])
        tc = pred_class if target_class is None else int(target_class)

        all_attr = []
        for _ in range(samples):
            baseline = torch.randn_like(x) * noise_sigma
            ig_out = self.attribute(img, target_class=tc, baseline=baseline.squeeze(0), steps=steps)
            all_attr.append(torch.from_numpy(ig_out.attributions))
        mean_attr = torch.stack(all_attr, dim=0).mean(dim=0).numpy()  # (3,H,W)
        return IGOutputs(attributions=mean_attr, pred_class=pred_class, pred_score=score)

# -----------------------------------------------------------------------------
# Turn IG attributions into positive heatmap for overlay
# -----------------------------------------------------------------------------
def ig_to_heatmap(attr: np.ndarray, abs_channels: bool = True) -> np.ndarray:
    """
    attr: (3,H,W)
    Returns (H,W) in [0,1], using sum of absolute channel attributions by default.
    """
    if abs_channels:
        heat = np.abs(attr).sum(axis=0)  # (H,W)
    else:
        heat = np.maximum(attr, 0).sum(axis=0)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    return heat

# -----------------------------------------------------------------------------
# Sample selection helpers from test_dataset
# -----------------------------------------------------------------------------
def gather_indices_by_class(dataset: torchvision.datasets.ImageFolder) -> Dict[int, List[int]]:
    idxs = {c: [] for c in range(len(dataset.classes))}
    for i, (_, y) in enumerate(dataset.samples):
        idxs[y].append(i)
    return idxs

def pick_samples_for_demo(dataset, class_to_pick: Dict[str, int], k_per_class: int = 2,
                          seed: int = 42) -> List[int]:
    """
    class_to_pick: dict like {"COVID":2, "Normal":2, "Viral Pneumonia":2}
    Returns list of dataset indices (not DataLoader order).
    """
    rng = np.random.default_rng(seed)
    by_class = gather_indices_by_class(dataset)
    name_to_id = {name: i for i, name in enumerate(dataset.classes)}
    chosen = []
    for name, k in class_to_pick.items():
        cid = name_to_id[name]
        pool = by_class[cid]
        k_eff = min(k, len(pool))
        chosen.extend(list(rng.choice(pool, size=k_eff, replace=False)))
    return chosen

# -----------------------------------------------------------------------------
# High-level runner: compute and visualize XAI for a single image tensor
# -----------------------------------------------------------------------------
def run_xai_for_image(model,
                      img_tensor_norm: torch.Tensor,   # (3,H,W), already normalized
                      class_names: List[str],
                      target_layer: Optional[torch.nn.Module] = None,
                      method: str = 'gradcam',         # 'gradcam', 'gradcam++', 'ig', 'smoothig'
                      ig_steps: int = 50,
                      ig_noise_sigma: float = 0.2,
                      ig_samples: int = 10,
                      alpha: float = 0.35,
                      show: bool = True) -> Dict[str, any]:
    """
    Returns dict with: pred_class, pred_score, heatmap (H,W), overlay (H,W,3), raw_cam/attr
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor_norm.unsqueeze(0).to(next(model.parameters()).device))
        probs = F.softmax(logits, dim=1)
        pred_id = int(probs.argmax(dim=1)[0])
        pred_score = float(probs[0, pred_id])
    pred_name = class_names[pred_id]

    # Prepare display image
    img_denorm = denormalize(img_tensor_norm)
    rgb_uint8 = to_numpy_uint8(img_denorm)

    if method.lower() in ['gradcam', 'grad-cam', 'gc']:
        cam = GradCAM(model, target_layer=target_layer)
        out = cam(img_tensor_norm)
        heat = out.cam  # (H,W) in [0,1]
        overlay = make_cam_overlay(rgb_uint8, heat)
        if show:
            vis_triptych(rgb_uint8, heat, overlay, titles=('Input', 'Grad-CAM', 'Overlay'))
        cam.remove_hooks()
        return dict(method='gradcam', pred_class=pred_name, pred_score=pred_score,
                    heatmap=heat, overlay=overlay, raw=out)

    elif method.lower() in ['gradcam++', 'grad-cam++', 'gcpp']:
        campp = GradCAMPlusPlus(model, target_layer=target_layer)
        out = campp(img_tensor_norm)
        heat = out.cam
        overlay = make_cam_overlay(rgb_uint8, heat)
        if show:
            vis_triptych(rgb_uint8, heat, overlay, titles=('Input', 'Grad-CAM++', 'Overlay'))
        campp.remove_hooks()
        return dict(method='gradcam++', pred_class=pred_name, pred_score=pred_score,
                    heatmap=heat, overlay=overlay, raw=out)

    elif method.lower() in ['ig', 'integratedgradients', 'integrated-gradients']:
        ig = IntegratedGradients(model)
        out = ig.attribute(img_tensor_norm)
        heat = ig_to_heatmap(out.attributions)
        overlay = make_cam_overlay(rgb_uint8, heat)
        if show:
            vis_triptych(rgb_uint8, heat, overlay, titles=('Input', 'Integrated Gradients', 'Overlay'))
        return dict(method='ig', pred_class=pred_name, pred_score=pred_score,
                    heatmap=heat, overlay=overlay, raw=out)

    elif method.lower() in ['smoothig', 'smooth-ig', 'sig']:
        ig = IntegratedGradients(model)
        out = ig.smooth_ig(img_tensor_norm, steps=ig_steps, noise_sigma=ig_noise_sigma, samples=ig_samples)
        heat = ig_to_heatmap(out.attributions)
        overlay = make_cam_overlay(rgb_uint8, heat)
        if show:
            vis_triptych(rgb_uint8, heat, overlay, titles=('Input', 'Smooth IG', 'Overlay'))
        return dict(method='smoothig', pred_class=pred_name, pred_score=pred_score,
                    heatmap=heat, overlay=overlay, raw=out)

    else:
        raise ValueError(f"Unknown method: {method}")

# -----------------------------------------------------------------------------
# BUILD A SMALL DEMO SET GUIDED BY YOUR CONFUSION MATRIX
# -----------------------------------------------------------------------------
# From your DenseNet121 CM example (rows=true, cols=pred):
# COVID: TP=140, FN_to_Normal=5, FN_to_VP=0
# Normal: FN_to_COVID=5, TP=136, FN_to_VP=4
# Viral Pn.: FN_to_others=0, TP=145

# We'll pick:
# - 2 COVID (likely TPs)
# - 2 Normal (include potential confusables)
# - 2 Viral Pneumonia
# Modify counts below if you want more/less.
demo_class_counts = {"COVID": 2, "Normal": 2, "Viral Pneumonia": 2}

# If you don’t already have test_dataset in scope, uncomment and adjust:
# eval_transform = transforms.Compose([
#     transforms.Lambda(lambda x: x.convert('RGB')),
#     transforms.Resize((224,224)),  # or 300 for EfficientNet-B3
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])
# from torchvision import datasets
# TEST_DIR = "/path/to/test"
# test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

# Choose indices
# chosen_indices = pick_samples_for_demo(test_dataset, demo_class_counts, k_per_class=2)
# For reproducibility across runs, do once:
# print("Chosen indices:", chosen_indices)

# -----------------------------------------------------------------------------
# DEMO: Run all four explainers on the chosen images
# -----------------------------------------------------------------------------
def tensor_from_dataset(dataset, idx: int) -> Tuple[torch.Tensor, int, str, str]:
    """
    Returns (img_tensor_norm, label_id, label_name, path)
    """
    path, label = dataset.samples[idx]
    pil = dataset.loader(path)
    img = dataset.transform(pil)
    label_name = dataset.classes[label]
    return img, label, label_name, path

def get_target_layer_for_densenet(model):
    """
    For DenseNet121 from timm/torchvision, pick the deepest Conv2d.
    This generic finder works well, but you can hardcode a known layer if desired.
    """
    return find_last_conv_layer(model)

def run_demo(model,
             dataset,
             indices: List[int],
             class_names: List[str],
             save_dir: Optional[str] = None,
             methods: List[str] = ['gradcam', 'gradcam++', 'ig', 'smoothig']):
    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    tl = get_target_layer_for_densenet(model)

    for idx in indices:
        img_t, y, y_name, path = tensor_from_dataset(dataset, idx)
        print(f"\nSample idx={idx} | true={y_name} | path={os.path.basename(path)}")
        for m in methods:
            out = run_xai_for_image(
                model=model,
                img_tensor_norm=img_t,
                class_names=class_names,
                target_layer=tl,
                method=m,
                ig_steps=50,
                ig_noise_sigma=0.2,
                ig_samples=8,
                alpha=0.35,
                show=True
            )
            if save_dir:
                # Save overlay
                title = f"{m}_pred-{out['pred_class']}_{os.path.splitext(os.path.basename(path))[0]}.png"
                save_path = os.path.join(save_dir, title)
                plt.imsave(save_path, out['overlay'])
                print(f"Saved: {save_path}")

model = build_model("DenseNet121")

checkpoint = torch.load(
    "best_densenet121.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

_, _, test_loader = create_dataloaders(
    input_size=224,
    batch_size=32
)

report_dict, cm, densenet_preds, densenet_labels = evaluate_on_test(
    model,
    test_loader,
    "DenseNet121"
)

normal_to_covid = np.where(
    (densenet_labels == 1) &
    (densenet_preds == 0)
)[0]

covid_to_normal = np.where(
    (densenet_labels == 0) &
    (densenet_preds == 1)
)[0]

print(normal_to_covid)
print(covid_to_normal)

misclassified_indices = (
    list(normal_to_covid) +
    list(covid_to_normal)
)

print("Misclassified:", misclassified_indices)
print("Count:", len(misclassified_indices))

import numpy as np

all_indices = np.arange(len(test_dataset))

remaining = np.setdiff1d(
    all_indices,
    misclassified_indices
)

rng = np.random.default_rng(42)

random_indices = rng.choice(
    remaining,
    size=15,
    replace=False
)

random_indices = list(random_indices)

print(random_indices)

hosen_indices = misclassified_indices + random_indices

print("Total selected:", len(chosen_indices))
print(chosen_indices)

run_demo(
    model,
    test_dataset,
    chosen_indices,
    CLASS_NAMES,
    save_dir="xai_outputs",
    methods=['gradcam', 'gradcam++', 'ig', 'smoothig']
)
