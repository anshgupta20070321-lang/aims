#!/usr/bin/env python
# =============================================================================
# AIMS DTU Research Intern 2026 — Explainable Computer Vision
# Part 2: Attention Rollout for DeiT-Base
#
# BUGS FIXED vs previous version:
#   FIX 1 (CRITICAL) — load_deit_from_ckpt: added fused_attn=False loop.
#                       In timm>=0.9+PyTorch>=2.0, fused_attn=True is the
#                       default. It routes forward through
#                       F.scaled_dot_product_attention, completely bypassing
#                       the explicit softmax and attn_drop calls. Hooks on
#                       blk.attn.softmax never fire → _attn_maps stays empty.
#   FIX 2 (CRITICAL) — _register_hooks: changed hook target from
#                       blk.attn.softmax (does NOT exist as nn.Module in
#                       timm 1.0.x → immediate AttributeError) to
#                       blk.attn.attn_drop, capturing inp[0] which is the
#                       post-softmax attention BEFORE the dropout no-op.
#   FIX 3 (CRITICAL) — layerwise_rollout: same hook fix as FIX 2.
#   FIX 4            — weights_only=False added to torch.load to suppress
#                       FutureWarning (checkpoint stores non-tensor dicts).
#   FIX 5            — plt.close(fig) added after every plt.show() call in
#                       all four visualisation functions to prevent figure
#                       memory accumulation inside a Kaggle notebook.
#   FIX 6            — summary box filenames corrected:
#                       covidtonormal → covid_to_normal
#                       normaltoviral_pneumonia → normal_to_viral_pneumonia
#   CLEANUP          — removed unused imports: copy, Subset, gridspec, mpatches
#                       replaced X|Y union hint with Optional[X] (Py 3.9 safe)
# =============================================================================

# %%
# ===========================================================================
# SECTION 0: IMPORTS & SETUP
# ===========================================================================
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Dataset / model constants (must match Part 1 exactly) ────────────────────
ROOT        = "/kaggle/input/datasets/ansh21032007/covid-19-dataset/COVID_19_dataset"
TEST_DIR    = os.path.join(ROOT, "test")
CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]
NUM_CLASSES = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# %%
# ===========================================================================
# SECTION 1: TRANSFORMS & MODEL LOADER
# ===========================================================================
def get_eval_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Identical to Part 1's eval_transform — must NOT diverge."""
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_deit_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Load DeiT-Base checkpoint and prepare it for attention rollout.

    Critical preparation step
    ─────────────────────────
    After loading weights we iterate every sub-module and set
    `fused_attn = False` wherever the attribute exists.

    Why this is necessary
    ─────────────────────
    timm >= 0.9 with PyTorch >= 2.0 sets fused_attn=True by default on the
    Attention modules inside each transformer block.  When fused_attn=True the
    forward path is:

        x = F.scaled_dot_product_attention(q, k, v, ...)   # one fused CUDA op

    The explicit `attn = attn.softmax(dim=-1)` and `attn = self.attn_drop(attn)`
    lines are NEVER reached, so any hook registered on those modules captures
    nothing.  _attn_maps stays empty → assertion fails with
    "Expected 12 attention maps, got 0".

    With fused_attn=False the path is:
        attn = (q @ k.T) * scale
        attn = attn.softmax(dim=-1)     # tensor method, no module to hook
        attn = self.attn_drop(attn)     # ← we hook inp[0] here  ✓
        x    = attn @ v

    This change does NOT alter any learned weights.  The output is numerically
    identical (no dropout in eval mode).
    """
    model = timm.create_model(
        "deit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.1,
    ).to(device)

    # FIX 4: weights_only=False — checkpoint stores a config dict alongside
    # tensors, so plain weights_only=True would raise an UnpicklingError.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # FIX 1 (CRITICAL): disable fused attention on every Attention module
    n_fixed = 0
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
            n_fixed += 1

    print(
        f"  ✓ Loaded '{ckpt_path}'  "
        f"val_F1={ckpt.get('val_f1', float('nan')):.4f}  "
        f"epoch={ckpt.get('epoch', '?')}"
    )
    print(f"  ✓ fused_attn disabled on {n_fixed} Attention modules")
    return model


# %%
# ===========================================================================
# SECTION 2: ATTENTION ROLLOUT
# ===========================================================================
class AttentionRollout:
    """
    Attention Rollout for DeiT-Base (timm 1.0.x, PyTorch 2.x).

    Hook strategy (FIX 2)
    ─────────────────────
    After load_deit_from_ckpt sets fused_attn=False, each block's forward is:

        attn = (q @ k.T) * scale
        attn = attn.softmax(dim=-1)   ← tensor method — cannot hook as module
        attn = self.attn_drop(attn)   ← nn.Dropout — we hook its INPUT here
        x    = attn @ v

    We use register_forward_hook on blk.attn.attn_drop and read inp[0]:
        inp[0]  shape: (B, num_heads, N, N) — post-softmax probabilities ✓
        out     shape: same but zeros dropped (no-op in eval) — do NOT use out

    In eval mode attn_drop is a no-op so inp[0] == out, but capturing inp[0]
    is semantically correct (and would still be correct during training).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        add_residual: bool = True,
    ):
        assert head_fusion in ("mean", "max", "min")
        assert 0.0 <= discard_ratio < 1.0

        self.model         = model
        self.device        = device
        self.head_fusion   = head_fusion
        self.discard_ratio = discard_ratio
        self.add_residual  = add_residual

        self._attn_maps: list = []
        self._hooks: list     = []
        self._register_hooks()

    # ── Hook registration ────────────────────────────────────────────────────
    def _register_hooks(self):
        """
        Hook blk.attn.attn_drop (nn.Dropout) — capture inp[0].

        Default-argument binding (_store=self._attn_maps) is used instead of
        a plain closure to avoid the classic loop-closure bug where every
        lambda would share the last value of `blk`.
        """
        for blk in self.model.blocks:
            # FIX 2: attn_drop not softmax; inp[0] not out
            def _hook(module, inp, out, _store=self._attn_maps):
                _store.append(inp[0].detach().cpu())   # (B, H, N, N)

            self._hooks.append(
                blk.attn.attn_drop.register_forward_hook(_hook)
            )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Rollout computation ───────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) on self.device — ImageNet-normalised.
        Returns:
            cam: (B, 224, 224) in [0, 1] — per-pixel relevance map.
        """
        self.model.eval()
        self._attn_maps.clear()

        with autocast("cuda" if self.device.type == "cuda" else "cpu"):
            _ = self.model(x)

        B = x.size(0)
        L = len(self._attn_maps)
        assert L == len(self.model.blocks), (
            f"Expected {len(self.model.blocks)} maps, got {L}. "
            "Call load_deit_from_ckpt() — it disables fused_attn."
        )

        # ── Step 1: head fusion + optional row-wise discard ──────────────────
        fused = []
        for A in self._attn_maps:           # A: (B, H, N, N) on CPU
            A = A.to(self.device)

            if self.head_fusion == "mean":
                A = A.mean(dim=1)           # (B, N, N)
            elif self.head_fusion == "max":
                A = A.max(dim=1).values
            else:
                A = A.min(dim=1).values

            if self.discard_ratio > 0.0:
                N = A.size(-1)
                k = max(1, int(N * self.discard_ratio))
                flat = A.view(B * N, N)     # per-row discard (not global)
                flat.scatter_(1, flat.topk(k, dim=1, largest=False).indices, 0.0)
                A = flat.view(B, N, N)

            if self.add_residual:           # Ã = (I + A) / 2  (Chefer et al.)
                N = A.size(-1)
                A = (A + torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)) * 0.5

            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)  # row-normalise
            fused.append(A)

        # ── Step 2: cumulative product  R = A_L @ ... @ A_1 ──────────────────
        R = fused[0]
        for i in range(1, L):
            R = torch.bmm(fused[i], R)

        # ── Step 3: CLS row → patch map → upsample ───────────────────────────
        # R[:, 0, :] = relevance from CLS to all 197 tokens
        # drop index 0 (CLS→CLS) → 196 patch tokens
        cls_attn = R[:, 0, 1:]                              # (B, 196)
        side     = int(cls_attn.size(-1) ** 0.5)            # 14
        assert side * side == cls_attn.size(-1)

        patch_map = cls_attn.view(B, 1, side, side)
        mn = patch_map.amin(dim=(2, 3), keepdim=True)
        mx = patch_map.amax(dim=(2, 3), keepdim=True)
        patch_map = (patch_map - mn) / (mx - mn + 1e-6)    # normalise to [0,1]

        cam = F.interpolate(
            patch_map, size=(x.size(2), x.size(3)),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                        # (B, H, W)
        return cam


# %%
# ===========================================================================
# SECTION 3: STRATIFIED SAMPLE BUILDER
# ===========================================================================
SAMPLE_SPEC = {
    (0, 0): 5,   # Correct COVID
    (1, 1): 5,   # Correct Normal
    (2, 2): 5,   # Correct Viral Pneumonia
    (0, 1): 1,   # COVID → Normal  (misclassification from confusion matrix)
    (1, 2): 1,   # Normal → Viral Pneumonia
}


@torch.no_grad()
def build_stratified_sample(
    model:       nn.Module,
    test_dir:    str,
    transform:   transforms.Compose,
    sample_spec: dict,
    device:      torch.device,
    batch_size:  int = 32,
    seed:        int = 42,
):
    """
    Run full-test-set inference, then pick examples matching each
    (true_class, pred_class) combination in sample_spec.

    Returns (imgs, labels, preds) tensors on CPU.
    Ordering follows dict insertion order (Python 3.7+), which must match
    GROUP_SPEC in visualize_rollout_grid.
    """
    set_seed(seed)
    model.eval()

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    # Inference pass — keep everything on CPU
    all_imgs, all_labels, all_preds = [], [], []
    for imgs, labels in loader:
        with autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(imgs.to(device, non_blocking=True))
        all_imgs.append(imgs)
        all_labels.append(labels)
        all_preds.append(logits.argmax(1).cpu())

    all_imgs   = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels)
    all_preds  = torch.cat(all_preds)

    sel_imgs, sel_labels, sel_preds = [], [], []
    for (true_cls, pred_cls), count in sample_spec.items():
        mask    = (all_labels == true_cls) & (all_preds == pred_cls)
        indices = mask.nonzero(as_tuple=False).squeeze(1)

        if len(indices) < count:
            print(f"  ⚠ Only {len(indices)} found for "
                  f"{CLASS_NAMES[true_cls]}→{CLASS_NAMES[pred_cls]}, "
                  f"wanted {count}")
            count = len(indices)

        chosen = indices[torch.randperm(len(indices))[:count]]
        sel_imgs.append(all_imgs[chosen])
        sel_labels.extend([true_cls] * count)
        sel_preds.extend([pred_cls]  * count)

        tick = "✓" if true_cls == pred_cls else "✗"
        print(f"  {tick}  {CLASS_NAMES[true_cls]:>17} → "
              f"{CLASS_NAMES[pred_cls]:<17}  {count} samples")

    imgs_out   = torch.cat(sel_imgs)
    labels_out = torch.tensor(sel_labels, dtype=torch.long)
    preds_out  = torch.tensor(sel_preds,  dtype=torch.long)
    print(f"\n  Total selected: {len(imgs_out)} images")
    return imgs_out, labels_out, preds_out


# %%
# ===========================================================================
# SECTION 4: VISUALIZATION HELPERS
# ===========================================================================
def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → uint8 HWC RGB."""
    mean = np.array(IMAGENET_MEAN, np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  np.float32).reshape(3, 1, 1)
    img  = img_t.cpu().numpy().astype(np.float32) * std + mean
    return (np.clip(img, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def apply_heatmap(cam: torch.Tensor,
                  cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """(H,W) [0,1] tensor → uint8 RGB heatmap."""
    return cv2.cvtColor(
        cv2.applyColorMap((cam.cpu().numpy() * 255).astype(np.uint8), cmap),
        cv2.COLOR_BGR2RGB,
    )


def blend(img_rgb: np.ndarray, heat_rgb: np.ndarray,
          alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img_rgb, 1 - alpha, heat_rgb, alpha, 0)


# GROUP_SPEC order must match SAMPLE_SPEC insertion order
GROUP_SPEC = [
    ("Correct COVID",             0, 0, 5),
    ("Correct Normal",            1, 1, 5),
    ("Correct Viral Pneumonia",   2, 2, 5),
    ("COVID → Normal",            0, 1, 1),
    ("Normal → Viral Pneumonia",  1, 2, 1),
]


def visualize_rollout_grid(
    imgs:        torch.Tensor,
    labels:      torch.Tensor,
    preds:       torch.Tensor,
    cams:        torch.Tensor,
    save_prefix: str   = "deit_rollout",
    alpha:       float = 0.45,
):
    """One figure per group: rows=samples, cols=[original|heatmap|overlay]."""
    idx = 0
    for group_label, true_cls, pred_cls, count in GROUP_SPEC:
        if count == 0:
            continue

        fig, axes = plt.subplots(count, 3,
                                 figsize=(9, 3.2 * count), squeeze=False)
        color = "#2ECC71" if true_cls == pred_cls else "#E74C3C"
        fig.suptitle(group_label, fontsize=14, fontweight="bold",
                     color=color, y=1.01)

        for col, title in enumerate(["Original", "Attention Map", "Overlay"]):
            axes[0][col].set_title(title, fontsize=11, pad=6)

        for row in range(count):
            i       = idx + row
            img_np  = denormalize(imgs[i])
            heat_np = apply_heatmap(cams[i])
            over_np = blend(img_np, heat_np, alpha)

            for col, panel in enumerate([img_np, heat_np, over_np]):
                ax = axes[row][col]
                ax.imshow(panel); ax.axis("off")
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(1.5)
                    sp.set_visible(True)

            axes[row][0].set_ylabel(
                f"GT: {CLASS_NAMES[labels[i]]}\nPred: {CLASS_NAMES[preds[i]]}",
                fontsize=9, rotation=0, labelpad=80, va="center",
            )

        plt.tight_layout()
        fname = f"{save_prefix}_{group_label.replace(' ','_').replace('→','to').lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)    # FIX 5: free figure memory
        print(f"  ✓ Saved → {fname}")
        idx += count


def visualize_mean_rollout(
    imgs:      torch.Tensor,
    labels:    torch.Tensor,
    cams:      torch.Tensor,
    save_path: str = "deit_rollout_mean_per_class.png",
):
    """Per-class mean rollout: row0=mean image, row1=mean rollout overlay."""
    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 8))
    fig.suptitle("DeiT-Base — Mean Attention Rollout per Class",
                 fontsize=14, fontweight="bold")

    for c, cls_name in enumerate(CLASS_NAMES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_img = imgs[mask].mean(0)
        mean_cam = cams[mask].mean(0)
        mn, mx   = mean_cam.min(), mean_cam.max()
        mean_cam = (mean_cam - mn) / (mx - mn + 1e-6)

        img_np = denormalize(mean_img)
        axes[0][c].imshow(img_np)
        axes[0][c].set_title(cls_name, fontsize=13, fontweight="bold")
        axes[0][c].axis("off")
        axes[1][c].imshow(blend(img_np, apply_heatmap(mean_cam), 0.5))
        axes[1][c].axis("off")

    axes[0][0].set_ylabel("Mean Image",          fontsize=11, labelpad=6)
    axes[1][0].set_ylabel("Mean Rollout Overlay", fontsize=11, labelpad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)    # FIX 5
    print(f"  ✓ Saved → {save_path}")


# %%
# ===========================================================================
# SECTION 5: ABLATION — head_fusion × discard_ratio
# ===========================================================================
def ablation_rollout(
    model:          nn.Module,
    img:            torch.Tensor,
    true_label:     int,
    pred_label:     int,
    device:         torch.device,
    head_fusions:   Optional[list] = None,
    discard_ratios: Optional[list] = None,
    save_path:      str = "deit_rollout_ablation.png",
):
    """3×3 grid: rows=head_fusion, cols=discard_ratio. Best on misclassified image."""
    if head_fusions   is None: head_fusions   = ["mean", "max", "min"]
    if discard_ratios is None: discard_ratios = [0.0, 0.5, 0.9]

    img_np    = denormalize(img)
    img_batch = img.unsqueeze(0).to(device)

    fig, axes = plt.subplots(len(head_fusions), len(discard_ratios),
                             figsize=(4.5 * len(discard_ratios),
                                      4.5 * len(head_fusions)))
    fig.suptitle(
        f"Rollout Ablation  |  GT: {CLASS_NAMES[true_label]}  "
        f"Pred: {CLASS_NAMES[pred_label]}",
        fontsize=13, fontweight="bold",
    )

    for r, hf in enumerate(head_fusions):
        for c, dr in enumerate(discard_ratios):
            ro  = AttentionRollout(model, device,
                                   head_fusion=hf, discard_ratio=dr)
            cam = ro(img_batch)[0]
            ro.remove_hooks()

            axes[r][c].imshow(blend(img_np, apply_heatmap(cam)))
            axes[r][c].axis("off")
            axes[r][c].set_title(f"fusion={hf}\ndiscard={dr:.0%}", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)    # FIX 5
    print(f"  ✓ Saved → {save_path}")


# %%
# ===========================================================================
# SECTION 6: LAYER-WISE ROLLOUT — depth 1 → 12
# ===========================================================================
def layerwise_rollout(
    model:         nn.Module,
    img:           torch.Tensor,
    device:        torch.device,
    head_fusion:   str   = "mean",
    discard_ratio: float = 0.0,
    save_path:     str   = "deit_rollout_layerwise.png",
):
    """Shows cumulative rollout at each transformer depth (1→12)."""
    img_batch  = img.unsqueeze(0).to(device)
    img_np     = denormalize(img)
    B          = 1
    num_blocks = len(model.blocks)

    # Single forward pass to collect all 12 attention maps
    attn_maps: list = []
    hooks = []
    for blk in model.blocks:
        # FIX 3: attn_drop + inp[0]  (same fix as AttentionRollout._register_hooks)
        def _hook(module, inp, out, _store=attn_maps):
            _store.append(inp[0].detach().cpu())   # (1, H, N, N)
        hooks.append(blk.attn.attn_drop.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad(), autocast("cuda" if device.type == "cuda" else "cpu"):
        _ = model(img_batch)

    for h in hooks:
        h.remove()

    cols      = 4
    rows      = -(-num_blocks // cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4.5 * cols, 4 * rows))
    axes_flat = axes.ravel()
    fig.suptitle("DeiT-Base — Layer-wise Cumulative Rollout (depth 1→12)",
                 fontsize=13, fontweight="bold")

    R: Optional[torch.Tensor] = None

    for depth in range(1, num_blocks + 1):
        A = attn_maps[depth - 1].to(device)   # (1, H, N, N)

        if head_fusion == "mean":
            A = A.mean(dim=1)
        elif head_fusion == "max":
            A = A.max(dim=1).values
        else:
            A = A.min(dim=1).values

        if discard_ratio > 0.0:
            N    = A.size(-1)
            k    = max(1, int(N * discard_ratio))
            flat = A.view(B * N, N)
            flat.scatter_(1, flat.topk(k, dim=1, largest=False).indices, 0.0)
            A = flat.view(B, N, N)

        N = A.size(-1)
        A = (A + torch.eye(N, device=device).unsqueeze(0)) * 0.5
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        R = A if R is None else torch.bmm(A, R)

        cls_attn  = R[0, 0, 1:]
        side      = int(cls_attn.size(-1) ** 0.5)
        patch_map = cls_attn.view(1, 1, side, side)
        mn, mx    = patch_map.min(), patch_map.max()
        patch_map = (patch_map - mn) / (mx - mn + 1e-6)
        cam = F.interpolate(patch_map, size=(INPUT_SIZE, INPUT_SIZE),
                            mode="bilinear",
                            align_corners=False).squeeze().cpu()

        ax = axes_flat[depth - 1]
        ax.imshow(blend(img_np, apply_heatmap(cam)))
        ax.set_title(f"Layer {depth}", fontsize=10)
        ax.axis("off")

    for ax in axes_flat[num_blocks:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)    # FIX 5
    print(f"  ✓ Saved → {save_path}")


# %%
# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ATTENTION ROLLOUT — DeiT-Base | AIMS DTU 2026")
    print("=" * 70)

    # 1. Load model — fused_attn disabled inside load_deit_from_ckpt
    print("\n[1/5] Loading checkpoint...")
    model  = load_deit_from_ckpt("best_deit_base.pth", DEVICE)
    eval_tf = get_eval_transform()

    # 2. Build stratified 17-image sample
    print("\n[2/5] Building stratified sample...")
    imgs, labels, preds = build_stratified_sample(
        model=model, test_dir=TEST_DIR, transform=eval_tf,
        sample_spec=SAMPLE_SPEC, device=DEVICE,
    )

    # 3. Compute rollout maps in mini-batches
    print("\n[3/5] Computing rollout maps...")
    rollout = AttentionRollout(model, DEVICE,
                               head_fusion="mean",
                               discard_ratio=0.0,
                               add_residual=True)
    all_cams = []
    for start in range(0, len(imgs), 8):
        all_cams.append(rollout(imgs[start:start + 8].to(DEVICE)).cpu())
    cams = torch.cat(all_cams)          # (17, 224, 224)
    rollout.remove_hooks()
    print(f"  ✓ {cams.size(0)} maps computed  shape={tuple(cams.shape)}")

    # 4. Visualise
    print("\n[4/5] Visualising...")
    visualize_rollout_grid(imgs, labels, preds, cams)

    correct = labels == preds
    visualize_mean_rollout(imgs[correct], labels[correct], cams[correct])

    # 5. Ablation + layer-wise on COVID→Normal misclassification
    print("\n[5/5] Ablation & layer-wise...")
    mis = (labels == 0) & (preds == 1)
    if mis.any():
        i = mis.nonzero(as_tuple=False)[0].item()
        ablation_rollout(model, imgs[i],
                         int(labels[i]), int(preds[i]), DEVICE,
                         save_path="deit_rollout_ablation_covid_to_normal.png")
        layerwise_rollout(model, imgs[i], DEVICE,
                          save_path="deit_rollout_layerwise_covid_to_normal.png")
    else:
        print("  ⚠ No COVID→Normal sample found — skipping ablation.")

    # FIX 6: correct filenames in summary
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  DONE — outputs saved:                                              ║
║   deit_rollout_correct_covid.png                                    ║
║   deit_rollout_correct_normal.png                                   ║
║   deit_rollout_correct_viral_pneumonia.png                          ║
║   deit_rollout_covid_to_normal.png                                  ║
║   deit_rollout_normal_to_viral_pneumonia.png                        ║
║   deit_rollout_mean_per_class.png                                   ║
║   deit_rollout_ablation_covid_to_normal.png                         ║
║   deit_rollout_layerwise_covid_to_normal.png                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
