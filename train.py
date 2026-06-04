#!/usr/bin/env python
# =============================================================================
# AIMS DTU Research Intern 2026 — Explainable Computer Vision
# Part 1: Model Training & Evaluation
# Models: DenseNet121, ResNet50, EfficientNet-B3, DeiT-Base
# Dataset: COVID-19 Chest X-Ray (3-class: COVID, Normal, Viral Pneumonia)
# =============================================================================
# %%
# ===========================================================================
# SECTION 0: REPRODUCIBILITY
# ===========================================================================
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
# %%
# ===========================================================================
# SECTION 1: DATASET PATHS & CONFIGURATION
# ===========================================================================
ROOT = "/kaggle/input/datasets/ansh21032007/covid-19-dataset/COVID_19_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "val")
TEST_DIR  = os.path.join(ROOT, "test")
CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = 3
# Verify dataset structure
for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
    total = 0
    print(f"\n{split_name} split:")
    for cls in CLASS_NAMES:
        cls_path = os.path.join(split_dir, cls)
        count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
        print(f"  {cls}: {count} images")
        total += count
    print(f"  Total: {total}")
# %%
# ===========================================================================
# SECTION 2 & 3: DATA PREPROCESSING AND AUGMENTATION
# ===========================================================================
# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
def get_transforms(input_size=224):
    """
    Create train and val/test transforms for a given input size.
    CRITICAL: Grayscale → RGB conversion is the FIRST transform.
    """
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform
def create_dataloaders(input_size=224, batch_size=32):
    """Create train, val, test DataLoaders with appropriate transforms."""
    train_tf, eval_tf = get_transforms(input_size)
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Class mapping: {train_dataset.class_to_idx}")
    return train_loader, val_loader, test_loader
# %%
# ===========================================================================
# SECTION 4: MODEL DEFINITIONS
# ===========================================================================
MODEL_CONFIGS = {
    'DenseNet121': {
        'timm_name': 'densenet121',
        'input_size': 224,
        'batch_size': 32,
        'is_vit': False,
    },
    'ResNet50': {
        'timm_name': 'resnet50',
        'input_size': 224,
        'batch_size': 32,
        'is_vit': False,
    },
    'EfficientNet-B3': {
        'timm_name': 'efficientnet_b3',
        'input_size': 300,
        'batch_size': 32,
        'is_vit': False,
    },
    'DeiT-Base': {
        'timm_name': 'deit_base_patch16_224',
        'input_size': 224,
        'batch_size': 16,
        'is_vit': True,
    },
}
def build_model(model_name):
    """Build a timm model with correct configuration."""
    cfg = MODEL_CONFIGS[model_name]
    if cfg['is_vit']:
        model = timm.create_model(
            cfg['timm_name'],
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=0.1,
        )
    else:
        model = timm.create_model(
            cfg['timm_name'],
            pretrained=True,
            num_classes=NUM_CLASSES,
        )
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {model_name}: {total_params/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    return model
# %%
# ===========================================================================
# SECTION 5–8: TRAINING ENGINE
# ===========================================================================
class MetricTracker:
    """Manages torchmetrics objects on GPU for efficient per-epoch tracking."""
    def __init__(self, num_classes, device):
        self.accuracy  = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
        self.recall    = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
        self.f1        = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
    def update(self, preds, targets):
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)
    def compute(self):
        return {
            'accuracy':  self.accuracy.compute().item(),
            'precision': self.precision.compute().item(),
            'recall':    self.recall.compute().item(),
            'f1':        self.f1.compute().item(),
        }
def train_one_epoch(model, loader, criterion, optimizer, scaler, is_vit=False):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        if is_vit:
            # Gradient clipping for ViT stability
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        num_batches += 1
    return running_loss / num_batches
@torch.no_grad()
def validate(model, loader, criterion, metrics):
    """Validate and compute all metrics."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    metrics.reset()
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        metrics.update(preds, labels)
        running_loss += loss.item()
        num_batches += 1
    val_loss = running_loss / num_batches
    metric_vals = metrics.compute()
    metric_vals['loss'] = val_loss
    return metric_vals
class EarlyStopping:
    """Early stopping based on validation F1 score (higher is better)."""
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
    def __call__(self, score, epoch):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True  # new best → save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # not improved
# %%
# ===========================================================================
# CNN TRAINING (DenseNet121, ResNet50, EfficientNet-B3)
# ===========================================================================
def get_backbone_and_head_params(model):
    """
    Split model parameters into backbone and head using timm's
    get_classifier() for universal compatibility.
    """
    classifier = model.get_classifier()
    classifier_param_ids = set(id(p) for p in classifier.parameters())
    backbone_params = []
    head_params = []
    for p in model.parameters():
        if id(p) in classifier_param_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return backbone_params, head_params
def train_cnn(model_name):
    """
    Train a CNN model with 2-phase strategy:
      Phase 1: Head-only (3-5 epochs, lr=1e-3)
      Phase 2: Full fine-tuning with differential LR (up to 25 epochs)
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*70}")
    cfg = MODEL_CONFIGS[model_name]
    set_seed(42)
    # Build model
    model = build_model(model_name)
    # Create data loaders
    print(f"\n  Creating DataLoaders (input_size={cfg['input_size']}, "
          f"batch_size={cfg['batch_size']})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )
    # Loss function (balanced dataset → no weights)
    criterion = nn.CrossEntropyLoss()
    # Metrics
    metrics = MetricTracker(NUM_CLASSES, DEVICE)
    # Mixed precision
    scaler = GradScaler('cuda')
    # History
    history = defaultdict(list)
    # Early stopping (across both phases)
    early_stopping = EarlyStopping(patience=7)
    # Best model state
    best_model_state = None
    best_f1 = 0.0
    total_epoch = 0  # global epoch counter
    # =====================================================================
    # PHASE 1: HEAD-ONLY TRAINING (5 epochs)
    # =====================================================================
    print(f"\n  --- Phase 1: Head-only training (5 epochs) ---")
    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier head
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (head only): {trainable:,}")
    # Optimizer for Phase 1: only head params
    optimizer_p1 = AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    phase1_epochs = 5
    for epoch in range(phase1_epochs):
        total_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p1, scaler)
        val_metrics = validate(model, val_loader, criterion, metrics)
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(optimizer_p1.param_groups[0]['lr'])
        improved = early_stopping(val_metrics['f1'], total_epoch)
        if improved:
            best_f1 = val_metrics['f1']
            best_model_state = copy.deepcopy(model.state_dict())
        print(f"  Epoch {total_epoch:02d} [P1] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"{'★ BEST' if improved else ''} | "
              f"{elapsed:.1f}s")
    # =====================================================================
    # PHASE 2: FULL FINE-TUNING (up to 25 epochs)
    # =====================================================================
    print(f"\n  --- Phase 2: Full fine-tuning (up to 25 epochs) ---")
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (full): {trainable:,}")
    backbone_params, head_params = get_backbone_and_head_params(model)
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
    print(f"  Head params:     {sum(p.numel() for p in head_params):,}")
    # Differential learning rates
    optimizer_p2 = AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer_p2, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    phase2_max_epochs = 25
    for epoch in range(phase2_max_epochs):
        total_epoch += 1
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p2, scaler)
        val_metrics = validate(model, val_loader, criterion, metrics)
        # Step scheduler on val F1
        scheduler.step(val_metrics['f1'])
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(optimizer_p2.param_groups[0]['lr'])
        improved = early_stopping(val_metrics['f1'], total_epoch)
        if improved:
            best_f1 = val_metrics['f1']
            best_model_state = copy.deepcopy(model.state_dict())
        current_lr_bb = optimizer_p2.param_groups[0]['lr']
        current_lr_hd = optimizer_p2.param_groups[1]['lr']
        print(f"  Epoch {total_epoch:02d} [P2] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {current_lr_bb:.1e}/{current_lr_hd:.1e} | "
              f"{'★ BEST' if improved else ''} | "
              f"ES: {early_stopping.counter}/{early_stopping.patience} | "
              f"{elapsed:.1f}s")
        if early_stopping.should_stop:
            print(f"\n  ✓ Early stopping triggered at epoch {total_epoch}. "
                  f"Best F1: {best_f1:.4f} at epoch {early_stopping.best_epoch}")
            break
    # Restore best model
    model.load_state_dict(best_model_state)
    # Save checkpoint
    save_path = f'best_{model_name.replace("-", "_").lower()}.pth'
    torch.save({
        'epoch': early_stopping.best_epoch,
        'model_state_dict': best_model_state,
        'val_f1': best_f1,
        'model_name': model_name,
        'config': cfg,
    }, save_path)
    print(f"  ✓ Best model saved to {save_path}")
    return model, history, test_loader, save_path
# %%
# ===========================================================================
# DeiT-Base TRAINING
# ===========================================================================
def train_deit():
    """
    Train DeiT-Base with single-phase full fine-tuning.
    Uses warmup (LinearLR) + CosineAnnealing via SequentialLR.
    Gradient clipping, label smoothing, weight_decay=0.05.
    """
    model_name = 'DeiT-Base'
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name} (Vision Transformer)")
    print(f"{'='*70}")
    set_seed(42)
    # Build model
    model = build_model(model_name)
    # Create data loaders
    print(f"\n  Creating DataLoaders (input_size={cfg['input_size']}, "
          f"batch_size={cfg['batch_size']})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )
    # Loss with label smoothing for ViT
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Metrics
    metrics = MetricTracker(NUM_CLASSES, DEVICE)
    # Mixed precision
    scaler = GradScaler('cuda')
    # Optimizer — full model, ViT-standard weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.05,
    )
    # Scheduler: Linear warmup (5 epochs) → Cosine annealing (30 epochs)
    warmup_epochs = 5
    max_epochs = 35
    cosine_epochs = max_epochs - warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6 / 5e-5,   # start at 1e-6
        end_factor=1.0,              # end at 5e-5
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    # History
    history = defaultdict(list)
    # Early stopping
    early_stopping = EarlyStopping(patience=7)
    best_model_state = None
    best_f1 = 0.0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        # Train (with gradient clipping via is_vit=True)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, is_vit=True
        )
        # Validate
        val_metrics = validate(model, val_loader, criterion, metrics)
        # Step scheduler (once per epoch, AFTER validation)
        scheduler.step()
        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        improved = early_stopping(val_metrics['f1'], epoch)
        if improved:
            best_f1 = val_metrics['f1']
            best_model_state = copy.deepcopy(model.state_dict())
        phase_label = "WARM" if epoch <= warmup_epochs else "COS "
        print(f"  Epoch {epoch:02d} [{phase_label}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"{'★ BEST' if improved else ''} | "
              f"ES: {early_stopping.counter}/{early_stopping.patience} | "
              f"{elapsed:.1f}s")
        if early_stopping.should_stop:
            print(f"\n  ✓ Early stopping triggered at epoch {epoch}. "
                  f"Best F1: {best_f1:.4f} at epoch {early_stopping.best_epoch}")
            break
    # Restore best model
    model.load_state_dict(best_model_state)
    # Save checkpoint
    save_path = 'best_deit_base.pth'
    torch.save({
        'epoch': early_stopping.best_epoch,
        'model_state_dict': best_model_state,
        'val_f1': best_f1,
        'model_name': model_name,
        'config': cfg,
    }, save_path)
    print(f"  ✓ Best model saved to {save_path}")
    return model, history, test_loader, save_path
# %%
# ===========================================================================
# SECTION 12: EVALUATION & VISUALIZATION
# ===========================================================================
@torch.no_grad()
def evaluate_on_test(model, test_loader, model_name):
    """
    Full evaluation on test set.
    Returns classification report dict and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast('cuda'):
            outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Classification report
    print(f"\n  {'='*50}")
    print(f"  TEST SET RESULTS — {model_name}")
    print(f"  {'='*50}")
    report_str = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report_str)
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return report_dict, cm, all_preds, all_labels
def plot_training_curves(history, model_name):
    """Plot training loss, validation loss, and F1 curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} — Training Curves', fontsize=14, fontweight='bold')
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train vs Val Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # F1 curve
    axes[1].plot(epochs, history['val_f1'], 'g-o', markersize=3, label='Val Macro F1')
    best_epoch = np.argmax(history['val_f1']) + 1
    best_f1 = max(history['val_f1'])
    axes[1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    axes[1].annotate(f'Best: {best_f1:.4f}\n(epoch {best_epoch})',
                     xy=(best_epoch, best_f1), fontsize=9,
                     xytext=(best_epoch + 1, best_f1 - 0.02),
                     arrowprops=dict(arrowstyle='->', color='gray'))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation Macro F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Learning rate curve
    axes[2].plot(epochs, history['lr'], 'm-o', markersize=3, label='Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.replace("-", "_").lower()}_training_curves.png',
                dpi=150, bbox_inches='tight')
    plt.show()
def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name.replace("-", "_").lower()}_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
# %%
# ===========================================================================
# MAIN EXECUTION: TRAIN ALL 4 MODELS
# ===========================================================================
# Store results for final comparison
all_results = {}
# --- Train CNN models ---
cnn_models = ['DenseNet121', 'ResNet50', 'EfficientNet-B3']
for model_name in cnn_models:
    model, history, test_loader, save_path = train_cnn(model_name)
    # Evaluate on test set
    report_dict, cm, preds, labels = evaluate_on_test(model, test_loader, model_name)
    # Plot training curves
    plot_training_curves(history, model_name)
    # Plot confusion matrix
    plot_confusion_matrix(cm, model_name)
    # Store results
    all_results[model_name] = {
        'report': report_dict,
        'cm': cm,
        'history': dict(history),
        'save_path': save_path,
    }
    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()
    print(f"\n  ✓ {model_name} complete. GPU memory freed.\n")
# %%
# --- Train DeiT-Base ---
model, history, test_loader, save_path = train_deit()
# Evaluate on test set
report_dict, cm, preds, labels = evaluate_on_test(model, test_loader, 'DeiT-Base')
# Plot training curves
plot_training_curves(history, 'DeiT-Base')
# Plot confusion matrix
plot_confusion_matrix(cm, 'DeiT-Base')
# Store results
all_results['DeiT-Base'] = {
    'report': report_dict,
    'cm': cm,
    'history': dict(history),
    'save_path': save_path,
}
del model
torch.cuda.empty_cache()
print(f"\n  ✓ DeiT-Base complete. GPU memory freed.\n")
# %%
# ===========================================================================
# FINAL COMPARISON TABLE
# ===========================================================================
print(f"\n{'='*80}")
print(f"  FINAL MODEL COMPARISON — TEST SET RESULTS")
print(f"{'='*80}")
header = f"{'Model':<20} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10}"
print(header)
print("-" * len(header))
best_model_name = None
best_overall_f1 = 0.0
best_cnn_name = None
best_cnn_f1 = 0.0
for model_name in ['DenseNet121', 'ResNet50', 'EfficientNet-B3', 'DeiT-Base']:
    r = all_results[model_name]['report']
    acc  = r['accuracy']
    prec = r['macro avg']['precision']
    rec  = r['macro avg']['recall']
    f1   = r['macro avg']['f1-score']
    print(f"{model_name:<20} | {acc:>10.4f} | {prec:>10.4f} | {rec:>10.4f} | {f1:>10.4f}")
    if f1 > best_overall_f1:
        best_overall_f1 = f1
        best_model_name = model_name
    if model_name != 'DeiT-Base' and f1 > best_cnn_f1:
        best_cnn_f1 = f1
        best_cnn_name = model_name
print(f"\n  ★ Best overall model:  {best_model_name} (F1 = {best_overall_f1:.4f})")
print(f"  ★ Best CNN model:      {best_cnn_name} (F1 = {best_cnn_f1:.4f})")
print(f"\n  → Best CNN ({best_cnn_name}) → Grad-CAM / Grad-CAM++ analysis")
print(f"  → DeiT-Base → Attention Rollout analysis")
# %%
# ===========================================================================
# PER-CLASS COMPARISON TABLE
# ===========================================================================
print(f"\n{'='*80}")
print(f"  PER-CLASS F1 SCORES")
print(f"{'='*80}")
header = f"{'Model':<20} | {'COVID':>8} | {'Normal':>8} | {'Viral Pn.':>10}"
print(header)
print("-" * len(header))
for model_name in ['DenseNet121', 'ResNet50', 'EfficientNet-B3', 'DeiT-Base']:
    r = all_results[model_name]['report']
    covid_f1  = r['COVID']['f1-score']
    normal_f1 = r['Normal']['f1-score']
    vp_f1     = r['Viral Pneumonia']['f1-score']
    print(f"{model_name:<20} | {covid_f1:>8.4f} | {normal_f1:>8.4f} | {vp_f1:>10.4f}")
# %%
# ===========================================================================
# SAVE COMPARISON RESULTS TO CSV
# ===========================================================================
import csv
with open('model_comparison_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'Macro Precision', 'Macro Recall',
                     'Macro F1', 'COVID F1', 'Normal F1', 'Viral Pneumonia F1',
                     'Best Epoch'])
    for model_name in ['DenseNet121', 'ResNet50', 'EfficientNet-B3', 'DeiT-Base']:
        r = all_results[model_name]['report']
        writer.writerow([
            model_name,
            f"{r['accuracy']:.4f}",
            f"{r['macro avg']['precision']:.4f}",
            f"{r['macro avg']['recall']:.4f}",
            f"{r['macro avg']['f1-score']:.4f}",
            f"{r['COVID']['f1-score']:.4f}",
            f"{r['Normal']['f1-score']:.4f}",
            f"{r['Viral Pneumonia']['f1-score']:.4f}",
            "N/A"  # epoch info is in checkpoint
        ])
print("\n  ✓ Results saved to model_comparison_results.csv")
# %%
# ===========================================================================
# SUMMARY — NEXT STEPS
# ===========================================================================
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRAINING COMPLETE — ALL 4 MODELS                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Saved checkpoints:                                                  ║
║    • best_densenet121.pth                                            ║
║    • best_resnet50.pth                                               ║
║    • best_efficientnet_b3.pth                                        ║
║    • best_deit_base.pth                                              ║
║                                                                      ║
║  Saved visualizations:                                               ║
║    • *_training_curves.png   (×4)                                    ║
║    • *_confusion_matrix.png  (×4)                                    ║
║                                                                      ║
║  Saved data:                                                         ║
║    • model_comparison_results.csv                                    ║
║                                                                      ║
║  NEXT: Part 2 — Explainability Analysis                             ║
║    • Grad-CAM / Grad-CAM++ on best CNN                              ║
║    • Attention Rollout on DeiT-Base                                  ║
║    • Insertion / Deletion / Entropy / AOPC evaluation                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
