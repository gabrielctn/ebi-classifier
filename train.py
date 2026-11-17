"""Training script for EBI vs NOT_EBI binary classification.

Overview
--------
This script trains an ImageNet-pretrained timm backbone on 3-channel ImageStream/IFC
exports (TIFF triplets) with weighted CrossEntropyLoss (2-class softmax).

Key features
------------
- Stratified train/val split or stratified K-fold CV.
- Optional class-imbalance handling (weighted loss and/or balanced sampling).
- Optional MixUp regularization.
- Rich validation metrics including threshold tuning (MCC/F1/balanced accuracy),
    plus ROC-AUC and PR-AUC from scikit-learn.
- Optional TensorBoard logging of metrics and example prediction groups (TP/TN/FP/FN).
- Optional threshold diagnostic plots for choosing a decision threshold.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os

# Force Matplotlib to use non-interactive backend to avoid crashes on macOS/MPS.
os.environ["MPLBACKEND"] = "Agg"

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import timm
import importlib

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    balanced_accuracy_score = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    matthews_corrcoef = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

try:
    from torchvision.utils import make_grid
except Exception:  # pragma: no cover
    make_grid = None  # type: ignore[assignment]

from dataset import (
    REQUIRED_CHANNELS_DEFAULT,
    ImagingFlowCytometryTripletDataset,
    TripletSample,
    build_triplet_index,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducible data splits and augmentation randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _unnormalize_for_viz(x: torch.Tensor, normalize: str) -> torch.Tensor:
    """Convert model-input tensor back to [0,1] RGB for TensorBoard image logging."""

    if normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
        x = x * std + mean
    return x.clamp(0.0, 1.0)


def pick_device() -> torch.device:
    """Pick the best available device (CUDA > Apple MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split(
    samples: Sequence[TripletSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[TripletSample], List[TripletSample]]:
    """Stratified hold-out split preserving class proportions as best as possible."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    idx_pos = [i for i, s in enumerate(samples) if s.label == 1]
    idx_neg = [i for i, s in enumerate(samples) if s.label == 0]

    rng = np.random.default_rng(seed)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_val = max(1, int(round(len(idx_pos) * val_ratio))) if len(idx_pos) > 0 else 0
    n_neg_val = max(1, int(round(len(idx_neg) * val_ratio))) if len(idx_neg) > 0 else 0

    val_idx = set(idx_pos[:n_pos_val] + idx_neg[:n_neg_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]

    return train, val


def stratified_kfold(
    samples: Sequence[TripletSample],
    k: int,
    seed: int,
) -> List[Tuple[List[TripletSample], List[TripletSample]]]:
    """Simple stratified K-fold split without sklearn.

    Returns list of (train_samples, val_samples) for each fold.
    """

    if k < 2:
        raise ValueError("k must be >= 2")

    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = np.array_split(np.array(pos, dtype=object), k)
    neg_folds = np.array_split(np.array(neg, dtype=object), k)

    folds: List[Tuple[List[TripletSample], List[TripletSample]]] = []
    for i in range(k):
        val = list(pos_folds[i].tolist()) + list(neg_folds[i].tolist())
        train: List[TripletSample] = []
        for j in range(k):
            if j == i:
                continue
            train.extend(pos_folds[j].tolist())
            train.extend(neg_folds[j].tolist())
        rng.shuffle(train)
        rng.shuffle(val)
        folds.append((train, val))

    return folds


def balanced_val_from_stratified(
    train_samples: Sequence[TripletSample],
    val_samples: Sequence[TripletSample],
    seed: int,
) -> Tuple[List[TripletSample], List[TripletSample]]:
    """Downsample the majority class in validation to match the minority count.

    This makes confusion-matrix-derived metrics less misleading when one class is rare.
    """

    val_pos = [s for s in val_samples if s.label == 1]
    val_neg = [s for s in val_samples if s.label == 0]
    if len(val_pos) == 0 or len(val_neg) == 0:
        return list(train_samples), list(val_samples)

    n = min(len(val_pos), len(val_neg))
    rng = np.random.default_rng(seed)
    rng.shuffle(val_pos)
    rng.shuffle(val_neg)
    new_val = val_pos[:n] + val_neg[:n]
    rng.shuffle(new_val)

    # Put the remaining validation samples back into training
    used_ids = {s.sample_id for s in new_val}
    moved_back = [s for s in val_samples if s.sample_id not in used_ids]
    new_train = list(train_samples) + moved_back
    return new_train, new_val


@torch.no_grad()
def compute_metrics(probs: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics from probabilities and labels.

    Args:
        probs: Shape [N] predicted probability for class=1 (EBI).
        y: Shape [N] labels (0/1).
        threshold: Decision threshold applied on probs.

    Returns:
        Dictionary of scalar metrics.
    """

    if accuracy_score is None:
        raise SystemExit(
            "scikit-learn is required for metrics. Install with: uv pip install scikit-learn"
        )

    probs = probs.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    preds = (probs >= float(threshold)).astype(np.int64)

    pred_pos_rate = float((preds == 1).mean())
    prob_mean = float(probs.mean())
    prob_std = float(probs.std())

    # Confusion matrix with explicit label order to keep TP/TN/FP/FN stable.
    cm = confusion_matrix(y_np, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    acc = float(accuracy_score(y_np, preds))
    prec = float(precision_score(y_np, preds, zero_division=0))
    rec = float(recall_score(y_np, preds, zero_division=0))
    f1 = float(f1_score(y_np, preds, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_np, preds))
    mcc = float(matthews_corrcoef(y_np, preds))

    # Threshold-free metrics. If a class is missing, sklearn will raise; return NaN.
    try:
        roc_auc = float(roc_auc_score(y_np, probs))
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_np, probs))
    except Exception:
        pr_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "pred_pos_rate": pred_pos_rate,
        "prob_mean": prob_mean,
        "prob_std": prob_std,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def threshold_curve(
    probs: np.ndarray,
    y: np.ndarray,
    metric: str = "mcc",
    num_thresholds: int = 201,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute score(metric) for a grid of decision thresholds.

    Uses scikit-learn metric implementations to avoid re-implementing common scores.

    Returns:
        (thresholds, scores)
    """

    if matthews_corrcoef is None:
        raise SystemExit(
            "scikit-learn is required for threshold curves. Install with: uv pip install scikit-learn"
        )

    metric = metric.lower().strip()
    thresholds = np.linspace(0.0, 1.0, int(num_thresholds), dtype=np.float64)
    probs = probs.astype(np.float64, copy=False)
    y = y.astype(np.int64, copy=False)

    scores: List[float] = []
    for t in thresholds:
        preds = (probs >= float(t)).astype(np.int64)
        if metric == "mcc":
            scores.append(float(matthews_corrcoef(y, preds)))
        elif metric == "f1":
            scores.append(float(f1_score(y, preds, zero_division=0)))
        elif metric in {"balanced_accuracy", "bal_acc"}:
            scores.append(float(balanced_accuracy_score(y, preds)))
        else:
            raise ValueError("threshold-metric must be one of: mcc, f1, balanced_accuracy")

    return thresholds, np.asarray(scores, dtype=np.float64)


def find_best_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    metric: str = "mcc",
    num_thresholds: int = 201,
) -> Tuple[float, float]:
    """Find best threshold in [0,1] by maximizing a chosen metric on validation.

    Returns (best_threshold, best_score).
    """

    if probs.ndim != 1 or y.ndim != 1:
        raise ValueError("probs and y must be 1D")
    if len(probs) != len(y):
        raise ValueError("probs and y must have same length")

    # If only one class is present, threshold tuning is meaningless.
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return 0.5, float("nan")

    thresholds, scores = threshold_curve(probs=probs, y=y, metric=metric, num_thresholds=num_thresholds)
    best_idx = int(np.nanargmax(scores))
    best_t = float(thresholds[best_idx])
    best_s = float(scores[best_idx])

    # Tie-breaker: if multiple thresholds give the same score, prefer one closest to 0.5
    ties = np.where(np.isclose(scores, best_s, atol=1e-12))[0]
    if len(ties) > 1:
        best_idx = int(ties[np.argmin(np.abs(thresholds[ties] - 0.5))])
        best_t = float(thresholds[best_idx])

    return best_t, best_s


def _require_matplotlib():
    """Import matplotlib lazily.

    This keeps plotting optional, avoids import costs during training, and prevents
    static-analysis import resolution issues when matplotlib isn't installed.
    """

    try:
        mpl = importlib.import_module("matplotlib")
        # Ensure a non-interactive backend for headless runs.
        if mpl.get_backend() != "Agg":
            mpl.use("Agg", force=True)
        plt_mod = importlib.import_module("matplotlib.pyplot")
        return mpl, plt_mod
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: uv pip install matplotlib"
        ) from e


def plot_threshold_diagnostics(
    *,
    probs: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
    scores: np.ndarray,
    best_threshold: float,
    metric: str,
    out_path: Path,
) -> None:
    """Save a figure that helps pick a decision threshold.

    The plot shows:
    - Distribution of predicted probabilities by class with the chosen threshold.
    - The chosen metric as a function of threshold.
    """

    _, plt = _require_matplotlib()

    probs = probs.astype(np.float64, copy=False)
    y = y.astype(np.int64, copy=False)

    pos = probs[y == 1]
    neg = probs[y == 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    try:
        # Panel 1: probability distributions
        bins = np.linspace(0.0, 1.0, 31)
        axes[0].hist(neg, bins=bins, alpha=0.6, density=True, label="NOT_EBI=0")
        axes[0].hist(pos, bins=bins, alpha=0.6, density=True, label="EBI=1")
        axes[0].axvline(float(best_threshold), color="black", linestyle="--", linewidth=2, label=f"thr={best_threshold:.3f}")
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_xlabel("Predicted probability p(EBI=1)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Predicted probability distribution")
        axes[0].legend(loc="best")

        # Panel 2: metric vs threshold
        axes[1].plot(thresholds, scores, linewidth=2)
        axes[1].axvline(float(best_threshold), color="black", linestyle="--", linewidth=2)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel("Decision threshold")
        axes[1].set_ylabel(metric)
        axes[1].set_title(f"{metric} vs threshold")

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
    finally:
        plt.close(fig)


def _find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    last: nn.Conv2d | None = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM")
    return last


def save_gradcam_figure(
    *,
    model: nn.Module,
    x: torch.Tensor,
    normalize: str,
    out_path: Path,
    class_idx: int = 1,
) -> None:
    """Save a Grad-CAM visualization figure for a small batch.

    Expects x as model-input tensor [B,3,H,W] (already normalized as configured).
    """

    _, plt = _require_matplotlib()

    device = next(model.parameters()).device
    x = x.to(device)

    target_layer = _find_last_conv2d(model)
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def bwd_hook(_, __, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        logits = model(x)
        score = logits[:, int(class_idx)].sum()
        score.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        acts = activations.detach()
        grads = gradients.detach()
        # Global-average-pool gradients to get channel weights
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(1)
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        x_vis = _unnormalize_for_viz(x.detach().cpu(), normalize=normalize)
        cam_cpu = cam.detach().cpu().numpy()

        b = int(x_vis.shape[0])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, b, figsize=(2.5 * b, 5))
        try:
            if b == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            cmap = plt.get_cmap("magma")

            for i in range(b):
                img = x_vis[i].permute(1, 2, 0).numpy()
                heat = cmap(cam_cpu[i])[:, :, :3]
                overlay = (0.55 * img + 0.45 * heat).clip(0.0, 1.0)

                axes[0, i].imshow(img)
                axes[0, i].axis("off")
                axes[0, i].set_title(f"img {i}")

                axes[1, i].imshow(overlay)
                axes[1, i].axis("off")
                axes[1, i].set_title("grad-cam")

            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
        finally:
            plt.close(fig)
    finally:
        h1.remove()
        h2.remove()


def make_balanced_sampler(samples: Sequence[TripletSample], seed: int) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that approximately balances the two classes."""
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    class_w = 1.0 / counts
    weights = class_w[labels]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
        generator=g,
    )


@torch.no_grad()
def evaluate_on_test_set(
    *,
    model: nn.Module,
    test_samples: Sequence[TripletSample],
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate the model on an independent test set.

    Args:
        model: Trained model in eval mode.
        test_samples: List of TripletSample for the test set.
        args: CLI args containing dataset/loader config.
        device: Device to run inference on (ignored on MPS - uses CPU for stability).
        threshold: Decision threshold tuned on validation set.

    Returns:
        Dictionary of test metrics.
    """
    if len(test_samples) == 0:
        return {}

    # On MPS, run test evaluation on CPU to avoid malloc crashes
    # This is a known MPS issue when switching between training and inference
    if device.type == "mps":
        torch.mps.synchronize()
        eval_device = torch.device("cpu")
        model = model.to(eval_device)
    else:
        eval_device = device

    model.eval()

    rgb_order = tuple(int(x) for x in str(args.rgb_order).split(","))

    test_ds = ImagingFlowCytometryTripletDataset(
        list(test_samples),
        image_size=args.image_size,
        required_channels=REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order=rgb_order,
        train=False,
        normalize=args.normalize,
        augment="none",
        intensity_norm=args.intensity_norm,
        percentile_low=float(args.percentile_low),
        percentile_high=float(args.percentile_high),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    all_probs = []
    all_labels = []

    for x, y in test_loader:
        x = x.to(eval_device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs)
        all_labels.append(y)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_probs, all_labels, threshold=threshold)
    return metrics


def make_model(
    model_name: str,
    *,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Build a timm model for 2-class classification (softmax)."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=2,
        in_chans=3,
        drop_path_rate=float(drop_path_rate),
        global_pool="avg",
    )
    return model


def _set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = bool(requires_grad)


def _freeze_backbone_keep_head_trainable(model: nn.Module) -> None:
    _set_requires_grad(model, False)
    # timm models expose a classifier/head via get_classifier()
    head = getattr(model, "get_classifier", None)
    if callable(head):
        clf = model.get_classifier()
        if clf is not None:
            for p in clf.parameters():
                p.requires_grad = True


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))


def _lr_finetune_schedule(
    *,
    epoch: int,
    freeze_epochs: int,
    warmup_epochs: int,
    total_epochs: int,
    lr_head: float,
    lr_finetune: float,
    min_lr: float,
) -> float:
    """Piecewise LR schedule.

    - Epochs 1..freeze_epochs: constant lr_head (head-only training)
    - Then warmup_epochs: linear warmup to lr_finetune
    - Then cosine decay to min_lr
    """

    if epoch <= freeze_epochs:
        return float(lr_head)

    fine_epoch = epoch - freeze_epochs  # 1-based within fine-tuning stage
    fine_total = max(1, total_epochs - freeze_epochs)

    warm = max(0, int(warmup_epochs))
    if warm > 0 and fine_epoch <= warm:
        return float(lr_finetune) * (fine_epoch / float(warm))

    # Cosine over the remaining fine-tune epochs
    t = fine_epoch - warm
    t_max = max(1, fine_total - warm)
    # Clamp to [0, t_max]
    t = min(max(0, int(t)), int(t_max))
    cos = 0.5 * (1.0 + math.cos(math.pi * (t / float(t_max))))
    return float(min_lr + (lr_finetune - min_lr) * cos)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mixup_alpha: float,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    losses: List[float] = []

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).to(torch.int64)

        # MixUp (optional) for 2-class softmax.
        # Uses soft labels with a weighted soft cross-entropy.
        y_soft: torch.Tensor | None = None
        if mixup_alpha and mixup_alpha > 0.0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1.0 - lam) * x[perm]

            y_onehot = torch.zeros((x.size(0), 2), device=x.device, dtype=torch.float32)
            y_onehot.scatter_(1, y.view(-1, 1), 1.0)
            y_soft = lam * y_onehot + (1.0 - lam) * y_onehot[perm]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if y_soft is None:
            loss = loss_fn(logits, y)
        else:
            logp = torch.log_softmax(logits, dim=1)
            # loss_fn may be CrossEntropyLoss with weights; emulate class weights for soft targets
            w = getattr(loss_fn, "weight", None)
            if w is not None:
                per_sample_w = (y_soft * w.view(1, -1)).sum(dim=1)
                loss = -(y_soft * logp).sum(dim=1)
                loss = (loss * per_sample_w).mean()
            else:
                loss = -(y_soft * logp).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float | str,
    threshold_metric: str,
    collect_diagnostics: bool = False,
    num_thresholds: int = 201,
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray] | None]:
    """Evaluate on validation loader.

    If `threshold == "auto"`, the decision threshold is tuned each epoch by
    maximizing `threshold_metric` on the validation set.
    """

    model.eval()

    losses: List[float] = []
    all_probs: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).to(torch.int64)

        logits = model(x)
        loss = loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]

        losses.append(float(loss.item()))
        all_probs.append(probs.detach().float().cpu())
        all_y.append(y.detach().cpu())

    diagnostics: Dict[str, np.ndarray] | None = None

    if all_probs:
        probs_cat = torch.cat(all_probs, dim=0)
        y_cat = torch.cat(all_y, dim=0)

        thr: float
        if isinstance(threshold, str) and threshold.lower() == "auto":
            # Tune the decision threshold on validation predictions.
            probs = probs_cat.cpu().numpy().astype(np.float64)
            y_np = y_cat.cpu().numpy().astype(np.int64)
            thr, thr_score = find_best_threshold(probs, y_np, metric=threshold_metric, num_thresholds=num_thresholds)
            metrics = compute_metrics(probs_cat, y_cat, threshold=thr)
            metrics["threshold_metric"] = str(threshold_metric)
            metrics["threshold_score"] = float(thr_score)
        else:
            thr = float(threshold)
            metrics = compute_metrics(probs_cat, y_cat, threshold=thr)
            metrics["threshold_metric"] = str(threshold_metric)

        if collect_diagnostics:
            probs = probs_cat.cpu().numpy().astype(np.float64)
            y_np = y_cat.cpu().numpy().astype(np.int64)
            thresholds, scores = threshold_curve(probs=probs, y=y_np, metric=threshold_metric, num_thresholds=num_thresholds)
            diagnostics = {
                "probs": probs,
                "y": y_np,
                "thresholds": thresholds,
                "scores": scores,
                "best_threshold": np.asarray([thr], dtype=np.float64),
            }
    else:
        metrics = {"accuracy": math.nan, "precision": math.nan, "recall": math.nan, "f1": math.nan}

    return (float(np.mean(losses)) if losses else math.nan), metrics, diagnostics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: str,
    best_metric_value: float,
    config: Dict,
    val_metrics: Dict[str, float],
) -> None:
    """Persist model/optimizer state + metadata to a .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_value": best_metric_value,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "val_metrics": val_metrics,
    }
    torch.save(payload, path)


@torch.no_grad()
def _log_merged_images_to_tensorboard(
    writer,
    samples: Sequence[TripletSample],
    image_size: int,
    max_per_class: int,
    rgb_channel_order: Sequence[int],
    intensity_norm: str,
    percentile_low: float,
    percentile_high: float,
    global_step: int = 0,
) -> None:
    """Log merged 3-channel images (pseudo-RGB) for pos/neg.

    Uses a non-normalized, non-augmented dataset so the Images tab looks natural.
    """

    if make_grid is None:
        return

    viz_ds = ImagingFlowCytometryTripletDataset(
        list(samples),
        image_size=image_size,
        required_channels=REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order=rgb_channel_order,
        train=False,
        normalize="none",
        augment="none",
        intensity_norm=intensity_norm,
        percentile_low=float(percentile_low),
        percentile_high=float(percentile_high),
    )

    pos_imgs: List[torch.Tensor] = []
    neg_imgs: List[torch.Tensor] = []

    for i in range(len(viz_ds)):
        x, y = viz_ds[i]
        # x: [3,H,W] float in [0,1]
        if int(y.item()) == 1:
            if len(pos_imgs) < max_per_class:
                pos_imgs.append(x.cpu())
        else:
            if len(neg_imgs) < max_per_class:
                neg_imgs.append(x.cpu())
        if len(pos_imgs) >= max_per_class and len(neg_imgs) >= max_per_class:
            break

    if pos_imgs:
        grid = make_grid(torch.stack(pos_imgs, dim=0), nrow=min(4, len(pos_imgs)))
        writer.add_image("images/val_pos", grid, global_step=global_step)
    if neg_imgs:
        grid = make_grid(torch.stack(neg_imgs, dim=0), nrow=min(4, len(neg_imgs)))
        writer.add_image("images/val_neg", grid, global_step=global_step)


@torch.no_grad()
def _log_prediction_groups_to_tensorboard(
    writer,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    normalize: str,
    max_per_group: int,
    global_step: int,
) -> None:
    """Log TP/TN/FP/FN image grids to TensorBoard (Images tab)."""

    if make_grid is None:
        return
    if max_per_group <= 0:
        return

    model.eval()
    groups: Dict[str, List[torch.Tensor]] = {"tp": [], "tn": [], "fp": [], "fn": []}

    for x, y in loader:
        # x is model-input (possibly normalized); keep a CPU copy for visualization.
        x_vis = _unnormalize_for_viz(x.detach().cpu(), normalize=normalize)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).to(torch.int64)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).to(torch.int64)
        y_int = y

        preds_cpu = preds.detach().cpu().numpy().astype(np.int64)
        y_cpu = y_int.detach().cpu().numpy().astype(np.int64)

        for i in range(len(preds_cpu)):
            if len(groups["tp"]) >= max_per_group and len(groups["tn"]) >= max_per_group and len(groups["fp"]) >= max_per_group and len(groups["fn"]) >= max_per_group:
                break
            pi = int(preds_cpu[i])
            yi = int(y_cpu[i])
            if pi == 1 and yi == 1 and len(groups["tp"]) < max_per_group:
                groups["tp"].append(x_vis[i])
            elif pi == 0 and yi == 0 and len(groups["tn"]) < max_per_group:
                groups["tn"].append(x_vis[i])
            elif pi == 1 and yi == 0 and len(groups["fp"]) < max_per_group:
                groups["fp"].append(x_vis[i])
            elif pi == 0 and yi == 1 and len(groups["fn"]) < max_per_group:
                groups["fn"].append(x_vis[i])

        if len(groups["tp"]) >= max_per_group and len(groups["tn"]) >= max_per_group and len(groups["fp"]) >= max_per_group and len(groups["fn"]) >= max_per_group:
            break

    for key, imgs in groups.items():
        if imgs:
            grid = make_grid(torch.stack(imgs, dim=0), nrow=min(4, len(imgs)))
            writer.add_image(f"images/pred_{key}", grid, global_step=global_step)


def _run_one_experiment(
    *,
    root: Path,
    args: argparse.Namespace,
    train_samples: Sequence[TripletSample],
    val_samples: Sequence[TripletSample],
    run_dir: Path,
    fold: int,
) -> Dict[str, float]:
    """Train/validate for one split or one CV fold.

    Writes:
    - config.json
    - metrics.csv (per epoch)
    - checkpoints/best.pt and checkpoints/last.pt
    - tb/ (optional)
    """

    device = pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)

    n_pos = sum(s.label == 1 for s in train_samples)
    n_neg = sum(s.label == 0 for s in train_samples)
    val_pos = sum(s.label == 1 for s in val_samples)
    val_neg = sum(s.label == 0 for s in val_samples)

    config = vars(args) | {
        "device": str(device),
        "fold": int(fold),
        "train_count": int(len(train_samples)),
        "val_count": int(len(val_samples)),
        "train_pos": int(n_pos),
        "train_neg": int(n_neg),
        "val_pos": int(val_pos),
        "val_neg": int(val_neg),
        "required_channels": list(REQUIRED_CHANNELS_DEFAULT),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(
        f"Fold {fold:02d} | Train: {len(train_samples)} (EBI=1: {n_pos}, NOT_EBI=0: {n_neg}) | "
        f"Val: {len(val_samples)} (EBI=1: {val_pos}, NOT_EBI=0: {val_neg}) | device={device}"
    )

    rgb_order = tuple(int(x) for x in str(args.rgb_order).split(","))

    train_ds = ImagingFlowCytometryTripletDataset(
        list(train_samples),
        image_size=args.image_size,
        required_channels=REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order=rgb_order,
        train=True,
        normalize=args.normalize,
        augment=args.augment,
        intensity_norm=args.intensity_norm,
        percentile_low=float(args.percentile_low),
        percentile_high=float(args.percentile_high),
    )
    val_ds = ImagingFlowCytometryTripletDataset(
        list(val_samples),
        image_size=args.image_size,
        required_channels=REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order=rgb_order,
        train=False,
        normalize=args.normalize,
        augment="none",
        intensity_norm=args.intensity_norm,
        percentile_low=float(args.percentile_low),
        percentile_high=float(args.percentile_high),
    )

    sampler = None
    if args.sampling == "balanced":
        sampler = make_balanced_sampler(train_samples, seed=args.seed)
    elif args.sampling == "auto":
        if n_pos > 0 and n_neg > 0:
            ratio = max(n_pos, n_neg) / max(1.0, float(min(n_pos, n_neg)))
            if ratio >= 3.0:
                # Only activate sampling automatically when imbalance is severe.
                sampler = make_balanced_sampler(train_samples, seed=args.seed)

    # Use num_workers=0 on MPS to avoid multiprocessing malloc crashes on macOS
    n_workers = 0 if device.type == "mps" else int(args.num_workers)
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    model = make_model(args.model, pretrained=not args.no_pretrained, drop_path_rate=float(args.drop_path_rate))
    model.to(device)

    # Weighted CrossEntropyLoss to counter class imbalance.
    if n_pos > 0 and n_neg > 0:
        total = float(n_pos + n_neg)
        w_pos = total / (2.0 * float(n_pos))
        w_neg = total / (2.0 * float(n_neg))
        class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    else:
        class_w = None

    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    freeze_epochs = int(args.freeze_epochs)
    if freeze_epochs > 0:
        _freeze_backbone_keep_head_trainable(model)
    optimizer = _make_optimizer(model, lr=float(args.lr), weight_decay=float(args.weight_decay))

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv_header_writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "epoch",
                "train_loss",
                "val_loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "mcc",
                "roc_auc",
                "pr_auc",
                "threshold",
                "threshold_metric",
                "threshold_score",
                "pred_pos_rate",
                "prob_mean",
                "prob_std",
                "tp",
                "tn",
                "fp",
                "fn",
            ],
        )
        csv_header_writer.writeheader()

    best_metric = "mcc"
    best_metric_value = -1.0
    best_epoch = 0

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            raise SystemExit(
                "TensorBoard logging requested but tensorboard is not installed. "
                "Install with: uv pip install tensorboard"
            )
        tb_dir = run_dir / "tb"
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("config/json", json.dumps(config, indent=2), global_step=0)
        if args.tb_image_count > 0:
            _log_merged_images_to_tensorboard(
                writer,
                samples=val_samples,
                image_size=args.image_size,
                max_per_class=int(args.tb_image_count),
                rgb_channel_order=rgb_order,
                intensity_norm=str(args.intensity_norm),
                percentile_low=float(args.percentile_low),
                percentile_high=float(args.percentile_high),
                global_step=0,
            )

    try:
        for epoch in range(1, args.epochs + 1):
            # Transition from head-only to full fine-tuning.
            if freeze_epochs > 0 and epoch == (freeze_epochs + 1):
                _set_requires_grad(model, True)
                optimizer = _make_optimizer(model, lr=float(args.lr_finetune), weight_decay=float(args.weight_decay))

            lr_now = _lr_finetune_schedule(
                epoch=epoch,
                freeze_epochs=freeze_epochs,
                warmup_epochs=int(args.warmup_epochs),
                total_epochs=int(args.epochs),
                lr_head=float(args.lr),
                lr_finetune=float(args.lr_finetune),
                min_lr=float(args.min_lr),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = float(lr_now)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                mixup_alpha=args.mixup_alpha,
            )

            threshold_value: float | str
            if args.threshold.lower() == "auto":
                threshold_value = "auto"
            else:
                threshold_value = float(args.threshold)

            val_loss, val_metrics, val_diag = validate(
                model,
                val_loader,
                loss_fn,
                device,
                threshold=threshold_value,
                threshold_metric=args.threshold_metric,
                collect_diagnostics=bool(args.plot_threshold),
            )

            if args.gradcam and (epoch % max(1, int(args.gradcam_every)) == 0):
                # Use a small batch from validation for explainability.
                try:
                    x0, _ = next(iter(val_loader))
                    x0 = x0[: int(args.gradcam_count)]
                    save_gradcam_figure(
                        model=model,
                        x=x0,
                        normalize=str(args.normalize),
                        out_path=run_dir / "plots" / f"gradcam_epoch_{epoch:03d}.png",
                        class_idx=1,
                    )
                except Exception as e:
                    print(f"Warning: Grad-CAM failed: {e}")

            row = {
                "fold": int(fold),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{
                    k: val_metrics.get(k, math.nan)
                    for k in [
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                        "mcc",
                        "roc_auc",
                        "pr_auc",
                        "threshold",
                        "threshold_metric",
                        "threshold_score",
                        "pred_pos_rate",
                        "prob_mean",
                        "prob_std",
                        "tp",
                        "tn",
                        "fp",
                        "fn",
                    ]
                },
            }

            with metrics_path.open("a", newline="") as f:
                csv_writer = csv.DictWriter(f, fieldnames=row.keys())
                csv_writer.writerow(row)

            if writer is not None:
                writer.add_scalar("loss/train", float(train_loss), epoch)
                writer.add_scalar("loss/val", float(val_loss), epoch)

                for k in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "mcc",
                    "roc_auc",
                    "pr_auc",
                    "threshold",
                    "threshold_score",
                    "pred_pos_rate",
                    "prob_mean",
                    "prob_std",
                ]:
                    if k in val_metrics:
                        writer.add_scalar(f"val/{k}", float(val_metrics[k]), epoch)
                writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)

                if args.tb_pred_image_count > 0 and (epoch % max(1, int(args.tb_pred_image_every)) == 0):
                    thr = float(val_metrics.get("threshold", 0.5))
                    _log_prediction_groups_to_tensorboard(
                        writer,
                        model=model,
                        loader=val_loader,
                        device=device,
                        threshold=thr,
                        normalize=args.normalize,
                        max_per_group=int(args.tb_pred_image_count),
                        global_step=epoch,
                    )

            print(
                f"Fold {fold:02d} | Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc={row['accuracy']:.3f} f1={row['f1']:.3f} mcc={row['mcc']:.3f} "
                f"auc={row['roc_auc']:.3f} ap={row['pr_auc']:.3f} "
                f"thr={row['threshold']:.2f} pos%={row['pred_pos_rate']:.2f} p={row['prob_mean']:.2f}±{row['prob_std']:.2f} "
                f"(tp={int(row['tp'])} tn={int(row['tn'])} fp={int(row['fp'])} fn={int(row['fn'])})"
            )

            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                best_metric_value=best_metric_value,
                config=config,
                val_metrics=val_metrics,
            )

            current_mcc = float(val_metrics.get("mcc", -1.0))
            if current_mcc > best_metric_value:
                best_metric_value = current_mcc
                best_epoch = int(epoch)
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_metric_value=best_metric_value,
                    config=config,
                    val_metrics=val_metrics,
                )

                # When enabled, keep an up-to-date plot for threshold selection.
                if args.plot_threshold and val_diag is not None:
                    plot_threshold_diagnostics(
                        probs=val_diag["probs"],
                        y=val_diag["y"],
                        thresholds=val_diag["thresholds"],
                        scores=val_diag["scores"],
                        best_threshold=float(val_diag["best_threshold"][0]),
                        metric=str(args.threshold_metric),
                        out_path=run_dir / "plots" / "threshold_best.png",
                    )

    finally:
        # ---- MPS-safe cleanup to avoid malloc crashes on macOS ----
        # Close TensorBoard writer first (before deleting model)
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass

        # Sync MPS before any cleanup
        if device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

        # Force garbage collection
        gc.collect()

        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return {"best_mcc": float(best_metric_value), "best_epoch": float(best_epoch)}


def _train_final_model(
    *,
    root: Path,
    args: argparse.Namespace,
    all_samples: Sequence[TripletSample],
    run_dir: Path,
    target_epochs: int,
    cv_threshold: float,
) -> nn.Module:
    """Train a final model on ALL data (no validation split) after CV.

    This is the standard ML workflow: use CV to estimate performance and find
    the best epoch, then train a final model on all available training data.

    Args:
        root: Project root path.
        args: CLI arguments.
        all_samples: All training samples (combined train+val from CV).
        run_dir: Directory to save final model artifacts.
        target_epochs: Number of epochs to train (typically avg best_epoch from CV).
        cv_threshold: Decision threshold averaged from CV folds.

    Returns:
        Trained model (on device, in eval mode).
    """
    device = pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)

    n_pos = sum(s.label == 1 for s in all_samples)
    n_neg = sum(s.label == 0 for s in all_samples)

    config = vars(args) | {
        "device": str(device),
        "fold": "final",
        "train_count": int(len(all_samples)),
        "val_count": 0,
        "train_pos": int(n_pos),
        "train_neg": int(n_neg),
        "val_pos": 0,
        "val_neg": 0,
        "target_epochs": int(target_epochs),
        "cv_threshold": float(cv_threshold),
        "required_channels": list(REQUIRED_CHANNELS_DEFAULT),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(
        f"\n{'='*60}\n"
        f"FINAL MODEL TRAINING (on all {len(all_samples)} samples)\n"
        f"EBI=1: {n_pos}, NOT_EBI=0: {n_neg} | epochs={target_epochs} | device={device}\n"
        f"{'='*60}",
        flush=True,
    )

    rgb_order = tuple(int(x) for x in str(args.rgb_order).split(","))

    train_ds = ImagingFlowCytometryTripletDataset(
        list(all_samples),
        image_size=args.image_size,
        required_channels=REQUIRED_CHANNELS_DEFAULT,
        rgb_channel_order=rgb_order,
        train=True,
        normalize=args.normalize,
        augment=args.augment,
        intensity_norm=args.intensity_norm,
        percentile_low=float(args.percentile_low),
        percentile_high=float(args.percentile_high),
    )

    # Balanced sampling if needed
    sampler = None
    if args.sampling == "balanced":
        sampler = make_balanced_sampler(all_samples, seed=args.seed)
    elif args.sampling == "auto":
        if n_pos > 0 and n_neg > 0:
            ratio = max(n_pos, n_neg) / max(1.0, float(min(n_pos, n_neg)))
            if ratio >= 3.0:
                sampler = make_balanced_sampler(all_samples, seed=args.seed)

    # Use num_workers=0 on MPS to avoid multiprocessing malloc crashes on macOS
    n_workers = 0 if device.type == "mps" else int(args.num_workers)
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    model = make_model(
        args.model,
        pretrained=not args.no_pretrained,
        drop_path_rate=args.drop_path_rate,
    )
    model.to(device)

    # Class weights for loss
    if n_pos > 0 and n_neg > 0:
        pos_weight = float(n_neg) / float(n_pos)
        class_weights = torch.tensor([1.0, pos_weight], device=device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Freeze backbone initially
    _freeze_backbone_keep_head_trainable(model)
    optimizer = _make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    mixup_alpha = float(args.mixup_alpha)

    # Training loop
    for epoch in range(1, target_epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            _set_requires_grad(model, True)
            optimizer = _make_optimizer(model, lr=args.lr_finetune, weight_decay=args.weight_decay)

        # Update learning rate
        new_lr = _lr_finetune_schedule(
            epoch=epoch,
            freeze_epochs=args.freeze_epochs,
            warmup_epochs=args.warmup_epochs,
            total_epochs=target_epochs,
            lr_head=args.lr,
            lr_finetune=args.lr_finetune,
            min_lr=args.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # Use the same train_one_epoch function for consistency
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            mixup_alpha=mixup_alpha,
        )

        print(f"Final Model | Epoch {epoch:03d}/{target_epochs} | train_loss={avg_loss:.4f} | lr={new_lr:.2e}")

    # Save final checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_config = config.copy()
    final_config["cv_threshold"] = cv_threshold

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": target_epochs,
            "config": final_config,
            "cv_threshold": cv_threshold,
        },
        ckpt_dir / "final.pt",
    )
    print(f"Final model saved to: {ckpt_dir / 'final.pt'}")

    model.eval()
    return model


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Binary classification for ImageStream EBI vs NOT_EBI")
    parser.add_argument("--data-root", type=str, default=".", help="Project root containing EBI/ and NOT_EBI/")
    parser.add_argument("--ebi-dir", type=str, default="EBI", help="Folder name for positive class")
    parser.add_argument("--not-ebi-dir", type=str, default="NOT_EBI", help="Folder name for negative class")
    parser.add_argument(
        "--ebi-test-dir",
        type=str,
        default="EBI_TEST_SET",
        help="Folder name for positive class test set",
    )
    parser.add_argument(
        "--not-ebi-test-dir",
        type=str,
        default="NOT_EBI_TEST_SET",
        help="Folder name for negative class test set",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip evaluation on independent test set even if test directories exist",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for head-only stage")
    parser.add_argument("--lr-finetune", type=float, default=1e-5, help="LR for fine-tuning stage")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs at start of fine-tuning")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Freeze backbone for first N epochs")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >=2, run stratified K-fold cross-validation (overrides --val-ratio)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="auto",
        help="Decision threshold on predicted p(class=1). Use 'auto' to tune on validation each epoch.",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="mcc",
        choices=["mcc", "f1", "balanced_accuracy"],
        help="Metric to maximize when --threshold=auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model",
        type=str,
        default="convnextv2_tiny.fcmae_ft_in22k_in1k",
        help="timm model name",
    )
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--normalize", type=str, default="none", choices=["imagenet", "none"])
    parser.add_argument(
        "--rgb-order",
        type=str,
        default="7,11,3",
        help="Comma-separated channel ids mapped to R,G,B (default: 7,11,3)",
    )
    parser.add_argument(
        "--intensity-norm",
        type=str,
        default="percentile",
        choices=["dtype", "percentile"],
        help="Per-channel intensity normalization before ImageNet normalization",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--augment",
        type=str,
        default="basic",
        choices=["none", "basic", "strong", "sota"],
        help="Augmentation policy applied to training images",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.1,
        help="MixUp alpha (0 disables). Typical: 0.1 to 0.4",
    )
    parser.add_argument(
        "--val-balanced",
        action="store_true",
        help="Downsample validation to be class-balanced (moves unused val samples back into training)",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="auto",
        choices=["auto", "none", "balanced"],
        help="Training sampling strategy (auto enables balanced sampling when classes are highly imbalanced)",
    )
    # Kept for backward-compat CLI; no-op with CrossEntropy.
    parser.add_argument("--no-pos-weight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard logs under runs/<timestamp>/tb",
    )
    parser.add_argument(
        "--tb-image-count",
        type=int,
        default=8,
        help="How many merged images per class to log to TensorBoard (validation split)",
    )
    parser.add_argument(
        "--tb-pred-image-count",
        type=int,
        default=8,
        help="How many images per group (tp/tn/fp/fn) to log to TensorBoard",
    )
    parser.add_argument(
        "--tb-pred-image-every",
        type=int,
        default=1,
        help="Log prediction-group images every N epochs (1 = every epoch)",
    )
    parser.add_argument(
        "--plot-threshold",
        action="store_true",
        help="Save a threshold diagnostic plot under runs/<timestamp>/.../plots/threshold_best.png",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Save Grad-CAM overlay figure(s) under runs/<timestamp>/.../plots/gradcam_epoch_XXX.png",
    )
    parser.add_argument("--gradcam-count", type=int, default=8, help="How many val images to include in Grad-CAM")
    parser.add_argument("--gradcam-every", type=int, default=5, help="Save Grad-CAM every N epochs")
    args = parser.parse_args()

    # Make it obvious the script started (helps when stdout is buffered).
    print(
        f"Starting train.py | model={args.model} epochs={args.epochs} batch={args.batch_size} "
        f"cv_folds={args.cv_folds} tensorboard={bool(args.tensorboard)}",
        flush=True,
    )

    set_seed(args.seed)

    root = Path(args.data_root).resolve()
    ebi_dir = root / args.ebi_dir
    not_dir = root / args.not_ebi_dir

    print(f"Indexing triplets in: {ebi_dir} and {not_dir}", flush=True)
    t0 = time.perf_counter()
    samples = build_triplet_index(ebi_dir=ebi_dir, not_ebi_dir=not_dir, required_channels=REQUIRED_CHANNELS_DEFAULT)
    dt = time.perf_counter() - t0
    print(f"Indexing done: {len(samples)} samples (t={dt:.2f}s)", flush=True)
    if len(samples) == 0:
        raise SystemExit(
            f"No samples found. Expected TIFF triplets in {ebi_dir} and {not_dir} with naming like <id>_Ch11.ome.tif"
        )

    # ----- Index test set (if available) -----
    ebi_test_dir = root / args.ebi_test_dir
    not_ebi_test_dir = root / args.not_ebi_test_dir
    test_samples: List[TripletSample] = []
    
    if not args.skip_test and ebi_test_dir.exists() and not_ebi_test_dir.exists():
        print(f"Indexing test set in: {ebi_test_dir} and {not_ebi_test_dir}", flush=True)
        t0 = time.perf_counter()
        test_samples = build_triplet_index(
            ebi_dir=ebi_test_dir,
            not_ebi_dir=not_ebi_test_dir,
            required_channels=REQUIRED_CHANNELS_DEFAULT,
        )
        dt = time.perf_counter() - t0
        n_test_pos = sum(s.label == 1 for s in test_samples)
        n_test_neg = sum(s.label == 0 for s in test_samples)
        print(
            f"Test set indexed: {len(test_samples)} samples "
            f"(EBI=1: {n_test_pos}, NOT_EBI=0: {n_test_neg}) (t={dt:.2f}s)",
            flush=True,
        )
    elif not args.skip_test:
        print(
            f"Test set directories not found ({ebi_test_dir}, {not_ebi_test_dir}). "
            "Skipping test evaluation. Use --skip-test to suppress this message.",
            flush=True,
        )

    run_root = root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    device = pick_device()

    if args.cv_folds and int(args.cv_folds) >= 2:
        if args.val_balanced:
            print("Warning: --val-balanced is disabled during cross-validation.")

        folds = stratified_kfold(samples, k=int(args.cv_folds), seed=args.seed)
        summary_rows: List[Dict[str, float]] = []
        
        for i, (train_samples, val_samples) in enumerate(folds, start=1):
            fold_dir = run_root / f"fold_{i:02d}"
            res = _run_one_experiment(
                root=root,
                args=args,
                train_samples=train_samples,
                val_samples=val_samples,
                run_dir=fold_dir,
                fold=i,
            )
            summary_rows.append({"fold": float(i), **res})

            # Note: Per-fold test evaluation removed due to MPS memory issues on macOS.
            # Test evaluation is done once at the end with the final model.

        cv_path = run_root / "cv_summary.csv"
        with cv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "best_mcc", "best_epoch"])
            w.writeheader()
            for r in summary_rows:
                w.writerow({"fold": int(r["fold"]), "best_mcc": r["best_mcc"], "best_epoch": int(r["best_epoch"])})

        best_mccs = np.array([r["best_mcc"] for r in summary_rows], dtype=np.float64)
        best_epochs = np.array([r["best_epoch"] for r in summary_rows], dtype=np.float64)
        
        # Compute average threshold from fold checkpoints
        fold_thresholds = []
        for i in range(1, int(args.cv_folds) + 1):
            ckpt_path = run_root / f"fold_{i:02d}" / "checkpoints" / "best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                thr = ckpt.get("val_metrics", {}).get("threshold", 0.5)
                fold_thresholds.append(thr)
        avg_threshold = float(np.mean(fold_thresholds)) if fold_thresholds else 0.5
        avg_best_epoch = int(round(np.mean(best_epochs)))
        
        print(
            f"Done CV ({int(args.cv_folds)} folds). "
            f"best_mcc mean={float(np.nanmean(best_mccs)):.3f} std={float(np.nanstd(best_mccs)):.3f} | "
            f"avg_best_epoch={avg_best_epoch} | avg_threshold={avg_threshold:.3f}"
        )

        # ----- Train final model on ALL data -----
        final_dir = run_root / "final_model"
        final_model = _train_final_model(
            root=root,
            args=args,
            all_samples=samples,
            run_dir=final_dir,
            target_epochs=avg_best_epoch,
            cv_threshold=avg_threshold,
        )

        # ----- Evaluate final model on test set -----
        if len(test_samples) > 0:
            final_test_metrics = evaluate_on_test_set(
                model=final_model,
                test_samples=test_samples,
                args=args,
                device=device,
                threshold=avg_threshold,
            )
            final_test_metrics["threshold_from_cv"] = avg_threshold
            final_test_metrics["training_epochs"] = avg_best_epoch

            # Save final test metrics
            final_test_csv_path = run_root / "final_test_metrics.csv"
            final_test_fieldnames = [
                "threshold_from_cv", "training_epochs", "accuracy", "precision", "recall", "f1",
                "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"
            ]
            with final_test_csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=final_test_fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerow(final_test_metrics)

            print(
                f"\n{'='*60}\n"
                f"FINAL MODEL TEST SET EVALUATION ({len(test_samples)} samples)\n"
                f"Model trained on ALL {len(samples)} training samples for {avg_best_epoch} epochs\n"
                f"threshold={avg_threshold:.3f} (avg from CV)\n"
                f"acc={final_test_metrics.get('accuracy', 0):.3f} "
                f"f1={final_test_metrics.get('f1', 0):.3f} "
                f"mcc={final_test_metrics.get('mcc', 0):.3f} "
                f"auc={final_test_metrics.get('roc_auc', 0):.3f} "
                f"pr_auc={final_test_metrics.get('pr_auc', 0):.3f}\n"
                f"tp={int(final_test_metrics.get('tp', 0))} tn={int(final_test_metrics.get('tn', 0))} "
                f"fp={int(final_test_metrics.get('fp', 0))} fn={int(final_test_metrics.get('fn', 0))}\n"
                f"Results saved to: {final_test_csv_path}\n"
                f"Final model saved to: {final_dir / 'checkpoints' / 'final.pt'}\n"
                f"{'='*60}",
                flush=True,
            )

        print(f"Artifacts: {run_root}")
    else:
        train_samples, val_samples = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)
        if args.val_balanced:
            train_samples, val_samples = balanced_val_from_stratified(train_samples, val_samples, seed=args.seed)

        _run_one_experiment(
            root=root,
            args=args,
            train_samples=train_samples,
            val_samples=val_samples,
            run_dir=run_root,
            fold=0,
        )

        # Evaluate on test set using best checkpoint
        if len(test_samples) > 0:
            best_ckpt_path = run_root / "checkpoints" / "best.pt"
            if best_ckpt_path.exists():
                ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model = make_model(
                    args.model,
                    pretrained=False,
                    drop_path_rate=args.drop_path_rate,
                )
                model.load_state_dict(ckpt["model_state"])
                model.to(device)
                model.eval()

                val_threshold = ckpt.get("val_metrics", {}).get("threshold", 0.5)
                test_metrics = evaluate_on_test_set(
                    model=model,
                    test_samples=test_samples,
                    args=args,
                    device=device,
                    threshold=val_threshold,
                )
                test_metrics["threshold_from_val"] = val_threshold

                # Save test metrics
                test_csv_path = run_root / "test_metrics.csv"
                test_fieldnames = ["threshold_from_val", "accuracy", "precision", "recall", "f1",
                                   "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"]
                with test_csv_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=test_fieldnames, extrasaction="ignore")
                    w.writeheader()
                    w.writerow(test_metrics)

                print(
                    f"\n{'='*60}\n"
                    f"TEST SET EVALUATION ({len(test_samples)} samples)\n"
                    f"threshold={val_threshold:.3f} (from validation)\n"
                    f"acc={test_metrics.get('accuracy', 0):.3f} "
                    f"f1={test_metrics.get('f1', 0):.3f} "
                    f"mcc={test_metrics.get('mcc', 0):.3f} "
                    f"auc={test_metrics.get('roc_auc', 0):.3f} "
                    f"pr_auc={test_metrics.get('pr_auc', 0):.3f}\n"
                    f"tp={int(test_metrics.get('tp', 0))} tn={int(test_metrics.get('tn', 0))} "
                    f"fp={int(test_metrics.get('fp', 0))} fn={int(test_metrics.get('fn', 0))}\n"
                    f"Results saved to: {test_csv_path}\n"
                    f"{'='*60}",
                    flush=True,
                )

        print(f"Done. Run artifacts in: {run_root}")


if __name__ == "__main__":
    main()
