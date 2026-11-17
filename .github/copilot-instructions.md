# EBI Classifier — AI Agent Instructions

## Project Overview

Binary image classifier for Imaging Flow Cytometry (ImageStream/IFC) data. Distinguishes **EBI** (Erythroblastic Islands) from **NOT_EBI** cells using 3-channel OME-TIFF microscopy images.

**Domain**: Biomedical imaging / Cell biology  
**Primary Metric**: Matthews Correlation Coefficient (MCC) — handles class imbalance better than accuracy  
**Best Performance**: ~92% MCC with 5-fold cross-validation

## Architecture

```
train.py                    # Full training pipeline with K-fold CV, threshold tuning, TensorBoard
tune.py                     # Hyperparameter optimization with Optuna (Bayesian search)
analyze_tuning.py           # Analysis and visualization of tuning results
dataset.py                  # Dataset indexing, TIFF loading, augmentations, pseudo-RGB conversion
app.py                      # Gradio inference UI for biologists
setup.sh                    # Environment setup with uv (Astral package manager)
EBI/                        # Positive class: TIFF triplets (Ch3, Ch7, Ch11 per sample)
NOT_EBI/                    # Negative class: TIFF triplets (Ch3, Ch7, Ch11 per sample)
runs/                       # Training artifacts: checkpoints, metrics.csv, TensorBoard logs
optuna_studies/             # Hyperparameter tuning results and visualizations
HYPERPARAMETER_TUNING.md    # Complete guide to hyperparameter optimization
TUNING_CHEATSHEET.md        # Quick reference for tuning commands
```

## Critical Data Format

Each sample consists of **3 single-channel OME-TIFF files** with naming convention:
```
<sample_id>_Ch3.ome.tif   # Channel 3
<sample_id>_Ch7.ome.tif   # Channel 7
<sample_id>_Ch11.ome.tif  # Channel 11
```

The dataset loader in [dataset.py](dataset.py) groups these into triplets and maps to pseudo-RGB:
- **R** ← Ch7, **G** ← Ch11, **B** ← Ch3 (configurable via `--rgb-order`)

## Key Commands

```bash
# Environment setup (uses uv, not pip)
./setup.sh
source .venv/bin/activate

# Training with 5-fold cross-validation (recommended)
python train.py --cv-folds 5 --tensorboard --model convnextv2_tiny.fcmae_ft_in22k_in1k

# Quick single-split training
python train.py --val-ratio 0.2 --epochs 15

# Hyperparameter tuning (Optuna with Bayesian optimization)
python tune.py --n-trials 30 --tune-all --pruning
python analyze_tuning.py  # Generate visualization report

# Train with optimized hyperparameters
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Launch inference UI
python app.py

# View training results
tensorboard --logdir runs/
```

## Model & Training Conventions

- **Backbone**: timm models (default: `convnextv2_tiny.fcmae_ft_in22k_in1k`)
- **Strategy**: Freeze backbone for `--freeze-epochs 3`, then fine-tune with lower LR
- **Loss**: Weighted CrossEntropyLoss + optional MixUp
- **Threshold**: Auto-tuned per epoch by maximizing MCC on validation

## Important Implementation Details

1. **Intensity normalization**: Use `--intensity-norm percentile` (clips to 1st-99th percentile) for 16-bit microscopy images, not dtype-max scaling

2. **Class imbalance handling**: 
   - `--sampling balanced` (WeightedRandomSampler)
   - Positive class weight in loss function
   - Stratified splits preserve class proportions

3. **Augmentations** (`--augment sota`):
   - RandomRotate90 (interpolation-free)
   - RandomGamma, RandomGaussianNoise
   - RandomCoarseDropout (Cutout-like)
   - RandAugment from torchvision

4. **Checkpoint structure** in `runs/<timestamp>/fold_XX/`:
   - `config.json` — full hyperparameters
   - `checkpoints/best.pt` — best MCC on validation
   - `metrics.csv` — per-epoch metrics
   - `plots/threshold_best.png` — decision threshold diagnostics

## Code Patterns

When modifying dataset loading:
```python
# Samples are indexed via build_triplet_index()
samples = build_triplet_index(ebi_dir, not_ebi_dir, required_channels=(11, 7, 3))

# Dataset expects TripletSample dataclass
@dataclass(frozen=True)
class TripletSample:
    sample_id: str
    label: int  # 1=EBI, 0=NOT_EBI
    channels: Dict[int, Path]  # channel_id -> file path
```

When adding metrics, follow the `compute_metrics()` pattern in train.py — return dict with threshold-dependent metrics + threshold-free metrics (ROC-AUC, PR-AUC).

## Dependencies

Managed via uv (not pip). Key packages: `torch`, `torchvision`, `timm`, `tifffile`, `scikit-learn`, `tensorboard`, `matplotlib`

## Do NOT

- Use standard ImageNet normalization for raw microscopy images (use `--normalize none` + percentile scaling)
- Assume 8-bit images — these are 16-bit TIFFs
- Create temporary files in the data directories (EBI/, NOT_EBI/)
