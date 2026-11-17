# Hyperparameter Tuning Guide

This guide explains how to use the state-of-the-art hyperparameter optimization system for the EBI classifier.

## Overview

The tuning system uses **Optuna**, a leading hyperparameter optimization framework that implements:

- **Bayesian Optimization** via Tree-structured Parzen Estimator (TPE) sampler
- **Early Stopping** with median pruning to discard unpromising trials
- **Parallel Execution** for faster optimization
- **Rich Visualizations** for understanding hyperparameter importance
- **Resumable Studies** with database persistence

## Quick Start

### 1. Install Dependencies

```bash
./setup.sh
source .venv-secure/bin/activate
```

This will install Optuna, Plotly, and other required packages.

### 2. Run Your First Tuning

```bash
# Quick tuning (20 trials, ~2-3 hours)
python tune.py --n-trials 20 --tune-all

# Fast tuning with pruning (stops bad trials early)
python tune.py --n-trials 30 --tune-all --pruning
```

### 3. View Results

```bash
# Generate comprehensive HTML report
python analyze_tuning.py

# View the report in your browser
open optuna_studies/ebi_study_*/report.html
```

## Tuning Strategies

### Full Optimization (Recommended for Production)

Tune all hyperparameters for best performance:

```bash
python tune.py \
  --n-trials 100 \
  --tune-all \
  --pruning \
  --n-jobs 2 \
  --storage sqlite:///optuna_studies/full_tune.db
```

**Time**: ~8-12 hours (depends on hardware)  
**Expected improvement**: 2-5% MCC over default settings

### Focused Optimization

Tune specific hyperparameter groups:

```bash
# Only tune learning rates and optimizer settings (fast)
python tune.py \
  --tune-optimizer \
  --n-trials 30 \
  --pruning

# Only tune augmentation strategy
python tune.py \
  --tune-augmentation \
  --n-trials 20

# Tune model architecture (slower but potentially high impact)
python tune.py \
  --tune-model \
  --n-trials 50 \
  --pruning
```

### Parallel Optimization

Run multiple trials simultaneously (requires sufficient RAM/GPU):

```bash
# 4 parallel workers
python tune.py \
  --n-trials 80 \
  --tune-all \
  --n-jobs 4 \
  --storage sqlite:///optuna_studies/parallel.db
```

**Note**: Each worker needs ~4GB RAM. For MPS/GPU, trials run sequentially per device.

### Resumable Studies

Continue a previous optimization:

```bash
# Start a named study with database storage
python tune.py \
  --study-name my_optimization \
  --storage sqlite:///optuna_studies/my_study.db \
  --n-trials 50 \
  --tune-all

# Later, resume the same study
python tune.py \
  --study-name my_optimization \
  --storage sqlite:///optuna_studies/my_study.db \
  --n-trials 50  # Adds 50 more trials
```

## Hyperparameter Groups

### Model Architecture (`--tune-model`)

- **model**: ConvNeXtV2, EfficientNet, ResNet, ViT variants
- **image_size**: 224, 256, or 288 pixels
- **drop_path_rate**: Stochastic depth regularization (0.0-0.3)

**Impact**: High (model choice critical)  
**Time**: Slow (different models have different speeds)

### Optimizer Settings (`--tune-optimizer`)

- **lr**: Learning rate for head training (1e-5 to 1e-3)
- **lr_finetune**: Learning rate for full fine-tuning (1e-6 to 5e-4)
- **min_lr**: Minimum LR for cosine decay
- **weight_decay**: L2 regularization (0.0-0.2)
- **warmup_epochs**: Linear warmup period (0-10)

**Impact**: Medium-High  
**Time**: Fast (same model architecture)

### Data Augmentation (`--tune-augmentation`)

- **augment**: basic, strong, or sota (SOTA = RandAugment)
- **intensity_norm**: dtype vs percentile-based normalization
- **percentile_low/high**: Percentile clipping range

**Impact**: Medium (depends on dataset)  
**Time**: Fast

### Regularization (`--tune-regularization`)

- **mixup_alpha**: MixUp strength (0.0-0.4, 0=disabled)
- **sampling**: Class balancing strategy (auto/none/balanced)

**Impact**: Medium  
**Time**: Fast

### Training Schedule (`--tune-training`)

- **epochs**: Total training epochs (15-40)
- **freeze_epochs**: Epochs to freeze backbone (0-8)
- **batch_size**: 16, 32, or 64

**Impact**: Medium  
**Time**: Variable (more epochs = slower)

## Understanding Results

### Optimization History

Shows MCC improvement over trials. Look for:
- **Convergence**: Flat line means search is converging
- **Jumps**: New best hyperparameters found
- **Plateaus**: May need more trials or different search space

### Parameter Importance

Identifies which hyperparameters matter most:
- **High importance**: Focus tuning here
- **Low importance**: Use default values

### Parallel Coordinate Plot

Shows relationship between hyperparameters and performance:
- **Lines**: Each line is one trial
- **Color**: Performance (blue=bad, red=good)
- **Patterns**: Look for common patterns in good trials

### Slice Plot

Shows performance vs. individual hyperparameters:
- **X-axis**: Hyperparameter value
- **Y-axis**: MCC score
- **Trend**: Optimal value range

## Training with Best Parameters

After tuning completes, train the final model:

```bash
# Option 1: Use generated script (easiest)
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Option 2: Manual command with best params
python train.py \
  --model convnextv2_base.fcmae_ft_in22k_in1k \
  --lr 0.00015 \
  --lr-finetune 0.00003 \
  --weight-decay 0.08 \
  --epochs 28 \
  --batch-size 32 \
  --augment sota \
  --mixup-alpha 0.2 \
  --cv-folds 5 \
  --tensorboard \
  --plot-threshold
```

## Advanced Usage

### Custom Search Spaces

Edit [tune.py](tune.py) `suggest_hyperparameters()` function to modify search ranges:

```python
# Example: Expand learning rate range
params["lr"] = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

# Example: Add new hyperparameter
params["my_param"] = trial.suggest_categorical("my_param", ["a", "b", "c"])
```

### Multi-Objective Optimization

Optimize for multiple objectives (e.g., MCC + training speed):

```python
# In tune.py objective() function:
return mean_mcc, training_time  # Return tuple

# When creating study:
study = optuna.create_study(
    directions=["maximize", "minimize"],  # MCC, time
)
```

### Conditional Search Spaces

Some hyperparameters depend on others:

```python
# Only tune percentiles if using percentile normalization
if params["intensity_norm"] == "percentile":
    params["percentile_low"] = trial.suggest_float("percentile_low", 0.1, 2.0)
    params["percentile_high"] = trial.suggest_float("percentile_high", 97.0, 99.9)
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Phase 1: Quick exploration (1-2 hours)
python tune.py --n-trials 20 --tune-optimizer --tune-augmentation

# Phase 2: Focused search (4-6 hours)
python tune.py --n-trials 50 --tune-all --pruning

# Phase 3: Fine-tuning (8-12 hours, optional)
python tune.py --n-trials 100 --tune-all --pruning --storage sqlite:///final.db
```

### 2. Use Database Storage

Always use `--storage` for important studies:

```bash
python tune.py \
  --storage sqlite:///optuna_studies/production.db \
  --study-name production_v1 \
  --n-trials 100
```

Benefits:
- Resume interrupted studies
- Analyze results anytime
- Share with team

### 3. Monitor Progress

```bash
# Terminal 1: Run tuning
python tune.py --n-trials 100 --tune-all

# Terminal 2: Watch progress
watch -n 30 "python analyze_tuning.py --no-report"
```

### 4. Enable Pruning for Speed

Pruning stops unpromising trials early:

```bash
python tune.py --n-trials 100 --pruning --pruning-warmup 10
```

Typically saves 30-50% computation time.

### 5. Validate on Test Set

After tuning, always validate on independent test set:

```bash
# Train with best params (from generated script)
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Results will include test set metrics automatically
```

## Troubleshooting

### "Trial timed out"

Increase timeout:

```bash
python tune.py --trial-timeout 7200  # 2 hours per trial
```

### "Out of memory"

Reduce parallel jobs or batch size:

```bash
python tune.py --n-jobs 1  # Sequential
```

Or modify search space in tune.py to exclude large batch sizes.

### "No improvement after many trials"

- Check if search space includes good values
- Use parameter importance plot to focus on important hyperparameters
- Try different sampler (edit tune.py)
- Dataset may be too small for complex models

### "Best params don't reproduce performance"

- Ensure same random seed
- Check that train.py uses same data split
- Cross-validation variance is normal (±1-2% MCC)

## Performance Expectations

Typical results from hyperparameter tuning:

| Trials | Time    | Expected MCC Improvement |
|--------|---------|--------------------------|
| 10     | 1-2h    | +0-2%                    |
| 30     | 3-4h    | +1-3%                    |
| 50     | 6-8h    | +2-4%                    |
| 100    | 12-16h  | +3-5%                    |

**Baseline (default)**: ~92% MCC  
**After tuning**: ~94-96% MCC (expected)

## Files and Outputs

```
optuna_studies/
└── ebi_study_20260112_143022/
    ├── config.json                    # Tuning configuration
    ├── study_statistics.json          # Summary stats
    ├── best_params.json               # Best hyperparameters found
    ├── train_with_best_params.sh      # Executable training script
    ├── report.html                    # Interactive HTML report
    ├── visualizations/
    │   ├── optimization_history.html
    │   ├── param_importances.html
    │   ├── parallel_coordinate.html
    │   └── slice_plot.html
    └── trial_XXXX/
        ├── params.json                # Trial hyperparameters
        ├── stdout.txt                 # Training output
        └── stderr.txt                 # Error messages
```

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **TPE Algorithm**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (NeurIPS 2011)
- **Hyperparameter Importance**: Hutter et al., "An Efficient Approach for Assessing Hyperparameter Importance" (ICML 2014)

## See Also

- [GUIDE_UTILISATEUR.md](GUIDE_UTILISATEUR.md) - Basic training guide
- [train.py](train.py) - Training script documentation
- [tune.py](tune.py) - Tuning script source code
- [analyze_tuning.py](analyze_tuning.py) - Analysis script
