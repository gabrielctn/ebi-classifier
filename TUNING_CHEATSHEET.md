# Hyperparameter Tuning Quick Reference

## Installation

```bash
./setup.sh && source .venv-secure/bin/activate
```

## Common Commands

```bash
# Quick tune (20 trials, ~2 hours)
python tune.py --n-trials 20 --tune-all

# Fast tune with pruning
python tune.py --n-trials 30 --tune-all --pruning

# Extensive tune (production)
python tune.py --n-trials 100 --tune-all --pruning \
  --storage sqlite:///optuna_studies/production.db

# Parallel optimization (4 workers)
python tune.py --n-trials 80 --tune-all --n-jobs 4

# Resume previous study
python tune.py --study-name my_study \
  --storage sqlite:///optuna_studies/my_study.db \
  --n-trials 50
```

## Focused Tuning

```bash
# Only optimizer (learning rates, weight decay) - FAST
python tune.py --tune-optimizer --n-trials 30

# Only augmentation - FAST
python tune.py --tune-augmentation --n-trials 20

# Only model architecture - SLOW but high impact
python tune.py --tune-model --n-trials 50 --pruning

# Optimizer + augmentation (good balance)
python tune.py --tune-optimizer --tune-augmentation --n-trials 40
```

## Analyze Results

```bash
# Generate HTML report
python analyze_tuning.py

# Specific study
python analyze_tuning.py --study-dir optuna_studies/ebi_study_20260112_143022

# Just print summary (no HTML)
python analyze_tuning.py --no-report

# Compare top 20 trials
python analyze_tuning.py --top-n 20
```

## Train with Best Parameters

```bash
# Use auto-generated script
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Or manually from best_params.json
cat optuna_studies/ebi_study_*/best_params.json
```

## Tuning Options

| Flag | Description | Impact | Speed |
|------|-------------|--------|-------|
| `--tune-optimizer` | LR, weight decay, warmup | High | Fast |
| `--tune-augmentation` | Augmentation strategy | Medium | Fast |
| `--tune-model` | Model architecture | High | Slow |
| `--tune-regularization` | MixUp, sampling | Medium | Fast |
| `--tune-training` | Epochs, freeze, batch size | Medium | Medium |
| `--tune-all` | All groups | High | Slow |

## Performance Expectations

| Trials | Time | Improvement |
|--------|------|-------------|
| 10 | 1-2h | +0-2% MCC |
| 30 | 3-4h | +1-3% MCC |
| 50 | 6-8h | +2-4% MCC |
| 100 | 12-16h | +3-5% MCC |

## Key Files

```
optuna_studies/ebi_study_*/
├── best_params.json              # Best hyperparameters
├── train_with_best_params.sh     # Training command
├── report.html                   # Full analysis report
└── visualizations/*.html         # Interactive plots
```

## Tips

- Always use `--storage` for production studies
- Enable `--pruning` to save 30-50% time
- Start with `--tune-optimizer` (fastest, good results)
- Monitor with `python analyze_tuning.py --no-report`
- Validate on test set after tuning
