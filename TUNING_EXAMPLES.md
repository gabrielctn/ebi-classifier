# Example Hyperparameter Tuning Configurations

This file contains ready-to-use commands for different tuning scenarios.

## Quick Start (First-Time Users)

```bash
# 1. Install dependencies
./setup.sh
source .venv-secure/bin/activate

# 2. Run a quick 10-trial tune to test the system
python tune.py --n-trials 10 --tune-optimizer --tune-augmentation

# 3. View results
python analyze_tuning.py
open optuna_studies/ebi_study_*/report.html
```

## Scenario-Based Configurations

### Scenario 1: Fast Improvement (2-3 hours)

**Goal**: Quick wins with minimal time investment  
**Best for**: Initial exploration, tight deadlines

```bash
python tune.py \
  --n-trials 30 \
  --tune-optimizer \
  --tune-augmentation \
  --pruning \
  --study-name quick_tune \
  --storage sqlite:///optuna_studies/quick_tune.db
```

**Expected**: +1-3% MCC improvement

### Scenario 2: Balanced Optimization (6-8 hours)

**Goal**: Good performance improvement with reasonable time  
**Best for**: Most production use cases

```bash
python tune.py \
  --n-trials 50 \
  --tune-all \
  --pruning \
  --pruning-warmup 10 \
  --study-name balanced_tune \
  --storage sqlite:///optuna_studies/balanced_tune.db
```

**Expected**: +2-4% MCC improvement

### Scenario 3: Comprehensive Search (12-16 hours)

**Goal**: Maximum performance, thorough exploration  
**Best for**: Final production model, research

```bash
python tune.py \
  --n-trials 100 \
  --tune-all \
  --pruning \
  --pruning-warmup 15 \
  --study-name comprehensive_tune \
  --storage sqlite:///optuna_studies/comprehensive_tune.db \
  --timeout 57600  # 16 hours
```

**Expected**: +3-5% MCC improvement

### Scenario 4: Parallel Multi-Day Search

**Goal**: Exhaustive search with parallel workers  
**Best for**: Cluster computing, overnight runs

```bash
# Run on machine with sufficient resources (16GB+ RAM)
python tune.py \
  --n-trials 200 \
  --tune-all \
  --pruning \
  --n-jobs 4 \
  --study-name exhaustive_tune \
  --storage sqlite:///optuna_studies/exhaustive_tune.db
```

**Expected**: +4-6% MCC improvement

### Scenario 5: Architecture Search Only

**Goal**: Find best model backbone  
**Best for**: When you want to keep other settings fixed

```bash
python tune.py \
  --n-trials 50 \
  --tune-model \
  --pruning \
  --study-name model_search \
  --storage sqlite:///optuna_studies/model_search.db
```

**Note**: This is slower because different models have different training speeds

### Scenario 6: Fine-Grained Optimization

**Goal**: Optimize learning rate schedule precisely  
**Best for**: After initial tuning, to squeeze extra performance

```bash
# First, find good model with Scenario 2
# Then, fine-tune optimizer settings:

python tune.py \
  --n-trials 40 \
  --tune-optimizer \
  --study-name lr_refinement \
  --storage sqlite:///optuna_studies/lr_refinement.db
```

## Resuming Studies

If a study is interrupted, resume it:

```bash
# Resume any of the above studies
python tune.py \
  --study-name balanced_tune \
  --storage sqlite:///optuna_studies/balanced_tune.db \
  --n-trials 50  # Add 50 more trials
```

## Monitoring Progress

```bash
# In a separate terminal, watch progress:
watch -n 60 'python analyze_tuning.py --no-report'

# Or check specific study:
watch -n 60 'python analyze_tuning.py \
  --study-dir optuna_studies/balanced_tune \
  --no-report'
```

## After Tuning

```bash
# 1. Generate final report
python analyze_tuning.py --study-dir optuna_studies/balanced_tune

# 2. Review best parameters
cat optuna_studies/balanced_tune/best_params.json

# 3. Train final model with best params
./optuna_studies/balanced_tune/train_with_best_params.sh

# 4. Check results
tensorboard --logdir runs/
```

## Custom Search Spaces

To modify what hyperparameters are tuned, edit [tune.py](tune.py):

```python
# In suggest_hyperparameters() function:

# Example: Narrow learning rate range based on initial results
params["lr"] = trial.suggest_float("lr", 1e-4, 5e-4, log=True)

# Example: Add specific models to test
model_choices = [
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "efficientnet_b3.ra2_in1k",
]
params["model"] = trial.suggest_categorical("model", model_choices)

# Example: Fix certain parameters
params["batch_size"] = 32  # Don't tune, always use 32
```

## Troubleshooting Common Issues

### Out of Memory

```bash
# Reduce parallel jobs
python tune.py --n-trials 50 --tune-all --n-jobs 1

# Or exclude large models by editing tune.py
```

### Too Slow

```bash
# Enable aggressive pruning
python tune.py --n-trials 50 --tune-all --pruning --pruning-warmup 5

# Focus on fast hyperparameters only
python tune.py --n-trials 40 --tune-optimizer --tune-augmentation
```

### No Improvement

```bash
# Check if search space is reasonable
python analyze_tuning.py

# Look at parameter importance plot
# Focus tuning on important parameters only
```

## Best Practices Checklist

- [ ] Always use `--storage` for resumability
- [ ] Start with quick tune (Scenario 1) to verify setup
- [ ] Use `--pruning` to save time
- [ ] Monitor progress with `analyze_tuning.py`
- [ ] Validate final model on test set
- [ ] Document best parameters for reproduction

## Performance Comparison

| Scenario | Trials | Time | Expected MCC | Use Case |
|----------|--------|------|--------------|----------|
| Default params | - | - | ~92% | Baseline |
| Quick (Scenario 1) | 30 | 2-3h | ~93-94% | Fast iteration |
| Balanced (Scenario 2) | 50 | 6-8h | ~94-95% | Production |
| Comprehensive (Scenario 3) | 100 | 12-16h | ~95-96% | Research |
| Exhaustive (Scenario 4) | 200 | 24-48h | ~96-97% | Final model |

*Note: Actual improvements depend on dataset characteristics*
