# Quick Start Guide - EBI Classifier

## Windows: using the packaged classifier

Extract `EBI_Classifier_Windows_Security_2026-11-17.zip` completely, open the extracted folder, and double-click `START_EBI.bat`. Keep the black window open while you use the app in your browser. The first start can take several minutes. For image naming, channel mapping, and troubleshooting, read [START_HERE.txt](windows/START_HERE.txt) or the [user guide](GUIDE_UTILISATEUR.md).

## Mac: using the packaged classifier

Double-click `EBI_Classifier_Mac_Security_2026-11-17.zip`, open the extracted folder, and double-click `START_EBI.command`. Keep Terminal open while you use the app in your browser. The package supports Intel and Apple Silicon Macs with macOS 13 or later. See [START_HERE.txt](macos/START_HERE.txt) for detailed instructions and troubleshooting.

## Linux: using the packaged classifier

Extract `EBI_Classifier_Linux_Security_2026-11-17.zip`, open the extracted folder, and double-click `EBI_Classifier.desktop`. If required, allow the launcher to run. Keep the terminal open while you use the browser. If your desktop blocks the launcher, open a terminal in the extracted folder and run `./START_EBI.sh`. See [START_HERE.txt](linux/START_HERE.txt) for supported systems and troubleshooting.

The remaining instructions are for the development environment and training tools.

## Development installation

```bash
# 1. Navigate to project directory
cd /path/to/EBI

# 2. Run setup
./setup.sh

# 3. Activate environment
source .venv-secure/bin/activate
```

## Usage Scenarios

### Scenario 1: I just want to classify images

```bash
# Launch web interface
python app.py

# The browser opens automatically; otherwise use the address printed in the terminal
# Drag TIFF files, click "Classify", download results
```

### Scenario 2: I want to train on my own data

```bash
# 1. Organize your data
#    EBI/sample001_Ch{3,7,11}.ome.tif
#    NOT_EBI/sample101_Ch{3,7,11}.ome.tif

# 2. Train with cross-validation
python train.py --cv-folds 5 --tensorboard

# 3. Monitor training
tensorboard --logdir runs/

# 4. Results in: runs/YYYYMMDD-HHMMSS/
```

### Scenario 3: I want the best possible model

```bash
# 1. Run hyperparameter optimization (6-8 hours)
python tune.py --n-trials 50 --tune-all --pruning

# 2. View results
python analyze_tuning.py
open optuna_studies/ebi_study_*/report.html

# 3. Train final model with best params
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Measure the resulting model on an independent test set
```

### Scenario 4: I'm in a hurry

```bash
# Quick optimization (2-3 hours)
python tune.py --n-trials 30 --tune-optimizer --tune-augmentation --pruning

# Analyze
python analyze_tuning.py

# Train
./optuna_studies/ebi_study_*/train_with_best_params.sh

# Measure the resulting model on an independent test set
```

## Common Commands

```bash
# Training
python train.py --help                           # See all options
python train.py --cv-folds 5                     # 5-fold CV (recommended)
python train.py --model efficientnet_b3.ra2_in1k # Different model
python train.py --augment sota                   # Best augmentation

# Hyperparameter tuning
python tune.py --n-trials 20 --tune-all          # Full search
python tune.py --tune-optimizer                  # Only LR/optimizer
python tune.py --n-jobs 4                        # Parallel (4 workers)

# Analysis
python analyze_tuning.py                         # Latest study
python analyze_tuning.py --top-n 20              # Compare top 20
python analyze_tuning.py --no-report             # Console only

# Monitoring
tensorboard --logdir runs/                       # View training
watch -n 60 'python analyze_tuning.py --no-report' # Watch tuning
```

## File Organization

```
Your data:
  EBI/
    sample001_Ch3.ome.tif
    sample001_Ch7.ome.tif
    sample001_Ch11.ome.tif
    sample002_Ch*.ome.tif
    ...
  
  NOT_EBI/
    sample101_Ch3.ome.tif
    sample101_Ch7.ome.tif
    sample101_Ch11.ome.tif
    ...

Results appear in:
  runs/YYYYMMDD-HHMMSS/         # Training results
  optuna_studies/ebi_study_*/   # Tuning results
```

## Troubleshooting

**Problem**: "Command not found: uv"
```bash
# Install uv first (macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Problem**: "Out of memory"
```bash
# Use smaller batch size
python train.py --batch-size 16

# Or smaller model
python train.py --model convnextv2_tiny.fcmae_ft_in22k_in1k
```

**Problem**: Training is slow
```bash
# Reduce epochs for testing
python train.py --epochs 10

# Or use single split instead of CV
python train.py --val-ratio 0.2
```

**Problem**: Tuning takes too long
```bash
# Enable aggressive pruning
python tune.py --n-trials 30 --pruning --pruning-warmup 5

# Or tune fewer hyperparameters
python tune.py --tune-optimizer --n-trials 20
```

## Next Steps

1. **Read full documentation**: [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)
2. **See examples**: [TUNING_EXAMPLES.md](TUNING_EXAMPLES.md)
3. **Quick reference**: [TUNING_CHEATSHEET.md](TUNING_CHEATSHEET.md)

## Getting Help

- Check `--help` flag on any script
- Review documentation in project root
- Examine example runs in `runs/` directory
- Look at tuning reports in `optuna_studies/*/report.html`
