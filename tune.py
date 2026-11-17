"""Hyperparameter optimization for EBI classifier using Optuna.

This script implements state-of-the-art Bayesian hyperparameter optimization with:
- TPE (Tree-structured Parzen Estimator) sampler for intelligent search
- Median pruner for early stopping of unpromising trials
- Multi-objective optimization support (MCC + efficiency)
- Integrated with existing train.py infrastructure
- Rich visualization and reporting

Usage Examples
--------------
# Quick search (10 trials)
python tune.py --n-trials 10

# Extensive search with early stopping
python tune.py --n-trials 50 --pruning

# Resume a previous study
python tune.py --study-name my_study --n-trials 20

# Distributed optimization across multiple processes
python tune.py --n-trials 100 --n-jobs 4

# Focus on specific hyperparameter groups
python tune.py --tune-optimizer --tune-augmentation --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )
except ImportError:
    print("Error: Optuna not installed. Run: uv pip install optuna", file=sys.stderr)
    sys.exit(1)


def suggest_hyperparameters(
    trial: optuna.Trial,
    tune_model: bool = True,
    tune_optimizer: bool = True,
    tune_augmentation: bool = True,
    tune_regularization: bool = True,
    tune_training: bool = True,
) -> Dict[str, Any]:
    """Suggest hyperparameters for a single trial.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters.
    tune_model : bool
        Whether to tune model architecture hyperparameters.
    tune_optimizer : bool
        Whether to tune optimizer hyperparameters (LR, weight decay, etc.).
    tune_augmentation : bool
        Whether to tune data augmentation strategy.
    tune_regularization : bool
        Whether to tune regularization (dropout, MixUp, etc.).
    tune_training : bool
        Whether to tune training schedule (epochs, freeze strategy, etc.).
    
    Returns
    -------
    params : Dict[str, Any]
        Dictionary of hyperparameters to pass to train.py.
    """
    params: Dict[str, Any] = {}

    # ========== Model Architecture ==========
    if tune_model:
        # Model selection from proven architectures for vision
        model_choices = ["convnextv2_tiny.fcmae_ft_in22k_in1k"]
        params["model"] = trial.suggest_categorical("model", model_choices)

        # DropPath/StochasticDepth rate
        params["drop_path_rate"] = trial.suggest_float("drop_path_rate", 0.0, 0.3)

        # Image size (larger may improve performance but slower)
        params["image_size"] = trial.suggest_categorical("image_size", [224])

    # ========== Optimizer & Learning Rate ==========
    if tune_optimizer:
        # Learning rates with log scale
        params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        params["lr_finetune"] = trial.suggest_float("lr_finetune", 1e-6, 5e-4, log=True)
        params["min_lr"] = trial.suggest_float("min_lr", 1e-7, 1e-5, log=True)

        # Weight decay (critical for generalization)
        params["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.2)

        # Warmup epochs
        params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 10)

    # ========== Data Augmentation ==========
    if tune_augmentation:
        params["augment"] = trial.suggest_categorical("augment", ["basic", "strong", "sota"])

        # Intensity normalization strategy
        params["intensity_norm"] = trial.suggest_categorical("intensity_norm", ["dtype", "percentile"])

        if params.get("intensity_norm") == "percentile":
            params["percentile_low"] = trial.suggest_float("percentile_low", 0.1, 2.0)
            params["percentile_high"] = trial.suggest_float("percentile_high", 97.0, 99.9)

    # ========== Regularization ==========
    if tune_regularization:
        # MixUp alpha (0 disables)
        params["mixup_alpha"] = trial.suggest_float("mixup_alpha", 0.0, 0.4)

        # Class imbalance handling
        params["sampling"] = trial.suggest_categorical("sampling", ["auto", "none", "balanced"])

    # ========== Training Schedule ==========
    if tune_training:
        # Total epochs
        params["epochs"] = trial.suggest_int("epochs", 15, 40)

        # Freeze backbone strategy
        params["freeze_epochs"] = trial.suggest_int("freeze_epochs", 0, 8)

        # Batch size (power of 2 for efficiency)
        params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

    return params


def objective(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
    tune_groups: Dict[str, bool],
) -> float:
    """Objective function to maximize (validation MCC).
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    base_args : argparse.Namespace
        Base arguments from command line (data paths, etc.).
    tune_groups : Dict[str, bool]
        Which hyperparameter groups to tune.
    
    Returns
    -------
    mcc : float
        Best validation MCC achieved during training.
    """
    # Suggest hyperparameters for this trial
    params = suggest_hyperparameters(trial, **tune_groups)
    
    # Build command for train.py
    cmd = [
        sys.executable,
        "train.py",
        "--cv-folds", str(base_args.cv_folds),  # Configurable CV folds
        "--data-root", str(base_args.data_root),
        "--seed", str(base_args.seed),
        "--num-workers", str(base_args.num_workers),  # Configurable workers
        "--tensorboard",  # Keep TensorBoard for monitoring
        "--skip-test",  # Skip test evaluation during tuning
    ]
    
    # Add suggested hyperparameters to command
    for key, value in params.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Create trial-specific run directory
    trial_dir = Path(base_args.output_dir) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trial parameters
    param_file = trial_dir / "params.json"
    with param_file.open("w") as f:
        json.dump(params, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number} | Testing: {params}")
    print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=base_args.trial_timeout,
        )
        
        # Save stdout and stderr for debugging
        (trial_dir / "stdout.txt").write_text(result.stdout)
        (trial_dir / "stderr.txt").write_text(result.stderr)
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed with return code {result.returncode}")
            print(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
            return float("-inf")
        
        # Parse CV results from the most recent run
        runs_dir = Path("runs")
        latest_run = max(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        cv_summary = latest_run / "cv_summary.csv"
        
        if not cv_summary.exists():
            print(f"Trial {trial.number}: cv_summary.csv not found")
            return float("-inf")
        
        # Read mean MCC from CV summary
        import csv
        with cv_summary.open() as f:
            reader = csv.DictReader(f)
            fold_mccs = [float(row["best_mcc"]) for row in reader]
        
        mean_mcc = float(np.mean(fold_mccs))
        std_mcc = float(np.std(fold_mccs))
        
        # Store trial artifacts path
        trial.set_user_attr("run_dir", str(latest_run))
        trial.set_user_attr("mean_mcc", mean_mcc)
        trial.set_user_attr("std_mcc", std_mcc)
        
        # Report intermediate value for pruning (use mean - std for conservative estimate)
        trial.report(mean_mcc - std_mcc, step=0)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"Trial {trial.number} completed | MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
        
        return mean_mcc
    
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out after {base_args.trial_timeout}s")
        return float("-inf")
    
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return float("-inf")


def create_visualizations(study: optuna.Study, output_dir: Path) -> None:
    """Generate and save Optuna visualization plots.
    
    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    output_dir : Path
        Directory to save plots.
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(str(viz_dir / "optimization_history.html"))
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_html(str(viz_dir / "parallel_coordinate.html"))
        
        # Parameter importances
        try:
            fig = plot_param_importances(study)
            fig.write_html(str(viz_dir / "param_importances.html"))
        except RuntimeError:
            print("Skipping param_importances (requires >=2 completed trials)")
        
        # Slice plot
        fig = plot_slice(study)
        fig.write_html(str(viz_dir / "slice_plot.html"))
        
        # Contour plots (select most important pairs)
        try:
            fig = plot_contour(study, params=["lr", "lr_finetune"])
            fig.write_html(str(viz_dir / "contour_lr.html"))
        except Exception:
            pass  # Skip if params not present
        
        try:
            fig = plot_contour(study, params=["batch_size", "epochs"])
            fig.write_html(str(viz_dir / "contour_training.html"))
        except Exception:
            pass
        
        print(f"\nVisualizations saved to: {viz_dir}")
        print(f"Open {viz_dir}/optimization_history.html in a browser to view results.")
    
    except Exception as e:
        print(f"Warning: Could not generate all visualizations: {e}")


def save_best_params(study: optuna.Study, output_dir: Path) -> None:
    """Save best hyperparameters to JSON and generate training command.
    
    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    output_dir : Path
        Directory to save results.
    """
    best_trial = study.best_trial
    
    # Save best parameters
    best_params_file = output_dir / "best_params.json"
    with best_params_file.open("w") as f:
        json.dump(
            {
                "trial_number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
            },
            f,
            indent=2,
        )
    
    # Generate training command
    cmd_file = output_dir / "train_with_best_params.sh"
    cmd_parts = ["python train.py"]
    
    for key, value in best_trial.params.items():
        cmd_parts.append(f"--{key.replace('_', '-')} {value}")
    
    cmd_parts.extend([
        "--cv-folds 5",  # Use 5-fold for final training
        "--tensorboard",
        "--plot-threshold",
    ])
    
    with cmd_file.open("w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Training command with best hyperparameters from Optuna tuning\n")
        f.write("# Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write(" \\\n  ".join(cmd_parts) + "\n")
    
    cmd_file.chmod(0o755)  # Make executable
    
    print(f"\nBest hyperparameters saved to: {best_params_file}")
    print(f"Training script saved to: {cmd_file}")
    print(f"\nBest trial #{best_trial.number} achieved MCC: {best_trial.value:.4f}")
    print(f"\nTo train with best params: ./{cmd_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for EBI classifier using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Study configuration
    parser.add_argument(
        "--study-name",
        type=str,
        default=f"ebi_study_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Name of the Optuna study (used for resuming)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for storing study (e.g., sqlite:///optuna.db). Default: in-memory",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1 = sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Stop study after N seconds (None = no limit)",
    )
    parser.add_argument(
        "--trial-timeout",
        type=int,
        default=3600,
        help="Timeout for a single trial in seconds (default: 1 hour)",
    )
    
    # Pruning
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable median pruner for early stopping of bad trials",
    )
    parser.add_argument(
        "--pruning-warmup",
        type=int,
        default=5,
        help="Number of trials before pruner starts",
    )
    
    # What to tune
    parser.add_argument(
        "--tune-model",
        action="store_true",
        help="Tune model architecture (model, image_size, drop_path)",
    )
    parser.add_argument(
        "--tune-optimizer",
        action="store_true",
        help="Tune optimizer settings (LR, weight_decay, warmup)",
    )
    parser.add_argument(
        "--tune-augmentation",
        action="store_true",
        help="Tune data augmentation strategy",
    )
    parser.add_argument(
        "--tune-regularization",
        action="store_true",
        help="Tune regularization (MixUp, sampling)",
    )
    parser.add_argument(
        "--tune-training",
        action="store_true",
        help="Tune training schedule (epochs, freeze, batch_size)",
    )
    parser.add_argument(
        "--tune-all",
        action="store_true",
        help="Tune all hyperparameter groups (overrides individual flags)",
    )
    
    # Data paths
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Project root containing EBI/ and NOT_EBI/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optuna_studies"),
        help="Directory to save study results",
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of data loader workers (increase for faster I/O on servers)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=2,
        help="Number of CV folds during tuning (2-3 for speed, 5 for accuracy)",
    )
    
    args = parser.parse_args()
    
    # Set tune groups
    if args.tune_all:
        tune_groups = {
            "tune_model": True,
            "tune_optimizer": True,
            "tune_augmentation": True,
            "tune_regularization": True,
            "tune_training": True,
        }
    else:
        tune_groups = {
            "tune_model": args.tune_model,
            "tune_optimizer": args.tune_optimizer,
            "tune_augmentation": args.tune_augmentation,
            "tune_regularization": args.tune_regularization,
            "tune_training": args.tune_training,
        }
    
    # If no groups specified, tune all by default
    if not any(tune_groups.values()):
        print("No hyperparameter groups specified, tuning all groups by default")
        tune_groups = {k: True for k in tune_groups}
    
    # Create output directory
    study_dir = args.output_dir / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = study_dir / "config.json"
    with config_file.open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    print("Starting Optuna hyperparameter optimization")
    print(f"Study name: {args.study_name}")
    print(f"Study directory: {study_dir}")
    print(f"Tuning groups: {[k for k, v in tune_groups.items() if v]}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    
    # Create sampler and pruner
    sampler = TPESampler(seed=args.seed)
    
    pruner = None
    if args.pruning:
        pruner = MedianPruner(
            n_startup_trials=args.pruning_warmup,
            n_warmup_steps=0,
        )
        print(f"Pruning enabled (warmup: {args.pruning_warmup} trials)")
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximize MCC
        load_if_exists=True,
    )
    
    # Run optimization
    print(f"\n{'='*80}")
    print("Starting optimization...")
    print(f"{'='*80}\n")
    
    study.optimize(
        lambda trial: objective(trial, args, tune_groups),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("Optimization completed!")
    print(f"{'='*80}")
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if len(study.trials) > 0:
        print("\nBest trial:")
        best_trial = study.best_trial
        print(f"  Trial number: {best_trial.number}")
        print(f"  Value (MCC): {best_trial.value:.4f}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save results
        save_best_params(study, study_dir)
        
        # Generate visualizations
        create_visualizations(study, study_dir)
        
        # Save full study statistics
        stats_file = study_dir / "study_statistics.json"
        with stats_file.open("w") as f:
            json.dump(
                {
                    "n_trials": len(study.trials),
                    "best_value": study.best_value,
                    "best_trial": best_trial.number,
                    "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    "complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    "failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                },
                f,
                indent=2,
            )
    
    print(f"\nAll results saved to: {study_dir}")


if __name__ == "__main__":
    main()
