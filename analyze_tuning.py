"""Visualization and analysis of Optuna hyperparameter tuning results.

This script provides comprehensive analysis of completed Optuna studies:
- Interactive visualizations (HTML)
- Statistical summaries
- Best trial analysis
- Hyperparameter importance rankings
- Trial comparison tables

Usage Examples
--------------
# Analyze the most recent study
python analyze_tuning.py

# Analyze a specific study
python analyze_tuning.py --study-dir optuna_studies/ebi_study_20260112_143022

# Generate report with custom output
python analyze_tuning.py --study-dir optuna_studies/my_study --output results.html

# Compare top N trials
python analyze_tuning.py --top-n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.visualization import (
        plot_edf,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
        plot_timeline,
    )
except ImportError:
    print("Error: Optuna not installed. Run: uv pip install optuna", file=sys.stderr)
    sys.exit(1)

# Import plotly only if needed (avoid unused import warning)
# These are imported inside functions where they're actually used


def load_study(study_dir: Path, study_name: Optional[str] = None) -> optuna.Study:
    """Load an Optuna study from directory.
    
    Parameters
    ----------
    study_dir : Path
        Directory containing study results.
    study_name : str, optional
        Name of the study. If None, uses directory name.
    
    Returns
    -------
    study : optuna.Study
        Loaded study object.
    """
    if study_name is None:
        study_name = study_dir.name
    
    # Check for SQLite database
    db_path = study_dir / "optuna.db"
    if db_path.exists():
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded study from database: {db_path}")
    else:
        # Try to reconstruct from trial files (if available)
        print(f"Warning: No database found at {db_path}")
        print("Attempting to reconstruct from trial files...")
        
        # Create in-memory study and load trials from JSON files
        study = optuna.create_study(direction="maximize")
        
        trial_dirs = sorted(study_dir.glob("trial_*"))
        if not trial_dirs:
            raise FileNotFoundError(f"No trial directories found in {study_dir}")
        
        print(f"Found {len(trial_dirs)} trial directories")
        # Note: Full reconstruction from files not implemented
        # This is a simplified loader - for full functionality use database storage
        
    return study


def create_trial_comparison_table(study: optuna.Study, top_n: int = 10) -> pd.DataFrame:
    """Create a comparison table of top N trials.
    
    Parameters
    ----------
    study : optuna.Study
        Optuna study object.
    top_n : int
        Number of top trials to include.
    
    Returns
    -------
    df : pd.DataFrame
        Comparison table.
    """
    # Get completed trials sorted by value
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    # Build table
    rows = []
    for rank, trial in enumerate(completed_trials[:top_n], start=1):
        row = {
            "Rank": rank,
            "Trial": trial.number,
            "MCC": f"{trial.value:.4f}",
        }
        
        # Add all parameters
        for key, value in trial.params.items():
            if isinstance(value, float):
                row[key] = f"{value:.4e}" if value < 0.001 else f"{value:.4f}"
            else:
                row[key] = str(value)
        
        # Add user attributes if available
        if "mean_mcc" in trial.user_attrs:
            row["Mean MCC"] = f"{trial.user_attrs['mean_mcc']:.4f}"
        if "std_mcc" in trial.user_attrs:
            row["Std MCC"] = f"{trial.user_attrs['std_mcc']:.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_report(
    study: optuna.Study,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Generate a comprehensive HTML report.
    
    Parameters
    ----------
    study : optuna.Study
        Optuna study object.
    output_path : Path
        Path to save HTML report.
    top_n : int
        Number of top trials to include in comparison.
    """
    # Create visualizations
    figs = {}
    
    # Optimization history
    try:
        figs["history"] = plot_optimization_history(study)
    except Exception as e:
        print(f"Warning: Could not create optimization history: {e}")
    
    # Parallel coordinate
    try:
        figs["parallel"] = plot_parallel_coordinate(study)
    except Exception as e:
        print(f"Warning: Could not create parallel coordinate plot: {e}")
    
    # Parameter importance
    try:
        figs["importance"] = plot_param_importances(study)
    except Exception as e:
        print(f"Warning: Could not create parameter importance plot: {e}")
    
    # Slice plot
    try:
        figs["slice"] = plot_slice(study)
    except Exception as e:
        print(f"Warning: Could not create slice plot: {e}")
    
    # EDF (empirical distribution function)
    try:
        figs["edf"] = plot_edf(study)
    except Exception as e:
        print(f"Warning: Could not create EDF plot: {e}")
    
    # Timeline
    try:
        figs["timeline"] = plot_timeline(study)
    except Exception as e:
        print(f"Warning: Could not create timeline: {e}")
    
    # Trial comparison table
    try:
        comparison_df = create_trial_comparison_table(study, top_n=top_n)
        comparison_html = comparison_df.to_html(index=False, classes="table table-striped")
    except Exception as e:
        print(f"Warning: Could not create comparison table: {e}")
        comparison_html = "<p>Error creating comparison table</p>"
    
    # Statistics
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    values = [t.value for t in completed]
    
    stats = {
        "Total trials": len(study.trials),
        "Completed": len(completed),
        "Pruned": len(pruned),
        "Failed": len(failed),
        "Best value": f"{study.best_value:.4f}" if completed else "N/A",
        "Best trial": study.best_trial.number if completed else "N/A",
        "Mean value": f"{np.mean(values):.4f}" if values else "N/A",
        "Std value": f"{np.std(values):.4f}" if values else "N/A",
        "Median value": f"{np.median(values):.4f}" if values else "N/A",
    }
    
    # Build HTML report
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Optuna Study Report - {study.study_name}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <style>
        body {{ padding: 20px; }}
        .plot-container {{ margin: 30px 0; }}
        .stat-card {{ margin: 10px 0; }}
        h1, h2 {{ margin-top: 30px; }}
        .table {{ font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>Optuna Hyperparameter Tuning Report</h1>
        <p class="lead">Study: {study.study_name}</p>
        
        <h2>Study Statistics</h2>
        <div class="row">
"""
    
    for key, value in stats.items():
        html += f"""
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">{key}</h6>
                        <p class="card-text h4">{value}</p>
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>Best Trial Parameters</h2>
        <div class="card">
            <div class="card-body">
"""
    
    if completed:
        best = study.best_trial
        html += f"<p><strong>Trial #{best.number}</strong> - MCC: {best.value:.4f}</p><ul>"
        for key, value in best.params.items():
            html += f"<li><strong>{key}</strong>: {value}</li>"
        html += "</ul>"
    else:
        html += "<p>No completed trials</p>"
    
    html += """
            </div>
        </div>
        
        <h2>Top Trials Comparison</h2>
        <div class="table-responsive">
"""
    
    html += comparison_html
    
    html += """
        </div>
        
        <h2>Visualizations</h2>
"""
    
    for name, fig in figs.items():
        html += f"""
        <div class="plot-container">
            <h3>{name.replace('_', ' ').title()}</h3>
            {fig.to_html(include_plotlyjs='cdn', div_id=f'{name}_plot')}
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    output_path.write_text(html)
    print(f"Report saved to: {output_path}")


def print_summary(study: optuna.Study, top_n: int = 5) -> None:
    """Print a text summary to console.
    
    Parameters
    ----------
    study : optuna.Study
        Optuna study object.
    top_n : int
        Number of top trials to show.
    """
    print(f"\n{'='*80}")
    print(f"Study Summary: {study.study_name}")
    print(f"{'='*80}\n")
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"Total trials: {len(study.trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Failed: {len(failed)}")
    
    if completed:
        values = [t.value for t in completed]
        print("\nPerformance (MCC):")
        print(f"  Best: {study.best_value:.4f} (trial #{study.best_trial.number})")
        print(f"  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        
        print("\nBest Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        print(f"\nTop {top_n} Trials:")
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]
        for rank, trial in enumerate(sorted_trials, start=1):
            print(f"  #{rank}: Trial {trial.number} - MCC: {trial.value:.4f}")
    else:
        print("\nNo completed trials found.")
    
    print(f"\n{'='*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Optuna hyperparameter tuning results"
    )
    
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="Path to study directory. If not provided, uses most recent in optuna_studies/",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the study (default: directory name)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for HTML report (default: <study_dir>/report.html)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top trials to include in comparison (default: 10)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating HTML report (only print summary)",
    )
    
    args = parser.parse_args()
    
    # Find study directory
    if args.study_dir is None:
        optuna_studies = Path("optuna_studies")
        if not optuna_studies.exists():
            print("Error: No optuna_studies directory found", file=sys.stderr)
            sys.exit(1)
        
        study_dirs = sorted(optuna_studies.glob("ebi_study_*"))
        if not study_dirs:
            print("Error: No studies found in optuna_studies/", file=sys.stderr)
            sys.exit(1)
        
        args.study_dir = study_dirs[-1]  # Most recent
        print(f"Using most recent study: {args.study_dir}")
    
    if not args.study_dir.exists():
        print(f"Error: Study directory not found: {args.study_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        args.output = args.study_dir / "report.html"
    
    # Load study
    try:
        study = load_study(args.study_dir, args.study_name)
    except Exception as e:
        print(f"Error loading study: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print_summary(study, top_n=args.top_n)
    
    # Generate report
    if not args.no_report:
        try:
            generate_report(study, args.output, top_n=args.top_n)
        except Exception as e:
            print(f"Error generating report: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
