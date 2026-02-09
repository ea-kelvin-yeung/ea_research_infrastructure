"""
Compare: Side-by-side comparison of two backtest runs.
Enables comparing signal versions or different configurations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlflow


@dataclass
class CompareResult:
    """Result from comparing two runs."""
    run_a: Dict  # Run A metadata and metrics
    run_b: Dict  # Run B metadata and metrics
    metrics_diff: pd.DataFrame  # Metrics comparison table
    daily_a: Optional[pd.DataFrame] = None  # Run A daily series
    daily_b: Optional[pd.DataFrame] = None  # Run B daily series


def load_run_data(run_id: str) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Load run metadata and daily returns from MLflow.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Tuple of (run_info dict, daily DataFrame or None)
    """
    client = mlflow.tracking.MlflowClient()
    
    # Get run info
    run = client.get_run(run_id)
    
    run_info = {
        'run_id': run_id,
        'signal_name': run.data.tags.get('signal_name', 'unknown'),
        'snapshot_id': run.data.tags.get('snapshot_id', 'unknown'),
        'git_sha': run.data.tags.get('git_sha', 'unknown'),
        'signal_hash': run.data.tags.get('signal_hash', 'unknown'),
        'start_time': run.info.start_time,
    }
    
    # Extract metrics
    for key, value in run.data.metrics.items():
        run_info[f'metric_{key}'] = value
    
    # Try to load daily data
    daily_df = None
    try:
        artifacts = client.list_artifacts(run_id)
        daily_artifact = next((a.path for a in artifacts if 'daily' in a.path and a.path.endswith('.parquet')), None)
        if daily_artifact:
            local_path = client.download_artifacts(run_id, daily_artifact)
            daily_df = pd.read_parquet(local_path)
    except Exception as e:
        print(f"Could not load daily data for {run_id}: {e}")
    
    return run_info, daily_df


def compare_runs(run_a_id: str, run_b_id: str) -> CompareResult:
    """
    Compare two MLflow runs.
    
    Args:
        run_a_id: First run ID
        run_b_id: Second run ID
        
    Returns:
        CompareResult with metrics diff and overlay data
    """
    # Load both runs
    run_a, daily_a = load_run_data(run_a_id)
    run_b, daily_b = load_run_data(run_b_id)
    
    # Build metrics comparison table
    metrics_diff = _build_metrics_diff(run_a, run_b)
    
    return CompareResult(
        run_a=run_a,
        run_b=run_b,
        metrics_diff=metrics_diff,
        daily_a=daily_a,
        daily_b=daily_b,
    )


def _build_metrics_diff(run_a: Dict, run_b: Dict) -> pd.DataFrame:
    """Build a comparison table of metrics between two runs."""
    # Find common metrics
    a_metrics = {k.replace('metric_', ''): v for k, v in run_a.items() if k.startswith('metric_')}
    b_metrics = {k.replace('metric_', ''): v for k, v in run_b.items() if k.startswith('metric_')}
    
    all_metrics = set(a_metrics.keys()) | set(b_metrics.keys())
    
    # Filter to key metrics (exclude per-config metrics for cleaner display)
    key_metrics = ['best_sharpe']
    # Add Sharpe metrics for common configs
    for m in all_metrics:
        if m.startswith('sharpe_') and m in a_metrics and m in b_metrics:
            key_metrics.append(m)
    
    rows = []
    for metric in sorted(key_metrics):
        a_val = a_metrics.get(metric)
        b_val = b_metrics.get(metric)
        
        if a_val is not None and b_val is not None:
            diff = b_val - a_val
            pct_diff = (diff / abs(a_val) * 100) if a_val != 0 else 0
            
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Run A': a_val,
                'Run B': b_val,
                'Diff': diff,
                'Diff %': pct_diff,
                'Better': 'B' if diff > 0 else ('A' if diff < 0 else 'Same'),
            })
        elif a_val is not None:
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Run A': a_val,
                'Run B': None,
                'Diff': None,
                'Diff %': None,
                'Better': 'A only',
            })
        elif b_val is not None:
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Run A': None,
                'Run B': b_val,
                'Diff': None,
                'Diff %': None,
                'Better': 'B only',
            })
    
    return pd.DataFrame(rows)


def get_overlay_data(daily_a: pd.DataFrame, daily_b: pd.DataFrame, 
                     label_a: str = 'Run A', label_b: str = 'Run B') -> pd.DataFrame:
    """
    Combine daily data from two runs for overlay plotting.
    
    Args:
        daily_a: Daily DataFrame from Run A
        daily_b: Daily DataFrame from Run B
        label_a: Label for Run A
        label_b: Label for Run B
        
    Returns:
        Combined DataFrame with 'run' column for coloring
    """
    if daily_a is None and daily_b is None:
        return pd.DataFrame()
    
    dfs = []
    
    if daily_a is not None and 'cumret' in daily_a.columns:
        # If there are multiple configs, use the first signal config
        if 'config' in daily_a.columns and 'type' in daily_a.columns:
            signal_data = daily_a[daily_a['type'] == 'signal']
            if len(signal_data) > 0:
                first_config = signal_data['config'].iloc[0]
                subset = signal_data[signal_data['config'] == first_config][['date', 'cumret']].copy()
            else:
                subset = daily_a[['date', 'cumret']].drop_duplicates('date')
        else:
            subset = daily_a[['date', 'cumret']].copy()
        
        subset['run'] = label_a
        dfs.append(subset)
    
    if daily_b is not None and 'cumret' in daily_b.columns:
        if 'config' in daily_b.columns and 'type' in daily_b.columns:
            signal_data = daily_b[daily_b['type'] == 'signal']
            if len(signal_data) > 0:
                first_config = signal_data['config'].iloc[0]
                subset = signal_data[signal_data['config'] == first_config][['date', 'cumret']].copy()
            else:
                subset = daily_b[['date', 'cumret']].drop_duplicates('date')
        else:
            subset = daily_b[['date', 'cumret']].copy()
        
        subset['run'] = label_b
        dfs.append(subset)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compute_cumret_diff(daily_a: pd.DataFrame, daily_b: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in cumulative returns between two runs.
    Useful for seeing when one run outperformed the other.
    
    Returns DataFrame with date and cumret_diff columns.
    """
    if daily_a is None or daily_b is None:
        return pd.DataFrame()
    
    if 'cumret' not in daily_a.columns or 'cumret' not in daily_b.columns:
        return pd.DataFrame()
    
    # Get the first config from each
    def get_first_config_cumret(df):
        if 'config' in df.columns and 'type' in df.columns:
            signal_data = df[df['type'] == 'signal']
            if len(signal_data) > 0:
                first_config = signal_data['config'].iloc[0]
                return signal_data[signal_data['config'] == first_config][['date', 'cumret']]
        return df[['date', 'cumret']].drop_duplicates('date')
    
    a = get_first_config_cumret(daily_a)
    b = get_first_config_cumret(daily_b)
    
    # Merge on date
    merged = a.merge(b, on='date', suffixes=('_a', '_b'), how='inner')
    merged['cumret_diff'] = merged['cumret_b'] - merged['cumret_a']
    
    return merged[['date', 'cumret_diff']]
