"""
Compare: Side-by-side comparison of two backtest runs.
Enables comparing signal versions or different configurations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import mlflow


@dataclass
class CompareResult:
    """Result from comparing two runs."""
    run_a: Dict  # Run A metadata and metrics
    run_b: Dict  # Run B metadata and metrics
    metrics_diff: pd.DataFrame  # Metrics comparison table
    daily_a: Optional[pd.DataFrame] = None  # Run A daily series
    daily_b: Optional[pd.DataFrame] = None  # Run B daily series


# Cache for loaded run data (avoids re-downloading artifacts)
_run_data_cache: Dict[str, Tuple[Dict, Optional[pd.DataFrame]]] = {}


def clear_run_cache():
    """Clear the run data cache."""
    _run_data_cache.clear()


def load_run_data(run_id: str, use_cache: bool = True) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Load run metadata and daily returns from MLflow.
    
    Args:
        run_id: MLflow run ID
        use_cache: Whether to use cached data (default True)
        
    Returns:
        Tuple of (run_info dict, daily DataFrame or None)
    """
    # Check cache first
    if use_cache and run_id in _run_data_cache:
        return _run_data_cache[run_id]
    
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
    
    # Cache the result
    result = (run_info, daily_df)
    _run_data_cache[run_id] = result
    
    return result


def compare_runs(run_a_id: str, run_b_id: str) -> CompareResult:
    """
    Compare two MLflow runs.
    
    Args:
        run_a_id: First run ID
        run_b_id: Second run ID
        
    Returns:
        CompareResult with metrics diff and overlay data
    """
    # Load both runs in parallel for faster loading
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(load_run_data, run_a_id)
        future_b = executor.submit(load_run_data, run_b_id)
        run_a, daily_a = future_a.result()
        run_b, daily_b = future_b.result()
    
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
    
    # Headline metrics to always include (in order)
    headline_metrics = [
        'best_sharpe', 'best_ann_ret', 'best_max_dd', 'best_turnover',
    ]
    
    # Start with headline metrics
    key_metrics = [m for m in headline_metrics if m in a_metrics or m in b_metrics]
    
    # Add Sharpe, ann_ret, turnover metrics for common configs
    for m in sorted(all_metrics):
        if m in key_metrics:
            continue
        # Include key per-config metrics
        if any(m.startswith(prefix) for prefix in ['sharpe_lag', 'ann_ret_lag', 'turnover_lag', 'max_dd_lag']):
            if m in a_metrics and m in b_metrics:
                key_metrics.append(m)
    
    # Metrics where lower is better
    lower_is_better = {'best_max_dd', 'best_turnover', 'max_dd', 'turnover'}
    
    rows = []
    for metric in key_metrics:
        a_val = a_metrics.get(metric)
        b_val = b_metrics.get(metric)
        
        # Determine if lower is better for this metric
        is_lower_better = any(lb in metric for lb in lower_is_better)
        
        if a_val is not None and b_val is not None:
            diff = b_val - a_val
            pct_diff = (diff / abs(a_val) * 100) if a_val != 0 else 0
            
            # Determine which is better
            if is_lower_better:
                better = 'B' if diff < 0 else ('A' if diff > 0 else 'Same')
            else:
                better = 'B' if diff > 0 else ('A' if diff < 0 else 'Same')
            
            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Run A': a_val,
                'Run B': b_val,
                'Diff': diff,
                'Diff %': pct_diff,
                'Better': better,
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


def _get_default_config(df: pd.DataFrame) -> Optional[str]:
    """
    Get the default config to use for comparison.
    Prefers 'lag0_residoff' (baseline), falls back to first available config.
    """
    if 'config' not in df.columns:
        return None
    
    configs = df['config'].unique()
    
    # Prefer lag0_residoff (the baseline config)
    if 'lag0_residoff' in configs:
        return 'lag0_residoff'
    
    # Fallback to any lag0 config
    for c in configs:
        if c.startswith('lag0'):
            return c
    
    # Last resort: first config
    return configs[0] if len(configs) > 0 else None


def get_overlay_data(daily_a: pd.DataFrame, daily_b: pd.DataFrame, 
                     label_a: str = 'Run A', label_b: str = 'Run B',
                     config: Optional[str] = None) -> pd.DataFrame:
    """
    Combine daily data from two runs for overlay plotting.
    
    Args:
        daily_a: Daily DataFrame from Run A
        daily_b: Daily DataFrame from Run B
        label_a: Label for Run A
        label_b: Label for Run B
        config: Specific config to use (default: 'lag0_residoff' or first available)
        
    Returns:
        Combined DataFrame with 'run' column for coloring
    """
    if daily_a is None and daily_b is None:
        return pd.DataFrame()
    
    dfs = []
    
    if daily_a is not None and 'cumret' in daily_a.columns:
        # If there are multiple configs, use specified or default config
        if 'config' in daily_a.columns and 'type' in daily_a.columns:
            signal_data = daily_a[daily_a['type'] == 'signal']
            if len(signal_data) > 0:
                target_config = config or _get_default_config(signal_data)
                subset = signal_data[signal_data['config'] == target_config][['date', 'cumret']].copy()
                if len(subset) == 0:
                    # Fallback if specified config not found
                    subset = signal_data[['date', 'cumret']].drop_duplicates('date')
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
                target_config = config or _get_default_config(signal_data)
                subset = signal_data[signal_data['config'] == target_config][['date', 'cumret']].copy()
                if len(subset) == 0:
                    subset = signal_data[['date', 'cumret']].drop_duplicates('date')
            else:
                subset = daily_b[['date', 'cumret']].drop_duplicates('date')
        else:
            subset = daily_b[['date', 'cumret']].copy()
        
        subset['run'] = label_b
        dfs.append(subset)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compute_rolling_sharpe(daily_df: pd.DataFrame, window_years: int = 3,
                           config: Optional[str] = None) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio over a specified window.
    
    Args:
        daily_df: Daily DataFrame with 'date' and 'ret' columns
        window_years: Rolling window size in years (default 3)
        config: Specific config to use (default: 'lag0_residoff' or first available)
        
    Returns:
        DataFrame with date and rolling_sharpe columns
    """
    if daily_df is None or 'ret' not in daily_df.columns:
        return pd.DataFrame()
    
    # Get the right config's data
    if 'config' in daily_df.columns and 'type' in daily_df.columns:
        signal_data = daily_df[daily_df['type'] == 'signal']
        if len(signal_data) > 0:
            target_config = config or _get_default_config(signal_data)
            df = signal_data[signal_data['config'] == target_config][['date', 'ret']].copy()
            if len(df) == 0:
                df = signal_data[['date', 'ret']].drop_duplicates('date')
        else:
            df = daily_df[['date', 'ret']].drop_duplicates('date')
    else:
        df = daily_df[['date', 'ret']].copy()
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate rolling window in trading days (approx 252 per year)
    window = window_years * 252
    
    if len(df) < window:
        return pd.DataFrame()
    
    # Calculate rolling Sharpe (annualized)
    sqrt252 = np.sqrt(252)
    rolling_mean = df['ret'].rolling(window=window).mean() * 252
    rolling_std = df['ret'].rolling(window=window).std() * sqrt252
    df['rolling_sharpe'] = rolling_mean / rolling_std
    
    return df[['date', 'rolling_sharpe']].dropna()


def get_rolling_sharpe_overlay(daily_a: pd.DataFrame, daily_b: pd.DataFrame,
                                label_a: str = 'Run A', label_b: str = 'Run B',
                                window_years: int = 3,
                                config: Optional[str] = None) -> pd.DataFrame:
    """
    Compute rolling Sharpe for both runs and combine for overlay plotting.
    
    Args:
        daily_a: Daily DataFrame from Run A
        daily_b: Daily DataFrame from Run B
        label_a: Label for Run A
        label_b: Label for Run B
        window_years: Rolling window size in years
        config: Specific config to use
        
    Returns:
        Combined DataFrame with 'run' column for coloring
    """
    dfs = []
    
    sharpe_a = compute_rolling_sharpe(daily_a, window_years, config)
    if len(sharpe_a) > 0:
        sharpe_a['run'] = label_a
        dfs.append(sharpe_a)
    
    sharpe_b = compute_rolling_sharpe(daily_b, window_years, config)
    if len(sharpe_b) > 0:
        sharpe_b['run'] = label_b
        dfs.append(sharpe_b)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compute_cumret_diff(daily_a: pd.DataFrame, daily_b: pd.DataFrame,
                        config: Optional[str] = None) -> pd.DataFrame:
    """
    Compute the difference in cumulative returns between two runs.
    Useful for seeing when one run outperformed the other.
    
    Args:
        daily_a: Daily DataFrame from Run A
        daily_b: Daily DataFrame from Run B
        config: Specific config to use (default: 'lag0_residoff' or first available)
    
    Returns DataFrame with date and cumret_diff columns.
    """
    if daily_a is None or daily_b is None:
        return pd.DataFrame()
    
    if 'cumret' not in daily_a.columns or 'cumret' not in daily_b.columns:
        return pd.DataFrame()
    
    # Get the default config's cumret
    def get_config_cumret(df, target_config=None):
        if 'config' in df.columns and 'type' in df.columns:
            signal_data = df[df['type'] == 'signal']
            if len(signal_data) > 0:
                cfg = target_config or _get_default_config(signal_data)
                subset = signal_data[signal_data['config'] == cfg][['date', 'cumret']]
                if len(subset) > 0:
                    return subset
        return df[['date', 'cumret']].drop_duplicates('date')
    
    a = get_config_cumret(daily_a, config)
    b = get_config_cumret(daily_b, config)
    
    # Merge on date
    merged = a.merge(b, on='date', suffixes=('_a', '_b'), how='inner')
    merged['cumret_diff'] = merged['cumret_b'] - merged['cumret_a']
    
    return merged[['date', 'cumret_diff']]
