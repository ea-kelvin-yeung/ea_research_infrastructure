"""
MLflow Tracking: Log suite runs for reproducibility.

Optimized for speed with batched metrics/artifacts and fast hashing.
"""

import logging
import mlflow

# Suppress alembic noise AFTER importing mlflow (alembic configures its handlers during import)
for _logger_name in ['alembic', 'alembic.runtime', 'alembic.runtime.plugins', 
                     'alembic.runtime.migration', 'mlflow.tracking']:
    _logger = logging.getLogger(_logger_name)
    _logger.handlers = []
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False

import subprocess
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from .suite import SuiteResult, get_best_config
from .tearsheet import compute_verdict, compute_composite_score, _extract_cap_breakdown, _extract_year_breakdown


# Cache git SHA (doesn't change during session)
_GIT_SHA_CACHE = None


def get_git_sha() -> str:
    """Get current git commit SHA (cached)."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True,
            timeout=2,
        )
        _GIT_SHA_CACHE = result.stdout.strip()
        return _GIT_SHA_CACHE
    except:
        _GIT_SHA_CACHE = 'unknown'
        return _GIT_SHA_CACHE


def compute_signal_hash(signal_df: pd.DataFrame) -> str:
    """
    Compute fast hash of signal DataFrame for reproducibility.
    
    Uses sampling + numeric hash for speed on large DataFrames.
    """
    n = len(signal_df)
    
    # For small DataFrames, hash everything
    if n <= 10000:
        # Use efficient binary representation instead of CSV
        content = pd.util.hash_pandas_object(signal_df).values.tobytes()
        return hashlib.md5(content).hexdigest()[:12]
    
    # For large DataFrames, use deterministic sampling + stats
    np.random.seed(42)
    sample_idx = np.random.choice(n, min(5000, n), replace=False)
    sample = signal_df.iloc[sorted(sample_idx)]
    
    # Hash: sample + shape + column stats
    hash_parts = [
        pd.util.hash_pandas_object(sample).values.tobytes(),
        str(signal_df.shape).encode(),
        str(signal_df.columns.tolist()).encode(),
    ]
    
    # Add numeric column stats for better uniqueness
    for col in signal_df.select_dtypes(include=[np.number]).columns[:5]:
        stats = f"{col}:{signal_df[col].mean():.6f}:{signal_df[col].std():.6f}"
        hash_parts.append(stats.encode())
    
    content = b''.join(hash_parts)
    return hashlib.md5(content).hexdigest()[:12]


def log_run(
    suite_result: SuiteResult,
    signal_name: str,
    catalog: dict,
    tearsheet_path: Optional[str] = None,
    experiment_name: str = 'backtest-poc',
    author: Optional[str] = None,
    signal_df: Optional[pd.DataFrame] = None,
    suite_options: Optional[dict] = None,
) -> str:
    """
    Log a suite run to MLflow.
    
    Optimized for speed:
    - Batched metrics logging
    - Batched artifact upload (single directory)
    - Fast signal hashing
    - Parallel file preparation
    """
    mlflow.set_experiment(experiment_name)
    
    # Get git SHA (cached) and compute signal hash in parallel
    git_sha = get_git_sha()
    
    # Fast signal hash
    signal_hash = compute_signal_hash(signal_df) if signal_df is not None else 'unknown'
    
    # Create temp artifact directory for batch upload
    artifact_dir = Path('artifacts') / f"run_{signal_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare all artifacts in parallel
        _prepare_artifacts_parallel(
            artifact_dir, suite_result, signal_name, catalog, 
            git_sha, signal_hash, signal_df, tearsheet_path
        )
        
        with mlflow.start_run(run_name=f"{signal_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Tags (single batch call)
            mlflow.set_tags({
                'signal_name': signal_name,
                'snapshot_id': catalog.get('snapshot_id', 'unknown'),
                'data_fingerprint': catalog.get('fingerprint', 'unknown'),
                'git_sha': git_sha,
                'signal_hash': signal_hash,
                'author': author or 'unknown',
            })
            
            # Params (single batch call)
            params = {}
            if suite_options:
                params.update({
                    'suite_lags': str(suite_options.get('lags', [])),
                    'suite_residualize_opts': str(suite_options.get('residualize_opts', [])),
                    'suite_include_baselines': suite_options.get('include_baselines', False),
                    'suite_start_date': str(suite_options.get('start_date', '')),
                    'suite_end_date': str(suite_options.get('end_date', '')),
                    'suite_universe_filter': suite_options.get('universe_filter', 'All Securities'),
                    'suite_signal_path': suite_options.get('signal_path', ''),
                })
            if suite_result.results:
                first_result = list(suite_result.results.values())[0]
                params.update(first_result.config)
            if params:
                mlflow.log_params(params)
            
            # Metrics - batch all into single call
            metrics = {}
            best_sharpe = None
            for config_key, result in suite_result.results.items():
                metrics[f'sharpe_{config_key}'] = result.sharpe
                metrics[f'ann_ret_{config_key}'] = result.annual_return
                metrics[f'max_dd_{config_key}'] = result.max_drawdown
                metrics[f'turnover_{config_key}'] = result.turnover
                if best_sharpe is None or result.sharpe > best_sharpe:
                    best_sharpe = result.sharpe
            
            if best_sharpe is not None:
                metrics['best_sharpe'] = best_sharpe
            
            for name, result in suite_result.baselines.items():
                metrics[f'sharpe_baseline_{name}'] = result.sharpe
            
            for _, row in suite_result.correlations.iterrows():
                baseline = row['baseline']
                if pd.notna(row['signal_corr']):
                    metrics[f'signal_corr_{baseline}'] = row['signal_corr']
                if pd.notna(row['pnl_corr']):
                    metrics[f'pnl_corr_{baseline}'] = row['pnl_corr']
            
            best_key = get_best_config(suite_result, 'sharpe')
            if best_key and best_key in suite_result.results:
                best_result = suite_result.results[best_key]
                metrics['best_ann_ret'] = best_result.annual_return
                metrics['best_max_dd'] = best_result.max_drawdown
                metrics['best_turnover'] = best_result.turnover
            
            # Filter out NaN values and log all metrics in one call
            metrics = {k: v for k, v in metrics.items() if pd.notna(v)}
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log all artifacts in single batch call
            mlflow.log_artifacts(str(artifact_dir))
            
            run_id = mlflow.active_run().info.run_id
        
        print(f"Logged to MLflow: run_id={run_id}")
        return run_id
    
    finally:
        # Cleanup temp artifact directory
        shutil.rmtree(artifact_dir, ignore_errors=True)


def _prepare_artifacts_parallel(
    artifact_dir: Path,
    suite_result: SuiteResult,
    signal_name: str,
    catalog: dict,
    git_sha: str,
    signal_hash: str,
    signal_df: Optional[pd.DataFrame],
    tearsheet_path: Optional[str],
):
    """Prepare all artifact files in parallel for fast batch upload."""
    
    def write_config():
        all_configs = {key: res.config for key, res in suite_result.results.items()}
        reproducibility_info = {
            'signal_name': signal_name,
            'snapshot_id': catalog.get('snapshot_id', 'unknown'),
            'git_sha': git_sha,
            'signal_hash': signal_hash,
            'configs': all_configs,
        }
        with open(artifact_dir / f"{signal_name}_config.json", 'w') as f:
            json.dump(reproducibility_info, f, indent=2, default=str)
    
    def write_summary():
        suite_result.summary.to_csv(artifact_dir / f"{signal_name}_summary.csv", index=False)
    
    def write_daily():
        daily_data = []
        for config_key, result in suite_result.results.items():
            if 'cumret' in result.daily.columns:
                cols = ['date', 'cumret']
                if 'ret' in result.daily.columns:
                    cols.append('ret')
                if 'drawdown' in result.daily.columns:
                    cols.append('drawdown')
                df = result.daily[cols].copy()
                df['config'] = config_key
                df['type'] = 'signal'
                daily_data.append(df)
        
        for name, result in suite_result.baselines.items():
            if 'cumret' in result.daily.columns:
                cols = ['date', 'cumret']
                if 'ret' in result.daily.columns:
                    cols.append('ret')
                if 'drawdown' in result.daily.columns:
                    cols.append('drawdown')
                df = result.daily[cols].copy()
                df['config'] = name
                df['type'] = 'baseline'
                daily_data.append(df)
        
        if daily_data:
            combined_daily = pd.concat(daily_data, ignore_index=True)
            combined_daily.to_parquet(artifact_dir / f"{signal_name}_daily.parquet")
    
    def write_fractile():
        best_key = get_best_config(suite_result, 'sharpe')
        if best_key and best_key in suite_result.results:
            best_result = suite_result.results[best_key]
            if best_result.fractile is not None and len(best_result.fractile) > 0:
                best_result.fractile.to_parquet(artifact_dir / f"{signal_name}_fractile.parquet", index=False)
    
    def write_ic():
        if suite_result.ic_series is not None and len(suite_result.ic_series) > 0:
            suite_result.ic_series.to_parquet(artifact_dir / f"{signal_name}_ic_series.parquet", index=False)
        if suite_result.ic_stats is not None:
            with open(artifact_dir / f"{signal_name}_ic_stats.json", 'w') as f:
                json.dump(suite_result.ic_stats, f, indent=2, default=str)
    
    def write_factors():
        if suite_result.factor_exposures is not None and len(suite_result.factor_exposures) > 0:
            suite_result.factor_exposures.to_parquet(artifact_dir / f"{signal_name}_factor_exposures.parquet", index=False)
    
    def write_correlations():
        if len(suite_result.correlations) > 0:
            suite_result.correlations.to_parquet(artifact_dir / f"{signal_name}_correlations.parquet", index=False)
    
    def write_coverage():
        if suite_result.coverage:
            with open(artifact_dir / f"{signal_name}_coverage.json", 'w') as f:
                json.dump(suite_result.coverage, f, indent=2, default=str)
        if signal_df is not None and 'date_sig' in signal_df.columns:
            coverage_ts = signal_df.groupby('date_sig').size().reset_index(name='count')
            coverage_ts.columns = ['date', 'count']
            coverage_ts.to_parquet(artifact_dir / f"{signal_name}_coverage_series.parquet", index=False)
    
    def write_verdict():
        verdict = compute_verdict(suite_result)
        with open(artifact_dir / f"{signal_name}_verdict.json", 'w') as f:
            json.dump(verdict, f, indent=2, default=str)
    
    def write_composite():
        composite_score = compute_composite_score(suite_result)
        with open(artifact_dir / f"{signal_name}_composite_score.json", 'w') as f:
            json.dump(composite_score, f, indent=2, default=str)
    
    def write_breakdowns():
        best_key = get_best_config(suite_result, 'sharpe')
        if best_key:
            cap_data = _extract_cap_breakdown(suite_result, best_key)
            if len(cap_data) > 0:
                cap_data.to_csv(artifact_dir / f"{signal_name}_cap_breakdown.csv", index=False)
            year_data = _extract_year_breakdown(suite_result, best_key)
            if len(year_data) > 0:
                year_data.to_csv(artifact_dir / f"{signal_name}_year_breakdown.csv", index=False)
    
    def copy_tearsheet():
        if tearsheet_path and Path(tearsheet_path).exists():
            shutil.copy(tearsheet_path, artifact_dir / Path(tearsheet_path).name)
    
    # Run all artifact preparation in parallel
    tasks = [
        write_config, write_summary, write_daily, write_fractile,
        write_ic, write_factors, write_correlations, write_coverage,
        write_verdict, write_composite, write_breakdowns, copy_tearsheet,
    ]
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Warning: artifact preparation failed: {e}")


def get_run_history(experiment_name: str = 'backtest-poc', max_results: int = 100) -> list:
    """Get recent runs from MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=['start_time DESC'],
        )
        return runs.to_dict('records')
    except:
        return []


def delete_runs(run_ids: list, experiment_name: str = 'backtest-poc') -> dict:
    """
    Delete multiple MLflow runs.
    
    Args:
        run_ids: List of run IDs to delete
        experiment_name: MLflow experiment name
        
    Returns:
        dict with 'deleted' (list of successfully deleted IDs) and 'failed' (list of failed IDs)
    """
    client = mlflow.tracking.MlflowClient()
    deleted = []
    failed = []
    
    for run_id in run_ids:
        try:
            client.delete_run(run_id)
            deleted.append(run_id)
        except Exception as e:
            failed.append({'run_id': run_id, 'error': str(e)})
    
    return {'deleted': deleted, 'failed': failed}
