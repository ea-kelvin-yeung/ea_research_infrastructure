"""
MLflow Tracking: Log suite runs for reproducibility.
~100 lines - simple logging, let MLflow handle the rest.
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
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from .suite import SuiteResult, get_best_config
from .tearsheet import compute_verdict, compute_composite_score, _extract_cap_breakdown, _extract_year_breakdown


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return 'unknown'


def compute_signal_hash(signal_df: pd.DataFrame) -> str:
    """Compute MD5 hash of signal DataFrame for reproducibility."""
    # Hash based on sorted values to be order-independent
    sorted_df = signal_df.sort_values(['security_id', 'date_sig']).reset_index(drop=True)
    # Use a stable representation
    content = sorted_df.to_csv(index=False).encode('utf-8')
    return hashlib.md5(content).hexdigest()[:12]


def log_run(
    suite_result: SuiteResult,
    signal_name: str,
    catalog: dict,
    tearsheet_path: Optional[str] = None,
    experiment_name: str = 'backtest-poc',
    author: Optional[str] = None,
    signal_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Log a suite run to MLflow.
    
    Args:
        suite_result: Result from run_suite()
        signal_name: Name of the signal
        catalog: Data catalog (for snapshot_id)
        tearsheet_path: Path to tearsheet HTML (optional)
        experiment_name: MLflow experiment name
        author: Author name (optional)
        signal_df: Original signal DataFrame (for computing hash)
        
    Returns:
        MLflow run ID
    """
    mlflow.set_experiment(experiment_name)
    
    # Compute signal hash for reproducibility
    signal_hash = compute_signal_hash(signal_df) if signal_df is not None else 'unknown'
    
    with mlflow.start_run(run_name=f"{signal_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Tags (includes all reproducibility info)
        mlflow.set_tags({
            'signal_name': signal_name,
            'snapshot_id': catalog.get('snapshot_id', 'unknown'),
            'git_sha': get_git_sha(),
            'signal_hash': signal_hash,
            'author': author or 'unknown',
        })
        
        # Params (from first config in results)
        if suite_result.results:
            first_result = list(suite_result.results.values())[0]
            mlflow.log_params(first_result.config)
        
        # Metrics - log for each config
        best_sharpe = None
        for config_key, result in suite_result.results.items():
            mlflow.log_metric(f'sharpe_{config_key}', result.sharpe)
            mlflow.log_metric(f'ann_ret_{config_key}', result.annual_return)
            mlflow.log_metric(f'max_dd_{config_key}', result.max_drawdown)
            mlflow.log_metric(f'turnover_{config_key}', result.turnover)
            # Track best sharpe
            if best_sharpe is None or result.sharpe > best_sharpe:
                best_sharpe = result.sharpe
        
        # Log best sharpe across all configs
        if best_sharpe is not None:
            mlflow.log_metric('best_sharpe', best_sharpe)
        
        # Log baseline metrics
        for name, result in suite_result.baselines.items():
            mlflow.log_metric(f'sharpe_baseline_{name}', result.sharpe)
        
        # Log correlations
        for _, row in suite_result.correlations.iterrows():
            baseline = row['baseline']
            mlflow.log_metric(f'signal_corr_{baseline}', row['signal_corr'])
            mlflow.log_metric(f'pnl_corr_{baseline}', row['pnl_corr'])
        
        # Artifacts
        if tearsheet_path and Path(tearsheet_path).exists():
            mlflow.log_artifact(tearsheet_path)
        
        # Log full config as JSON for reproducibility
        config_path = Path('artifacts') / f"{signal_name}_config.json"
        config_path.parent.mkdir(exist_ok=True)
        all_configs = {key: res.config for key, res in suite_result.results.items()}
        reproducibility_info = {
            'signal_name': signal_name,
            'snapshot_id': catalog.get('snapshot_id', 'unknown'),
            'git_sha': get_git_sha(),
            'signal_hash': signal_hash,
            'configs': all_configs,
        }
        with open(config_path, 'w') as f:
            json.dump(reproducibility_info, f, indent=2, default=str)
        mlflow.log_artifact(str(config_path))
        
        # Log summary as CSV
        summary_path = Path('artifacts') / f"{signal_name}_summary.csv"
        suite_result.summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        
        # Log daily series for all configs + baselines (combined)
        daily_data = []
        for config_key, result in suite_result.results.items():
            if 'cumret' in result.daily.columns:
                df = result.daily[['date', 'cumret']].copy()
                df['config'] = config_key
                df['type'] = 'signal'
                daily_data.append(df)
        
        # Include baseline daily data
        for name, result in suite_result.baselines.items():
            if 'cumret' in result.daily.columns:
                df = result.daily[['date', 'cumret']].copy()
                df['config'] = name
                df['type'] = 'baseline'
                daily_data.append(df)
        
        if daily_data:
            combined_daily = pd.concat(daily_data, ignore_index=True)
            daily_path = Path('artifacts') / f"{signal_name}_daily.parquet"
            combined_daily.to_parquet(daily_path)
            mlflow.log_artifact(str(daily_path))
        
        # Log IC series if available
        if suite_result.ic_series is not None and len(suite_result.ic_series) > 0:
            ic_path = Path('artifacts') / f"{signal_name}_ic_series.parquet"
            suite_result.ic_series.to_parquet(ic_path, index=False)
            mlflow.log_artifact(str(ic_path))
        
        # Log IC stats as JSON
        if suite_result.ic_stats is not None:
            ic_stats_path = Path('artifacts') / f"{signal_name}_ic_stats.json"
            with open(ic_stats_path, 'w') as f:
                json.dump(suite_result.ic_stats, f, indent=2, default=str)
            mlflow.log_artifact(str(ic_stats_path))
        
        # Log factor exposures if available
        if suite_result.factor_exposures is not None and len(suite_result.factor_exposures) > 0:
            factor_path = Path('artifacts') / f"{signal_name}_factor_exposures.parquet"
            suite_result.factor_exposures.to_parquet(factor_path, index=False)
            mlflow.log_artifact(str(factor_path))
        
        # Log correlations
        if len(suite_result.correlations) > 0:
            corr_path = Path('artifacts') / f"{signal_name}_correlations.parquet"
            suite_result.correlations.to_parquet(corr_path, index=False)
            mlflow.log_artifact(str(corr_path))
        
        # Log coverage as JSON
        if suite_result.coverage:
            coverage_path = Path('artifacts') / f"{signal_name}_coverage.json"
            with open(coverage_path, 'w') as f:
                json.dump(suite_result.coverage, f, indent=2, default=str)
            mlflow.log_artifact(str(coverage_path))
        
        # Log coverage time series for chart
        if signal_df is not None and 'date_sig' in signal_df.columns:
            coverage_ts = signal_df.groupby('date_sig').size().reset_index(name='count')
            coverage_ts.columns = ['date', 'count']
            coverage_ts_path = Path('artifacts') / f"{signal_name}_coverage_series.parquet"
            coverage_ts.to_parquet(coverage_ts_path, index=False)
            mlflow.log_artifact(str(coverage_ts_path))
        
        # Log verdict as JSON
        verdict = compute_verdict(suite_result)
        verdict_path = Path('artifacts') / f"{signal_name}_verdict.json"
        with open(verdict_path, 'w') as f:
            json.dump(verdict, f, indent=2, default=str)
        mlflow.log_artifact(str(verdict_path))
        
        # Log composite score as JSON
        composite_score = compute_composite_score(suite_result)
        score_path = Path('artifacts') / f"{signal_name}_composite_score.json"
        with open(score_path, 'w') as f:
            json.dump(composite_score, f, indent=2, default=str)
        mlflow.log_artifact(str(score_path))
        
        # Log best metrics for History tab headline display
        best_key = get_best_config(suite_result, 'sharpe')
        if best_key and best_key in suite_result.results:
            best_result = suite_result.results[best_key]
            mlflow.log_metric('best_ann_ret', best_result.annual_return)
            mlflow.log_metric('best_max_dd', best_result.max_drawdown)
            mlflow.log_metric('best_turnover', best_result.turnover)
        
        # Log cap breakdown as CSV
        if best_key:
            cap_data = _extract_cap_breakdown(suite_result, best_key)
            if len(cap_data) > 0:
                cap_path = Path('artifacts') / f"{signal_name}_cap_breakdown.csv"
                cap_data.to_csv(cap_path, index=False)
                mlflow.log_artifact(str(cap_path))
        
        # Log year breakdown as CSV
        if best_key:
            year_data = _extract_year_breakdown(suite_result, best_key)
            if len(year_data) > 0:
                year_path = Path('artifacts') / f"{signal_name}_year_breakdown.csv"
                year_data.to_csv(year_path, index=False)
                mlflow.log_artifact(str(year_path))
        
        run_id = mlflow.active_run().info.run_id
    
    print(f"Logged to MLflow: run_id={run_id}")
    return run_id


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
