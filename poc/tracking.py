"""
MLflow Tracking: Log suite runs for reproducibility.
~80 lines - simple logging, let MLflow handle the rest.
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
from pathlib import Path
from typing import Optional
from datetime import datetime

from .suite import SuiteResult


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


def log_run(
    suite_result: SuiteResult,
    signal_name: str,
    catalog: dict,
    tearsheet_path: Optional[str] = None,
    experiment_name: str = 'backtest-poc',
    author: Optional[str] = None,
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
        
    Returns:
        MLflow run ID
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{signal_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Tags
        mlflow.set_tags({
            'signal_name': signal_name,
            'snapshot_id': catalog.get('snapshot_id', 'unknown'),
            'git_sha': get_git_sha(),
            'author': author or 'unknown',
        })
        
        # Params (from first config in results)
        if suite_result.results:
            first_result = list(suite_result.results.values())[0]
            mlflow.log_params(first_result.config)
        
        # Metrics - log for each config
        for config_key, result in suite_result.results.items():
            mlflow.log_metric(f'sharpe_{config_key}', result.sharpe)
            mlflow.log_metric(f'ann_ret_{config_key}', result.annual_return)
            mlflow.log_metric(f'max_dd_{config_key}', result.max_drawdown)
            mlflow.log_metric(f'turnover_{config_key}', result.turnover)
        
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
        
        # Log summary as CSV
        summary_path = Path('artifacts') / f"{signal_name}_summary.csv"
        summary_path.parent.mkdir(exist_ok=True)
        suite_result.summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        
        # Log daily series for all configs (combined)
        daily_data = []
        for config_key, result in suite_result.results.items():
            if 'cumret' in result.daily.columns:
                df = result.daily[['date', 'cumret']].copy()
                df['config'] = config_key
                daily_data.append(df)
        
        if daily_data:
            import pandas as pd
            combined_daily = pd.concat(daily_data, ignore_index=True)
            daily_path = Path('artifacts') / f"{signal_name}_daily.parquet"
            combined_daily.to_parquet(daily_path)
            mlflow.log_artifact(str(daily_path))
        
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
