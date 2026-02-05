"""
Suite Runner: Run a grid of backtest configs in one call.
~150 lines - the "one button" experience.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from itertools import product
from joblib import Parallel, delayed

from .wrapper import BacktestConfig, BacktestResult, run_backtest
from .baselines import generate_all_baselines, compute_signal_correlation, BASELINES, DEFAULT_START_DATE, DEFAULT_END_DATE


# Default suite configuration
DEFAULT_GRID = {
    'lags': [0, 1, 2],
    'residualize': ['off', 'industry'],
}


@dataclass
class SuiteResult:
    """Result from running a full suite."""
    results: Dict[str, BacktestResult]       # Config key -> result
    baselines: Dict[str, BacktestResult]     # Baseline name -> result
    summary: pd.DataFrame                     # One row per config
    correlations: pd.DataFrame                # Signal/PnL correlations to baselines


def _run_single_config(signal_df, catalog, lag, residualize):
    """Helper to run a single config (for parallel execution)."""
    config = BacktestConfig(lag=lag, residualize=residualize)
    try:
        result = run_backtest(signal_df, catalog, config, validate=False)
        return config.config_key(), result
    except Exception as e:
        print(f"Warning: {config.config_key()} failed: {e}")
        return config.config_key(), None


def run_suite(
    signal_df: pd.DataFrame,
    catalog: dict,
    grid: dict = None,
    include_baselines: bool = True,
    n_jobs: int = 1,
    baseline_start_date: str = DEFAULT_START_DATE,
    baseline_end_date: str = DEFAULT_END_DATE,
) -> SuiteResult:
    """
    Run a suite of backtests across a config grid.
    
    Args:
        signal_df: Signal DataFrame
        catalog: Data catalog
        grid: Config grid (default: DEFAULT_GRID)
        include_baselines: Whether to run baselines for comparison
        n_jobs: Number of parallel jobs (1 = sequential)
        baseline_start_date: Start date for baseline signals (default: 2017-01-01)
        baseline_end_date: End date for baseline signals (default: 2018-12-31)
        
    Returns:
        SuiteResult with all results, summary table, and correlations
    """
    if grid is None:
        grid = DEFAULT_GRID
    
    lags = grid.get('lags', [0])
    residualize_opts = grid.get('residualize', ['off'])
    
    # Generate all config combinations
    configs = list(product(lags, residualize_opts))
    
    print(f"Running {len(configs)} backtest configs...")
    
    # Run all configs
    if n_jobs == 1:
        # Sequential
        results = {}
        for lag, resid in configs:
            key, result = _run_single_config(signal_df, catalog, lag, resid)
            if result is not None:
                results[key] = result
    else:
        # Parallel
        raw_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_config)(signal_df, catalog, lag, resid)
            for lag, resid in configs
        )
        results = {k: v for k, v in raw_results if v is not None}
    
    print(f"Completed {len(results)} backtests")
    
    # Run baselines
    baselines = {}
    if include_baselines:
        print(f"Running baseline signals ({baseline_start_date} to {baseline_end_date})...")
        baseline_signals = generate_all_baselines(catalog, start_date=baseline_start_date, end_date=baseline_end_date)
        for name, baseline_df in baseline_signals.items():
            try:
                config = BacktestConfig(lag=0, residualize='off')
                baselines[name] = run_backtest(baseline_df, catalog, config, validate=False)
                print(f"  {name}: Sharpe = {baselines[name].sharpe:.2f}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")
    
    # Build summary table
    summary = _build_summary(results, baselines)
    
    # Compute correlations
    correlations = _compute_correlations(signal_df, catalog, results, baselines, baseline_start_date, baseline_end_date)
    
    return SuiteResult(
        results=results,
        baselines=baselines,
        summary=summary,
        correlations=correlations,
    )


def _build_summary(results: Dict[str, BacktestResult], baselines: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Build summary table from results."""
    rows = []
    
    # Add signal results
    for key, result in results.items():
        rows.append({
            'config': key,
            'type': 'signal',
            'sharpe': result.sharpe,
            'ann_ret': result.annual_return,
            'max_dd': result.max_drawdown,
            'turnover': result.turnover,
        })
    
    # Add baseline results
    for name, result in baselines.items():
        rows.append({
            'config': name,
            'type': 'baseline',
            'sharpe': result.sharpe,
            'ann_ret': result.annual_return,
            'max_dd': result.max_drawdown,
            'turnover': result.turnover,
        })
    
    return pd.DataFrame(rows)


def _compute_correlations(
    signal_df: pd.DataFrame,
    catalog: dict,
    results: Dict[str, BacktestResult],
    baselines: Dict[str, BacktestResult],
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Compute signal and PnL correlations to baselines."""
    rows = []
    
    baseline_signals = generate_all_baselines(catalog, start_date=start_date, end_date=end_date)
    
    for baseline_name, baseline_df in baseline_signals.items():
        # Signal correlation
        sig_corr = compute_signal_correlation(signal_df, baseline_df)
        
        # PnL correlation (using lag0_residoff config)
        pnl_corr = np.nan
        if 'lag0_residoff' in results and baseline_name in baselines:
            signal_daily = results['lag0_residoff'].daily
            baseline_daily = baselines[baseline_name].daily
            if 'ret' in signal_daily.columns and 'ret' in baseline_daily.columns:
                merged = signal_daily[['date', 'ret']].merge(
                    baseline_daily[['date', 'ret']],
                    on='date',
                    suffixes=('', '_baseline')
                )
                if len(merged) > 10:
                    pnl_corr = merged['ret'].corr(merged['ret_baseline'])
        
        rows.append({
            'baseline': baseline_name,
            'signal_corr': sig_corr,
            'pnl_corr': pnl_corr,
        })
    
    return pd.DataFrame(rows)


def get_best_config(suite_result: SuiteResult, metric: str = 'sharpe') -> str:
    """Get the config key with the best metric value."""
    signal_rows = suite_result.summary[suite_result.summary['type'] == 'signal']
    if len(signal_rows) == 0:
        return None
    return signal_rows.loc[signal_rows[metric].idxmax(), 'config']
