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
from .catalog import RISK_FACTORS


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
    factor_exposures: Optional[pd.DataFrame] = None  # Signal correlations to risk factors
    coverage: Optional[Dict] = None           # Signal coverage metrics


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
            # Skip if baseline has too few rows (e.g., momentum needs 252 days lookback)
            if len(baseline_df) < 100:
                print(f"  {name}: Skipped (only {len(baseline_df)} rows, need more data)")
                continue
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
    
    # Compute factor exposures
    print("Computing factor exposures...")
    factor_exposures = _compute_factor_exposures(signal_df, catalog)
    
    # Compute coverage metrics
    print("Computing coverage metrics...")
    coverage = _compute_coverage(signal_df, catalog)
    
    return SuiteResult(
        results=results,
        baselines=baselines,
        summary=summary,
        correlations=correlations,
        factor_exposures=factor_exposures,
        coverage=coverage,
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
    # Handle case where all values are NaN
    valid_rows = signal_rows[signal_rows[metric].notna()]
    if len(valid_rows) == 0:
        return signal_rows['config'].iloc[0] if len(signal_rows) > 0 else None
    return valid_rows.loc[valid_rows[metric].idxmax(), 'config']


def _compute_factor_exposures(signal_df: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    """
    Compute correlations between signal and risk factors.
    
    Uses pre-indexed factors DataFrame from catalog if available (10x faster).
    Falls back to risk DataFrame merge if factors not pre-computed.
    
    Returns DataFrame with columns: factor, correlation, abs_correlation
    """
    # Check required signal columns
    signal_cols = ['security_id', 'date_sig', 'signal']
    if not all(c in signal_df.columns for c in signal_cols):
        return pd.DataFrame()
    
    # Try to use pre-indexed factors (much faster)
    factors_df = catalog.get('factors')
    if factors_df is not None and len(factors_df) > 0:
        # Fast path: use pre-indexed factors with get_indexer + take
        signal_subset = signal_df[signal_cols].copy()
        
        # Build MultiIndex for lookup
        lookup_idx = pd.MultiIndex.from_arrays(
            [signal_subset['security_id'].values, signal_subset['date_sig'].values],
            names=['security_id', 'date']
        )
        
        # Get positions in factors index
        positions = factors_df.index.get_indexer(lookup_idx)
        valid_mask = positions >= 0
        
        if valid_mask.sum() < 100:
            return pd.DataFrame()
        
        # Get signal values for valid matches
        signal_values = signal_subset['signal'].values[valid_mask]
        
        # Compute correlations using numpy for speed
        rows = []
        for factor in factors_df.columns:
            factor_values = factors_df[factor].values[positions[valid_mask]]
            # Remove NaN pairs
            valid_pairs = ~(np.isnan(signal_values) | np.isnan(factor_values))
            if valid_pairs.sum() < 100:
                corr = np.nan
            else:
                corr = np.corrcoef(signal_values[valid_pairs], factor_values[valid_pairs])[0, 1]
            rows.append({
                'factor': factor,
                'correlation': corr,
                'abs_correlation': abs(corr) if not np.isnan(corr) else np.nan,
            })
    else:
        # Fallback: use risk DataFrame with merge
        risk_df = catalog.get('risk')
        if risk_df is None:
            return pd.DataFrame()
        
        signal_subset = signal_df[signal_cols].copy()
        
        # Get available risk factors
        available_factors = [f for f in RISK_FACTORS if f in risk_df.columns]
        if not available_factors:
            return pd.DataFrame()
        
        # Prepare risk data with renamed date column
        risk_cols = ['security_id', 'date'] + available_factors
        risk_subset = risk_df[risk_cols].rename(columns={'date': 'date_sig'})
        
        # Merge signal with risk factors
        merged = signal_subset.merge(risk_subset, on=['security_id', 'date_sig'], how='inner')
        
        if len(merged) < 100:
            return pd.DataFrame()
        
        # Compute correlations
        rows = []
        for factor in available_factors:
            if factor in merged.columns:
                corr = merged['signal'].corr(merged[factor])
                rows.append({
                    'factor': factor,
                    'correlation': corr,
                    'abs_correlation': abs(corr) if not np.isnan(corr) else np.nan,
                })
    
    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values('abs_correlation', ascending=False)
    
    return result


def _compute_coverage(signal_df: pd.DataFrame, catalog: dict) -> Dict:
    """
    Compute signal coverage metrics.
    
    Returns dict with:
        - avg_securities_per_day: Average number of securities with signal per day
        - coverage_pct: Percentage of universe covered
        - total_days: Number of days with signal
    """
    if 'date_sig' not in signal_df.columns or 'security_id' not in signal_df.columns:
        return {}
    
    # Securities per day
    securities_per_day = signal_df.groupby('date_sig')['security_id'].nunique()
    avg_securities = securities_per_day.mean()
    
    # Total unique securities in signal
    unique_signal_securities = signal_df['security_id'].nunique()
    
    # Try to get universe size from risk file
    risk_df = catalog.get('risk')
    if risk_df is not None and 'security_id' in risk_df.columns:
        unique_universe_securities = risk_df['security_id'].nunique()
        coverage_pct = 100 * unique_signal_securities / unique_universe_securities if unique_universe_securities > 0 else np.nan
    else:
        coverage_pct = np.nan
    
    return {
        'avg_securities_per_day': avg_securities,
        'coverage_pct': coverage_pct,
        'total_days': securities_per_day.count(),
        'unique_securities': unique_signal_securities,
    }
