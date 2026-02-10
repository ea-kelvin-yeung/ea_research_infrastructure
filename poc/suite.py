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
    ic_stats: Optional[Dict] = None           # IC summary stats (mean, t-stat, hit_rate)
    ic_series: Optional[pd.DataFrame] = None  # Daily IC time series


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
    baseline_signals = {}  # Store generated signals to avoid recomputing
    if include_baselines:
        print(f"Running baseline signals ({baseline_start_date} to {baseline_end_date})...")
        snapshot_path = catalog.get('snapshot_path')
        baseline_signals = generate_all_baselines(
            catalog, 
            start_date=baseline_start_date, 
            end_date=baseline_end_date,
            snapshot_path=snapshot_path,
        )
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
                import traceback
                print(f"  {name}: Failed - {e}")
                print(f"    Traceback: {traceback.format_exc()}")
    
    # Build summary table
    summary = _build_summary(results, baselines)
    
    # Compute correlations (reuse baseline_signals to avoid regenerating)
    correlations = _compute_correlations(signal_df, catalog, results, baselines, baseline_signals)
    
    # Compute factor exposures
    print("Computing factor exposures...")
    factor_exposures = _compute_factor_exposures(signal_df, catalog)
    
    # Compute coverage metrics
    print("Computing coverage metrics...")
    coverage = _compute_coverage(signal_df, catalog)
    
    # Compute Information Coefficient (IC)
    print("Computing IC...")
    ic_stats, ic_series = _compute_ic(signal_df, catalog)
    if ic_stats:
        print(f"  IC mean: {ic_stats['mean']:.4f}, t-stat: {ic_stats['t_stat']:.2f}, hit rate: {ic_stats['hit_rate']:.1f}%")
    
    return SuiteResult(
        results=results,
        baselines=baselines,
        summary=summary,
        correlations=correlations,
        factor_exposures=factor_exposures,
        coverage=coverage,
        ic_stats=ic_stats,
        ic_series=ic_series,
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
    baseline_signals: Dict[str, pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute signal and PnL correlations to baselines."""
    rows = []
    
    # Use provided baseline_signals to avoid regenerating
    if baseline_signals is None:
        baseline_signals = {}
    
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


def _compute_ic(signal_df: pd.DataFrame, catalog: dict) -> tuple:
    """
    Compute Information Coefficient (IC) - daily cross-sectional Spearman correlation
    between signal and forward returns.
    
    Returns:
        Tuple of (ic_stats dict, ic_series DataFrame)
        
        ic_stats contains:
        - mean: Mean daily IC
        - std: Std dev of daily IC
        - t_stat: t-statistic (mean / (std / sqrt(n)))
        - hit_rate: % of days with positive IC
        - ir: Information Ratio (mean / std)
        
        ic_series contains:
        - date: Trading date
        - ic: Daily IC value
    """
    if 'date_sig' not in signal_df.columns or 'security_id' not in signal_df.columns:
        return None, None
    
    if 'signal' not in signal_df.columns:
        return None, None
    
    # Determine which date column to use for the signal date
    # We'll group IC by this date for the time series
    if 'date_avail' in signal_df.columns:
        sig_date_col = 'date_avail'
    else:
        sig_date_col = 'date_sig'
    
    # Get returns data - prefer master_data (pre-indexed)
    master_data = catalog.get('master')
    if master_data is not None:
        # Fast path: use pre-indexed master_data
        signal = signal_df[['security_id', sig_date_col, 'signal']].copy()
        signal['security_id'] = signal['security_id'].astype(np.int32)
        signal[sig_date_col] = pd.to_datetime(signal[sig_date_col])
        
        # Look up forward return: return on the NEXT trading day after signal is available
        # This ensures IC measures predictive power, not contemporaneous correlation
        # Note: Using calendar day + 1; for exact trading day alignment, would need datefile
        signal['lookup_date'] = signal[sig_date_col] + pd.Timedelta(days=1)
        
        # Build lookup index using the forward-looking date
        lookup_idx = pd.MultiIndex.from_arrays(
            [signal['security_id'].values, signal['lookup_date'].values],
            names=['security_id', 'date']
        )
        
        # Get positions
        positions = master_data.index.get_indexer(lookup_idx)
        valid_mask = positions >= 0
        
        if valid_mask.sum() == 0:
            return None, None
        
        # Build merged DataFrame using numpy take
        merged = signal.loc[valid_mask].copy()
        merged['fwd_ret'] = np.take(master_data['ret'].values, positions[valid_mask])
        # Use signal date for grouping IC by date (not the return date)
        merged['date_sig'] = merged[sig_date_col]
    else:
        # Fallback to merge
        ret_df = catalog.get('ret')
        if ret_df is None or 'ret' not in ret_df.columns:
            return None, None
        
        signal = signal_df[['security_id', sig_date_col, 'signal']].copy()
        signal[sig_date_col] = pd.to_datetime(signal[sig_date_col])
        
        # Look up forward return: return on the NEXT day after signal is available
        signal['lookup_date'] = signal[sig_date_col] + pd.Timedelta(days=1)
        
        ret = ret_df[['security_id', 'date', 'ret']].copy()
        ret['date'] = pd.to_datetime(ret['date'])
        
        merged = signal.merge(
            ret.rename(columns={'date': 'lookup_date', 'ret': 'fwd_ret'}),
            on=['security_id', 'lookup_date'],
            how='inner'
        )
        merged['date_sig'] = merged[sig_date_col]
    
    if len(merged) == 0:
        return None, None
    
    # Compute daily IC using vectorized numpy (faster than groupby.apply)
    # Sort by date for efficient groupby
    merged = merged.sort_values('date_sig')
    
    dates = merged['date_sig'].values
    signals = merged['signal'].values.astype(np.float64)
    returns = merged['fwd_ret'].values.astype(np.float64)
    
    # Find unique dates and their boundaries
    unique_dates, start_indices = np.unique(dates, return_index=True)
    end_indices = np.append(start_indices[1:], len(dates))
    
    # Compute IC for each date using numpy
    ic_values = []
    ic_dates = []
    for date, start, end in zip(unique_dates, start_indices, end_indices):
        sig_day = signals[start:end]
        ret_day = returns[start:end]
        
        # Remove NaNs
        valid = ~(np.isnan(sig_day) | np.isnan(ret_day))
        if valid.sum() < 10:
            continue
        
        sig_valid = sig_day[valid]
        ret_valid = ret_day[valid]
        
        # Spearman correlation = Pearson on ranks
        sig_ranks = sig_valid.argsort().argsort()
        ret_ranks = ret_valid.argsort().argsort()
        
        n = len(sig_ranks)
        sig_mean = sig_ranks.mean()
        ret_mean = ret_ranks.mean()
        
        cov = np.sum((sig_ranks - sig_mean) * (ret_ranks - ret_mean))
        sig_std = np.sqrt(np.sum((sig_ranks - sig_mean) ** 2))
        ret_std = np.sqrt(np.sum((ret_ranks - ret_mean) ** 2))
        
        if sig_std > 0 and ret_std > 0:
            corr = cov / (sig_std * ret_std)
            ic_values.append(corr)
            ic_dates.append(date)
    
    if len(ic_values) == 0:
        return None, None
    
    ic_by_date = pd.Series(ic_values, index=ic_dates)
    ic_series = pd.DataFrame({
        'date': ic_by_date.index,
        'ic': ic_by_date.values
    }).dropna()
    
    if len(ic_series) == 0:
        return None, None
    
    # Compute summary stats
    ic_values = ic_series['ic'].values
    n = len(ic_values)
    ic_mean = np.mean(ic_values)
    ic_std = np.std(ic_values)
    ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
    ic_hit_rate = np.mean(ic_values > 0) * 100  # % of positive IC days
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0  # Information Ratio
    
    ic_stats = {
        'mean': ic_mean,
        'std': ic_std,
        't_stat': ic_t_stat,
        'hit_rate': ic_hit_rate,
        'ir': ic_ir,
        'n_days': n,
    }
    
    return ic_stats, ic_series
