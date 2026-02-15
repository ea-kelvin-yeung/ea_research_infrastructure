"""
Suite Runner: Run a grid of backtest configs in one call.
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


def _run_single_config(signal_df, catalog, lag, residualize, byvar_list=None):
    """Helper to run a single config (for parallel execution)."""
    if byvar_list is None:
        byvar_list = ['overall', 'year', 'cap']
    config = BacktestConfig(lag=lag, residualize=residualize, byvar_list=byvar_list)
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
    byvar_list: list = None,
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
        byvar_list: Analysis slices (default: ['overall', 'year', 'cap'])
        
    Returns:
        SuiteResult with all results, summary table, and correlations
    """
    if byvar_list is None:
        byvar_list = ['overall', 'year', 'cap']
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
            key, result = _run_single_config(signal_df, catalog, lag, resid, byvar_list)
            if result is not None:
                results[key] = result
    else:
        # Parallel
        raw_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_config)(signal_df, catalog, lag, resid, byvar_list)
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
    
    # Run analytics in parallel for speed
    from concurrent.futures import ThreadPoolExecutor
    
    print("Computing analytics (parallel)...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_exposures = executor.submit(_compute_factor_exposures, signal_df, catalog)
        future_coverage = executor.submit(_compute_coverage, signal_df, catalog)
        future_ic = executor.submit(_compute_ic, signal_df, catalog)
        
        factor_exposures = future_exposures.result()
        coverage = future_coverage.result()
        ic_stats, ic_series = future_ic.result()
    
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
    
    Uses Polars for fast vectorized correlation computation.
    
    Returns DataFrame with columns: factor, correlation, abs_correlation
    """
    import polars as pl
    
    # Check required signal columns
    signal_cols = ['security_id', 'date_sig', 'signal']
    if not all(c in signal_df.columns for c in signal_cols):
        return pd.DataFrame()
    
    # Convert signal to Polars
    signal_pl = pl.from_pandas(signal_df[signal_cols]).rename({'date_sig': 'date'})
    signal_pl = signal_pl.with_columns(pl.col('security_id').cast(pl.Int32))
    
    # Get risk data - prefer master_pl from catalog
    master_pl = catalog.get('master_pl')
    if master_pl is None:
        risk_df = catalog.get('risk')
        if risk_df is None:
            return pd.DataFrame()
        master_pl = pl.from_pandas(risk_df)
    
    # Get available factors
    available_factors = [f for f in RISK_FACTORS if f in master_pl.columns]
    if not available_factors:
        return pd.DataFrame()
    
    # Join signal with factors
    factor_cols = ['security_id', 'date'] + available_factors
    merged = signal_pl.join(
        master_pl.select([c for c in factor_cols if c in master_pl.columns]),
        on=['security_id', 'date'],
        how='inner'
    )
    
    if len(merged) < 100:
        return pd.DataFrame()
    
    # Compute correlations using Polars (vectorized)
    rows = []
    for factor in available_factors:
        if factor in merged.columns:
            # Use select with pearson_corr expression
            corr = merged.select(pl.corr('signal', factor)).item()
            rows.append({
                'factor': factor,
                'correlation': corr,
                'abs_correlation': abs(corr) if corr is not None and not np.isnan(corr) else np.nan,
            })
    
    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values('abs_correlation', ascending=False)
    
    return result


def _compute_coverage(signal_df: pd.DataFrame, catalog: dict) -> Dict:
    """
    Compute signal coverage metrics using Polars for speed.
    
    Returns dict with:
        - avg_securities_per_day: Average number of securities with signal per day
        - coverage_pct: Percentage of universe covered
        - total_days: Number of days with signal
    """
    import polars as pl
    
    if 'date_sig' not in signal_df.columns or 'security_id' not in signal_df.columns:
        return {}
    
    # Convert to Polars for fast aggregation
    signal_pl = pl.from_pandas(signal_df[['date_sig', 'security_id']])
    
    # Securities per day (fast Polars groupby)
    per_day = signal_pl.group_by('date_sig').agg(pl.col('security_id').n_unique().alias('n_securities'))
    avg_securities = per_day['n_securities'].mean()
    total_days = len(per_day)
    
    # Unique securities in signal
    unique_signal_securities = signal_pl['security_id'].n_unique()
    
    # Try to get universe size from master_pl or risk file
    master_pl = catalog.get('master_pl')
    if master_pl is not None and 'security_id' in master_pl.columns:
        unique_universe_securities = master_pl['security_id'].n_unique()
    else:
        risk_df = catalog.get('risk')
        if risk_df is not None and 'security_id' in risk_df.columns:
            unique_universe_securities = risk_df['security_id'].nunique()
        else:
            unique_universe_securities = None
    
    coverage_pct = 100 * unique_signal_securities / unique_universe_securities if unique_universe_securities else np.nan
    
    return {
        'avg_securities_per_day': float(avg_securities) if avg_securities is not None else np.nan,
        'coverage_pct': float(coverage_pct) if coverage_pct is not None else np.nan,
        'total_days': total_days,
        'unique_securities': unique_signal_securities,
    }


def _compute_ic(signal_df: pd.DataFrame, catalog: dict) -> tuple:
    """
    Compute Information Coefficient (IC) - daily cross-sectional Spearman correlation
    between signal and forward returns.
    
    Uses Polars for fast vectorized computation.
    
    Returns:
        Tuple of (ic_stats dict, ic_series DataFrame)
    """
    import polars as pl
    
    if 'date_sig' not in signal_df.columns or 'security_id' not in signal_df.columns:
        return None, None
    
    if 'signal' not in signal_df.columns:
        return None, None
    
    # Determine which date column to use
    sig_date_col = 'date_avail' if 'date_avail' in signal_df.columns else 'date_sig'
    
    # Convert signal to Polars
    signal_pl = pl.from_pandas(signal_df[['security_id', sig_date_col, 'signal']])
    signal_pl = signal_pl.with_columns([
        pl.col('security_id').cast(pl.Int32),
        pl.col(sig_date_col).cast(pl.Datetime).alias('sig_date'),
        (pl.col(sig_date_col).cast(pl.Datetime) + pl.duration(days=1)).alias('lookup_date'),
    ])
    
    # Get returns data - prefer master_pl
    master_pl = catalog.get('master_pl')
    if master_pl is None:
        master_data = catalog.get('master')
        if master_data is not None:
            if isinstance(master_data, pd.DataFrame):
                if isinstance(master_data.index, pd.MultiIndex):
                    master_pl = pl.from_pandas(master_data.reset_index())
                else:
                    master_pl = pl.from_pandas(master_data)
            else:
                master_pl = master_data
        else:
            ret_df = catalog.get('ret')
            if ret_df is None or 'ret' not in ret_df.columns:
                return None, None
            master_pl = pl.from_pandas(ret_df)
    
    # Join to get forward returns - cast datetime to same precision
    ret_subset = (
        master_pl.select(['security_id', 'date', 'ret'])
        .rename({'date': 'lookup_date', 'ret': 'fwd_ret'})
        .with_columns(pl.col('lookup_date').cast(pl.Datetime('us')))
    )
    signal_pl = signal_pl.with_columns(pl.col('lookup_date').cast(pl.Datetime('us')))
    merged = signal_pl.join(ret_subset, on=['security_id', 'lookup_date'], how='inner')
    
    if len(merged) == 0:
        return None, None
    
    # Compute daily IC using Polars group_by with rank correlation
    # Spearman = Pearson correlation on ranks
    ic_daily = (
        merged
        .filter(pl.col('signal').is_not_null() & pl.col('fwd_ret').is_not_null())
        .with_columns([
            pl.col('signal').rank().over('sig_date').alias('sig_rank'),
            pl.col('fwd_ret').rank().over('sig_date').alias('ret_rank'),
            pl.len().over('sig_date').alias('n_obs'),
        ])
        .filter(pl.col('n_obs') >= 10)  # Minimum observations per day
        .group_by('sig_date')
        .agg([
            pl.corr('sig_rank', 'ret_rank').alias('ic'),
        ])
        .sort('sig_date')
        .drop_nulls()
    )
    
    if len(ic_daily) == 0:
        return None, None
    
    # Convert to pandas for output
    ic_by_date = ic_daily.to_pandas().set_index('sig_date')['ic']
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
