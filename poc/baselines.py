"""
Baseline Library: Standard comparison signals.
~100 lines - three simple signals for benchmarking.
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Callable, Dict, Optional
from joblib import Memory

# Default date range for baselines
DEFAULT_START_DATE = '2017-01-01'
DEFAULT_END_DATE = '2018-12-31'

# Cache for baseline signals (persists to disk)
CACHE_DIR = Path('.cache/baselines')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)


def _get_catalog_hash(catalog: dict) -> str:
    """Get a hash to identify the catalog data for caching."""
    # Use shape + first/last dates as quick identifier
    ret_info = f"ret_{len(catalog['ret'])}_{catalog['ret']['date'].min()}_{catalog['ret']['date'].max()}"
    risk_info = f"risk_{len(catalog['risk'])}_{catalog['risk']['date'].min()}_{catalog['risk']['date'].max()}"
    return hashlib.md5(f"{ret_info}_{risk_info}".encode()).hexdigest()[:12]


def _filter_dates(df: pd.DataFrame, date_col: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if start_date:
        df = df[df[date_col] >= start_date]
    if end_date:
        df = df[df[date_col] <= end_date]
    return df


@memory.cache
def _compute_reversal(ret_df: pd.DataFrame, lookback: int, start_date: str, end_date: str, catalog_hash: str) -> pd.DataFrame:
    """Cached reversal computation."""
    ret = ret_df[['security_id', 'date', 'ret']].copy()
    ret = _filter_dates(ret, 'date', start_date, end_date)
    ret = ret.sort_values(['security_id', 'date'])
    
    ret['cum_ret'] = ret.groupby('security_id')['ret'].transform(
        lambda x: (1 + x).rolling(lookback, min_periods=lookback).apply(
            lambda y: y.prod() - 1, raw=True
        )
    )
    ret['signal'] = -ret['cum_ret']
    
    result = ret[['security_id', 'date', 'signal']].copy()
    result = result.rename(columns={'date': 'date_sig'})
    result['date_avail'] = result['date_sig'] + pd.Timedelta(days=1)
    return result.dropna()


def generate_reversal_signal(
    catalog: dict, 
    lookback: int = 5,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Short-term reversal signal: -1 * past N-day return.
    
    Idea: stocks that went down recently will bounce back.
    """
    catalog_hash = _get_catalog_hash(catalog)
    return _compute_reversal(catalog['ret'], lookback, start_date, end_date, catalog_hash)


@memory.cache
def _compute_momentum(ret_df: pd.DataFrame, lookback: int, skip: int, start_date: str, end_date: str, catalog_hash: str) -> pd.DataFrame:
    """Cached momentum computation."""
    ret = ret_df[['security_id', 'date', 'ret']].copy()
    ret = ret.sort_values(['security_id', 'date'])
    
    def rolling_ret(x, n):
        return (1 + x).rolling(n, min_periods=n).apply(lambda y: y.prod() - 1, raw=True)
    
    ret['ret_full'] = ret.groupby('security_id')['ret'].transform(rolling_ret, lookback)
    ret['ret_skip'] = ret.groupby('security_id')['ret'].transform(rolling_ret, skip)
    ret['signal'] = ret['ret_full'] - ret['ret_skip']
    ret = _filter_dates(ret, 'date', start_date, end_date)
    
    result = ret[['security_id', 'date', 'signal']].copy()
    result = result.rename(columns={'date': 'date_sig'})
    result['date_avail'] = result['date_sig'] + pd.Timedelta(days=1)
    return result.dropna()


def generate_momentum_signal(
    catalog: dict, 
    lookback: int = 252, 
    skip: int = 21,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    12-1 momentum signal: past 12-month return excluding last month.
    
    Idea: stocks with strong past performance (excluding very recent) continue to perform.
    """
    catalog_hash = _get_catalog_hash(catalog)
    return _compute_momentum(catalog['ret'], lookback, skip, start_date, end_date, catalog_hash)


@memory.cache
def _compute_value(risk_df: pd.DataFrame, start_date: str, end_date: str, catalog_hash: str) -> pd.DataFrame:
    """Cached value signal computation."""
    risk = risk_df[['security_id', 'date', 'value']].copy()
    risk = _filter_dates(risk, 'date', start_date, end_date)
    
    result = risk.rename(columns={'date': 'date_sig', 'value': 'signal'})
    result['date_avail'] = result['date_sig'] + pd.Timedelta(days=1)
    return result.dropna()


def generate_value_signal(
    catalog: dict,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Value signal: from risk file 'value' factor.
    
    Idea: cheap stocks (high book-to-market, etc.) outperform.
    """
    catalog_hash = _get_catalog_hash(catalog)
    return _compute_value(catalog['risk'], start_date, end_date, catalog_hash)


# Registry of all baseline signals
BASELINES: Dict[str, Callable] = {
    'reversal_5d': generate_reversal_signal,
    'value': generate_value_signal,
}


def generate_all_baselines(
    catalog: dict,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    snapshot_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all baseline signals.
    
    If snapshot_path is provided and pre-computed baselines exist, loads from disk
    and filters to date range (much faster than recomputing).
    
    Args:
        catalog: Data catalog
        start_date: Start date for signals (default: 2017-01-01)
        end_date: End date for signals (default: 2018-12-31)
        snapshot_path: Optional path to snapshot dir with pre-computed baselines
    
    Returns:
        Dict mapping baseline name to signal DataFrame
    """
    # Try to load pre-computed baselines
    if snapshot_path:
        precomputed = load_precomputed_baselines(snapshot_path)
        if precomputed:
            # Filter to requested date range
            result = {}
            for name, df in precomputed.items():
                filtered = _filter_dates(df, 'date_sig', start_date, end_date)
                result[name] = filtered
            return result
    
    # Fall back to computing from scratch
    return {
        'reversal_5d': generate_reversal_signal(catalog, lookback=5, start_date=start_date, end_date=end_date),
        'value': generate_value_signal(catalog, start_date=start_date, end_date=end_date),
    }


def clear_baseline_cache():
    """Clear the baseline signal cache."""
    memory.clear(warn=False)
    print(f"Cleared baseline cache at {CACHE_DIR}")


def precompute_baselines_to_snapshot(catalog: dict, snapshot_path: str) -> Dict[str, Path]:
    """
    Pre-compute baseline signals and save to snapshot folder for fast loading.
    
    Args:
        catalog: Data catalog (must have 'ret' and 'risk')
        snapshot_path: Path to snapshot directory
        
    Returns:
        Dict mapping baseline name to saved file path
    """
    snapshot_dir = Path(snapshot_path)
    
    # Use full date range from the data
    start_date = catalog['ret']['date'].min().strftime('%Y-%m-%d')
    end_date = catalog['ret']['date'].max().strftime('%Y-%m-%d')
    
    print(f"Pre-computing baselines for {start_date} to {end_date}...")
    
    saved_files = {}
    
    # Reversal 5-day
    print("  Computing reversal_5d...")
    reversal = generate_reversal_signal(catalog, lookback=5, start_date=start_date, end_date=end_date)
    reversal_path = snapshot_dir / 'baseline_reversal_5d.parquet'
    reversal.to_parquet(reversal_path, index=False)
    saved_files['reversal_5d'] = reversal_path
    print(f"    Saved {len(reversal):,} rows to {reversal_path.name}")
    
    # Value
    print("  Computing value...")
    value = generate_value_signal(catalog, start_date=start_date, end_date=end_date)
    value_path = snapshot_dir / 'baseline_value.parquet'
    value.to_parquet(value_path, index=False)
    saved_files['value'] = value_path
    print(f"    Saved {len(value):,} rows to {value_path.name}")
    
    print(f"Done! Saved {len(saved_files)} baseline files to {snapshot_dir}")
    return saved_files


def load_precomputed_baselines(snapshot_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load pre-computed baseline signals from snapshot folder.
    
    Args:
        snapshot_path: Path to snapshot directory
        
    Returns:
        Dict mapping baseline name to DataFrame, or None if files don't exist
    """
    snapshot_dir = Path(snapshot_path)
    
    baselines = {}
    
    reversal_path = snapshot_dir / 'baseline_reversal_5d.parquet'
    value_path = snapshot_dir / 'baseline_value.parquet'
    
    if reversal_path.exists():
        baselines['reversal_5d'] = pd.read_parquet(reversal_path)
    
    if value_path.exists():
        baselines['value'] = pd.read_parquet(value_path)
    
    if baselines:
        return baselines
    return None


def compute_signal_correlation(signal_df: pd.DataFrame, baseline_df: pd.DataFrame) -> float:
    """
    Compute cross-sectional correlation between two signals.
    
    Returns the average daily rank correlation.
    """
    # Merge signals on security_id and date
    merged = signal_df.merge(
        baseline_df,
        on=['security_id', 'date_sig'],
        suffixes=('', '_baseline')
    )
    
    if len(merged) == 0:
        return np.nan
    
    # Compute daily rank correlation
    def daily_corr(df):
        if len(df) < 10:
            return np.nan
        return df['signal'].corr(df['signal_baseline'], method='spearman')
    
    daily_corrs = merged.groupby('date_sig').apply(daily_corr, include_groups=False)
    
    return daily_corrs.mean()
