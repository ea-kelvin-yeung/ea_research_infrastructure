"""
Baseline Library: Standard comparison signals.
~100 lines - three simple signals for benchmarking.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Optional

# Default date range for baselines
DEFAULT_START_DATE = '2017-01-01'
DEFAULT_END_DATE = '2018-12-31'


def _filter_dates(df: pd.DataFrame, date_col: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if start_date:
        df = df[df[date_col] >= start_date]
    if end_date:
        df = df[df[date_col] <= end_date]
    return df


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
    ret = catalog['ret'][['security_id', 'date', 'ret']].copy()
    ret = _filter_dates(ret, 'date', start_date, end_date)
    ret = ret.sort_values(['security_id', 'date'])
    
    # Calculate rolling return
    ret['cum_ret'] = ret.groupby('security_id')['ret'].transform(
        lambda x: (1 + x).rolling(lookback, min_periods=lookback).apply(
            lambda y: y.prod() - 1, raw=True
        )
    )
    
    # Reversal = negative of past return
    ret['signal'] = -ret['cum_ret']
    
    # Format for signal contract
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
    ret = catalog['ret'][['security_id', 'date', 'ret']].copy()
    ret = ret.sort_values(['security_id', 'date'])
    
    # Calculate rolling returns on FULL data first (need lookback history)
    def rolling_ret(x, n):
        return (1 + x).rolling(n, min_periods=n).apply(lambda y: y.prod() - 1, raw=True)
    
    ret['ret_full'] = ret.groupby('security_id')['ret'].transform(rolling_ret, lookback)
    ret['ret_skip'] = ret.groupby('security_id')['ret'].transform(rolling_ret, skip)
    
    # Momentum = full period minus recent period
    ret['signal'] = ret['ret_full'] - ret['ret_skip']
    
    # Filter to date range AFTER calculating rolling (need history for lookback)
    ret = _filter_dates(ret, 'date', start_date, end_date)
    
    # Format for signal contract
    result = ret[['security_id', 'date', 'signal']].copy()
    result = result.rename(columns={'date': 'date_sig'})
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
    risk = catalog['risk'][['security_id', 'date', 'value']].copy()
    risk = _filter_dates(risk, 'date', start_date, end_date)
    
    # Format for signal contract
    result = risk.rename(columns={'date': 'date_sig', 'value': 'signal'})
    result['date_avail'] = result['date_sig'] + pd.Timedelta(days=1)
    
    return result.dropna()


# Registry of all baseline signals
BASELINES: Dict[str, Callable] = {
    'reversal_5d': generate_reversal_signal,
    'momentum_12_1': generate_momentum_signal,
    'value': generate_value_signal,
}


def generate_all_baselines(
    catalog: dict,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all baseline signals.
    
    Args:
        catalog: Data catalog
        start_date: Start date for signals (default: 2017-01-01)
        end_date: End date for signals (default: 2018-12-31)
    
    Returns:
        Dict mapping baseline name to signal DataFrame
    """
    return {
        'reversal_5d': generate_reversal_signal(catalog, lookback=5, start_date=start_date, end_date=end_date),
        'momentum_12_1': generate_momentum_signal(catalog, lookback=252, skip=21, start_date=start_date, end_date=end_date),
        'value': generate_value_signal(catalog, start_date=start_date, end_date=end_date),
    }


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
    
    daily_corrs = merged.groupby('date_sig').apply(daily_corr)
    
    return daily_corrs.mean()
