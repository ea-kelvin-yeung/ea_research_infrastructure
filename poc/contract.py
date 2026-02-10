"""
Signal Contract: Schema validation and date alignment.
~100 lines - keep it simple.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

# Add parent directory to path to import backtest_engine
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest_engine import gen_date_trading


# Required columns for any signal
REQUIRED_COLUMNS = ['security_id', 'date_sig', 'date_avail', 'signal']

# Expected dtypes
EXPECTED_DTYPES = {
    'security_id': ['int64', 'int32', 'object', 'str'],
    'date_sig': ['datetime64[ns]'],
    'date_avail': ['datetime64[ns]'],
    'signal': ['float64', 'float32', 'int64', 'int32'],
}


def validate_signal(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate a signal DataFrame against the contract.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if df is None or df.empty:
        return False, ["DataFrame is None or empty"]
    
    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if errors:
        return False, errors
    
    # Check for duplicates
    dups = df.duplicated(subset=['security_id', 'date_sig'], keep=False)
    if dups.any():
        n_dups = dups.sum()
        errors.append(f"Found {n_dups} duplicate (security_id, date_sig) pairs")
    
    # Check signal is numeric and finite
    if not pd.api.types.is_numeric_dtype(df['signal']):
        errors.append("Signal column must be numeric")
    else:
        n_inf = np.isinf(df['signal']).sum()
        n_nan = df['signal'].isna().sum()
        if n_inf > 0:
            errors.append(f"Signal contains {n_inf} infinite values")
        if n_nan > 0:
            errors.append(f"Signal contains {n_nan} NaN values (will be dropped)")
    
    # Check dates are datetime
    for date_col in ['date_sig', 'date_avail']:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            errors.append(f"{date_col} must be datetime type, got {df[date_col].dtype}")
    
    # Check no lookahead (date_avail >= date_sig)
    if pd.api.types.is_datetime64_any_dtype(df['date_sig']) and pd.api.types.is_datetime64_any_dtype(df['date_avail']):
        lookahead = (df['date_avail'] < df['date_sig']).sum()
        if lookahead > 0:
            errors.append(f"Found {lookahead} rows where date_avail < date_sig (lookahead bias)")
    
    return len(errors) == 0, errors


def align_dates(
    df: pd.DataFrame,
    datefile: pd.DataFrame,
    lag: int = 0,
    avail_hour: int = 8,
) -> pd.DataFrame:
    """
    Align signal dates to trading dates, adding date_ret column.
    
    Args:
        df: Signal DataFrame with date_sig, date_avail columns
        datefile: Trading calendar with 'date' and 'n' columns
        lag: Additional lag in trading days (0 = trade next available day)
        avail_hour: Hour when signal is available (default 8 = before market open)
    
    Returns:
        DataFrame with date_ret and date_openret columns added
    """
    # Prepare signal for gen_date_trading
    signal_df = df.copy()
    
    # Rename signal column temporarily if needed
    sig_col = 'signal'
    varlist = [sig_col] if sig_col in signal_df.columns else []
    
    # Use existing gen_date_trading function
    aligned = gen_date_trading(
        infile=signal_df,
        datefile=datefile,
        varlist=varlist,
        avail_time=avail_hour,
        date_signal='date_sig',
        date_available='date_avail',
        buffer=lag,
    )
    
    return aligned


def prepare_signal(
    df: pd.DataFrame,
    datefile: pd.DataFrame,
    lag: int = 0,
    avail_hour: int = 16,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Convenience function: validate and align a signal in one step.
    
    Raises:
        ValueError: If validation fails
    """
    if validate:
        is_valid, errors = validate_signal(df)
        if not is_valid:
            raise ValueError(f"Signal validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    # Drop NaN signals
    df = df.dropna(subset=['signal'])
    
    # Align dates
    aligned = align_dates(df, datefile, lag=lag, avail_hour=avail_hour)
    
    return aligned
