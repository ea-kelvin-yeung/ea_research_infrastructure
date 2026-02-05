#!/usr/bin/env python
"""
Generate a deterministic 1-year test signal for backtest engine validation.

This creates a reproducible test signal that can be used to verify
that the original and optimized backtest engines produce identical results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.catalog import load_catalog, list_snapshots


def generate_test_signal(
    start_date: str = '2018-01-01',
    end_date: str = '2018-12-31',
    n_securities: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a deterministic test signal.
    
    Args:
        start_date: Start of signal period
        end_date: End of signal period  
        n_securities: Number of securities to include
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: security_id, date_sig, date_avail, signal
    """
    np.random.seed(seed)
    
    # Load catalog to get valid security_ids and trading dates
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        raise RuntimeError("No snapshots found. Create one first with: python -c \"from poc.catalog import create_snapshot; create_snapshot()\"")
    
    catalog = load_catalog(f'snapshots/{snapshots[0]}')
    
    # Get unique securities from ret file
    all_securities = catalog['ret']['security_id'].unique()
    
    # Get trading dates in range
    dates = catalog['dates']['date'].unique()
    dates = pd.to_datetime(dates)
    dates = dates[(dates >= start_date) & (dates <= end_date)]
    dates = sorted(dates)
    
    print(f"Available: {len(all_securities)} securities, {len(dates)} trading dates")
    
    # Sample securities
    n_securities = min(n_securities, len(all_securities))
    sample_securities = np.random.choice(all_securities, n_securities, replace=False)
    
    print(f"Using {n_securities} securities, {len(dates)} dates")
    
    # Generate signal for each date and security
    rows = []
    for date in dates:
        for sec_id in sample_securities:
            # Generate signal with some structure (not pure random)
            # This creates a signal with cross-sectional variation
            signal_value = np.random.randn() + 0.1 * (sec_id % 100) / 100
            
            rows.append({
                'security_id': int(sec_id),
                'date_sig': date,
                'date_avail': date + pd.Timedelta(days=1),
                'signal': signal_value,
            })
    
    signal_df = pd.DataFrame(rows)
    signal_df['date_sig'] = pd.to_datetime(signal_df['date_sig'])
    signal_df['date_avail'] = pd.to_datetime(signal_df['date_avail'])
    
    print(f"Generated signal: {len(signal_df):,} rows")
    print(f"Date range: {signal_df['date_sig'].min()} to {signal_df['date_sig'].max()}")
    
    return signal_df


def main():
    """Generate and save test signal."""
    print("=" * 60)
    print("Generating 1-Year Test Signal")
    print("=" * 60)
    
    # Generate signal
    signal_df = generate_test_signal(
        start_date='2018-01-01',
        end_date='2018-12-31',
        n_securities=500,
        seed=42,
    )
    
    # Save to fixtures
    output_path = Path(__file__).parent / 'fixtures' / 'test_signal_1year.parquet'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Validate
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == len(signal_df), "Row count mismatch"
    assert list(loaded.columns) == ['security_id', 'date_sig', 'date_avail', 'signal'], "Column mismatch"
    
    print("\nValidation passed!")
    

if __name__ == '__main__':
    main()
