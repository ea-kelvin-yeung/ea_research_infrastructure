#!/usr/bin/env python3
"""
Profile backtest to identify performance bottlenecks.

Usage:
    python profile_backtest.py [--top N] [--cumulative]
"""

import cProfile
import pstats
import io
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'poc'))

from poc.catalog import load_catalog, create_snapshot, list_snapshots
from poc.wrapper import run_backtest, BacktestConfig


def create_sample_signal(catalog: dict, n_securities: int = 500) -> pd.DataFrame:
    """Create a sample signal for profiling."""
    ret = catalog['ret']
    
    securities = ret['security_id'].unique()
    if len(securities) > n_securities:
        securities = np.random.choice(securities, n_securities, replace=False)
    
    dates = catalog['dates']
    date_list = dates['date'].unique()
    sample_dates = date_list[-252:]  # Last 1 year of trading days
    
    records = []
    for date in sample_dates:
        for sec_id in securities:
            records.append({
                'security_id': sec_id,
                'date_sig': date,
                'date_avail': date,
                'signal': np.random.randn(),
            })
    
    signal_df = pd.DataFrame(records)
    print(f"Created sample signal: {len(signal_df)} rows, {len(securities)} securities, {len(sample_dates)} dates")
    return signal_df


def run_profiled_backtest(catalog: dict, signal_df: pd.DataFrame, config: BacktestConfig):
    """Run a single backtest (for profiling)."""
    return run_backtest(signal_df, catalog, config, validate=False)


def main():
    parser = argparse.ArgumentParser(description='Profile backtest performance')
    parser.add_argument('--top', type=int, default=30, help='Number of top functions to show')
    parser.add_argument('--cumulative', action='store_true', help='Sort by cumulative time instead of total time')
    parser.add_argument('--securities', type=int, default=500, help='Number of securities in sample signal')
    parser.add_argument('--resid', action='store_true', help='Enable residualization (slower path)')
    args = parser.parse_args()
    
    # Ensure snapshot exists
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        print("No snapshots found. Creating from data/ directory...")
        create_snapshot('data', 'snapshots', 'default')
        snapshots = ['default']
    
    snapshot_name = snapshots[0]
    print(f"Loading catalog from snapshots/{snapshot_name}...")
    catalog = load_catalog(f"snapshots/{snapshot_name}")
    print(f"Catalog loaded: {len(catalog['ret'])} ret rows, {len(catalog['risk'])} risk rows")
    
    print("\nCreating sample signal...")
    np.random.seed(42)
    signal_df = create_sample_signal(catalog, n_securities=args.securities)
    
    config = BacktestConfig(
        lag=0,
        residualize='all' if args.resid else 'off',
        tc_model='naive',
        weight='equal',
        fractile=(10, 90),
        byvar_list=['overall', 'year', 'cap'],
    )
    
    print(f"\nProfiling backtest (residualize={config.residualize})...")
    print("=" * 60)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = run_profiled_backtest(catalog, signal_df, config)
    
    profiler.disable()
    
    print(f"\nBacktest completed. Sharpe: {result.sharpe:.3f}")
    print("=" * 60)
    print(f"\nTop {args.top} functions by {'cumulative' if args.cumulative else 'total'} time:\n")
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    
    if args.cumulative:
        stats.sort_stats('cumulative')
    else:
        stats.sort_stats('tottime')
    
    stats.print_stats(args.top)
    print(stream.getvalue())
    
    print("\n" + "=" * 60)
    print("Detailed breakdown of top 5 functions:\n")
    
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats('tottime')
    stats2.print_callees(5)
    print(stream2.getvalue())


if __name__ == '__main__':
    main()
