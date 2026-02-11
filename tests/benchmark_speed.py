#!/usr/bin/env python3
"""Speed benchmark: BacktestFast vs original Backtest."""

import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from poc.catalog import load_catalog
from poc.contract import prepare_signal
from backtest_engine import Backtest
from backtest_engine_fast import BacktestFast

# ============ CONFIG ============
SNAPSHOT = "snapshots/2026-02-10-v1"
N_SECURITIES = None  # None = all securities
N_YEARS = 10
N_RUNS = 3
UNIVERSE_ONLY = True
# ================================


def create_signal(catalog: dict) -> pd.DataFrame:
    """Create signal for benchmarking."""
    ret = catalog['ret']
    securities = ret['security_id'].unique()
    if N_SECURITIES is not None and len(securities) > N_SECURITIES:
        securities = np.random.choice(securities, N_SECURITIES, replace=False)
    
    dates = catalog['dates']
    date_list = sorted(dates['date'].unique())
    n_days = N_YEARS * 252
    sample_dates = date_list[-n_days:]
    
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
    signal_df['security_id'] = signal_df['security_id'].astype(np.int32)
    
    aligned = prepare_signal(signal_df, catalog['dates'], lag=0, validate=False)
    print(f"Signal: {len(aligned):,} rows, {len(securities)} securities, {len(sample_dates)} dates ({N_YEARS} years)")
    return aligned


def run_backtest_fast(signal_df, catalog):
    """Run BacktestFast with master_data."""
    bt = BacktestFast(
        infile=signal_df,
        retfile=catalog['ret'],
        otherfile=catalog['risk'],
        datefile=catalog['dates'],
        master_data=catalog.get('master'),
        sigvar='signal',
        byvar_list=['overall'],
        fractile=(10, 90),
        weight='equal',
        tc_model='naive',
        verbose=False,
    )
    if hasattr(bt, 'set_precomputed_indexes'):
        bt.set_precomputed_indexes(
            dates_indexed=catalog.get('dates_indexed'),
            asof_tables=catalog.get('asof_tables'),
            master_pl=catalog.get('master_pl'),
            otherfile_pl=catalog.get('otherfile_pl'),
            retfile_pl=catalog.get('retfile_pl'),
            datefile_pl=catalog.get('datefile_pl'),
            asof_tables_pl=catalog.get('asof_tables_pl'),
        )
    return bt.gen_result()


def run_backtest_original(signal_df, catalog):
    """Run original Backtest."""
    bt = Backtest(
        infile=signal_df,
        retfile=catalog['ret'],
        otherfile=catalog['risk'],
        datefile=catalog['dates'],
        sigvar='signal',
        byvar_list=['overall'],
        fractile=(10, 90),
        weight='equal',
        tc_model='naive',
        verbose=False,
    )
    return bt.gen_result()


def benchmark(name, run_fn, signal_df, catalog):
    """Run benchmark N_RUNS times."""
    times = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        result = run_fn(signal_df, catalog)
        elapsed = time.perf_counter() - start
        sharpe = result[0]['sharpe_ret'].iloc[0] if 'sharpe_ret' in result[0].columns else 0
        times.append(elapsed)
        print(f"  {name} Run {i+1}: {elapsed:.2f}s (Sharpe: {sharpe:.3f})")
    return times


def main():
    univ_str = "universe_only" if UNIVERSE_ONLY else "full"
    print(f"Loading catalog from {SNAPSHOT} ({univ_str})...")
    catalog = load_catalog(SNAPSHOT, universe_only=UNIVERSE_ONLY)
    print(f"Catalog: {len(catalog['ret']):,} ret, {len(catalog.get('master', pd.DataFrame())):,} master rows")
    
    if 'master' in catalog and catalog['master'] is not None:
        _ = len(catalog['master'])
    if 'master_pl' in catalog and catalog['master_pl'] is not None:
        _ = len(catalog['master_pl'])
    print("Data preloaded")
    
    np.random.seed(42)
    signal_df = create_signal(catalog)
    
    sec_str = f"{N_SECURITIES}" if N_SECURITIES else "all"
    print(f"\n{'='*60}")
    print(f"Benchmark: {N_YEARS} years, {sec_str} securities, {N_RUNS} runs, {univ_str}")
    print(f"{'='*60}\n")
    
    times_fast = benchmark("BacktestFast", run_backtest_fast, signal_df, catalog)
    print()
    times_orig = benchmark("Backtest    ", run_backtest_original, signal_df, catalog)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"BacktestFast: avg={np.mean(times_fast):.2f}s, min={np.min(times_fast):.2f}s")
    print(f"Backtest:     avg={np.mean(times_orig):.2f}s, min={np.min(times_orig):.2f}s")
    print(f"Speedup: {np.mean(times_orig)/np.mean(times_fast):.2f}x")


if __name__ == '__main__':
    main()
