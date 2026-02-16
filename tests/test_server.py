"""
Test BacktestService with persistent master data.

Demonstrates the benefit of loading data once and running multiple backtests.
Uses real signal data: data/reversal_signal_analyst.csv (2012-2021, 10 years)

Run:
    python tests/test_server.py                    # Direct BacktestFastV2
    python tests/test_server.py --use-service      # Use BacktestService.run_fast()
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Real signal file path
SIGNAL_FILE = "data/reversal_signal_analyst.csv"


def load_real_signal(start_year: int = 2012, end_year: int = 2021) -> pd.DataFrame:
    """Load real signal data from CSV, filtered to date range."""
    sig = pd.read_csv(SIGNAL_FILE)
    sig['date_sig'] = pd.to_datetime(sig['date_sig'])
    sig['date_avail'] = pd.to_datetime(sig['date_avail'])
    
    # Filter to date range (inclusive)
    sig = sig[
        (sig['date_sig'].dt.year >= start_year) &
        (sig['date_sig'].dt.year <= end_year)
    ]
    
    # Add date_ret (next business day) - pre-align for direct engine use
    sig['date_ret'] = sig['date_sig'] + pd.Timedelta(days=1)
    
    return sig


def test_with_service(args):
    """Test using BacktestService.run_fast()."""
    from api.service import BacktestService
    
    print("="*70)
    print("BACKTEST SERVICE TEST (using run_fast)")
    print("="*70)
    
    # Reset and load with date filtering at init
    BacktestService.reset()
    
    t0 = time.perf_counter()
    service = BacktestService.get(
        args.snapshot,
        start_date=f"{args.start_year}-01-01",
        end_date=f"{args.end_year}-12-31",
    )
    load_time = time.perf_counter() - t0
    
    print(f"\n1. COLD START:")
    print(f"   Data load:   {load_time:.1f}s")
    print(f"   Master rows: {len(service._master_pd):,}")
    
    # Load signal
    sig = load_real_signal(args.start_year, args.end_year)
    print(f"   Signal file: {SIGNAL_FILE}")
    print(f"   Signal rows: {len(sig):,}")
    
    # First run
    t0 = time.perf_counter()
    result = service.run_fast(sig, sigvar='signal', byvar_list=['overall'])
    first_run = time.perf_counter() - t0
    
    cold_total = load_time + first_run
    print(f"   First run:   {first_run:.1f}s")
    print(f"   Cold total:  {cold_total:.1f}s")
    
    # Warm runs
    print(f"\n2. WARM START ({args.num_runs} runs):")
    warm_times = []
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        result = service.run_fast(sig, sigvar='signal', byvar_list=['overall'])
        elapsed = time.perf_counter() - t0
        warm_times.append(elapsed)
        
        summary = result[0]
        sharpe = summary.iloc[0]['sharpe_ret'] if len(summary) > 0 else 0
        print(f"   Run {i+1}: {elapsed:.1f}s (Sharpe: {sharpe:.2f})")
    
    return cold_total, np.mean(warm_times), len(sig)


def test_direct(args):
    """Test using direct BacktestFastV2."""
    from poc.catalog import load_catalog
    from backtest_wrapper import BacktestFastV2
    
    print("="*70)
    print("DIRECT BACKTEST TEST (using BacktestFastV2)")
    print("="*70)
    
    t0 = time.perf_counter()
    
    # Load and filter catalog
    snapshot = args.snapshot or "2026-02-10-v1"
    catalog = load_catalog(f"snapshots/{snapshot}", universe_only=False)
    
    # Filter to date range
    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"
    catalog['ret'] = catalog['ret'][
        (catalog['ret']['date'] >= start_date) & 
        (catalog['ret']['date'] <= end_date)
    ]
    catalog['dates'] = catalog['dates'][
        (catalog['dates']['date'] >= start_date) & 
        (catalog['dates']['date'] <= end_date)
    ]
    master = catalog['master']
    if isinstance(master.index, pd.MultiIndex):
        master = master.reset_index()
    master = master[(master['date'] >= start_date) & (master['date'] <= end_date)]
    datefile = catalog['dates']
    
    load_time = time.perf_counter() - t0
    
    print(f"\n1. COLD START:")
    print(f"   Data load:   {load_time:.1f}s")
    print(f"   Master rows: {len(master):,}")
    
    # Load signal
    sig = load_real_signal(args.start_year, args.end_year)
    sig = sig.rename(columns={'signal': 'my_signal'})
    print(f"   Signal file: {SIGNAL_FILE}")
    print(f"   Signal rows: {len(sig):,}")
    
    # First run
    t0 = time.perf_counter()
    bt = BacktestFastV2(
        infile=sig, retfile=master, otherfile=master, datefile=datefile,
        sigvar='my_signal', byvar_list=['overall'], resid=False,
    )
    result = bt.gen_result()
    first_run = time.perf_counter() - t0
    
    cold_total = load_time + first_run
    print(f"   First run:   {first_run:.1f}s")
    print(f"   Cold total:  {cold_total:.1f}s")
    
    # Warm runs
    print(f"\n2. WARM START ({args.num_runs} runs):")
    warm_times = []
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        bt = BacktestFastV2(
            infile=sig, retfile=master, otherfile=master, datefile=datefile,
            sigvar='my_signal', byvar_list=['overall'], resid=False,
        )
        result = bt.gen_result()
        elapsed = time.perf_counter() - t0
        warm_times.append(elapsed)
        
        summary = result[0]
        sharpe = summary.iloc[0]['sharpe_ret'] if len(summary) > 0 else 0
        print(f"   Run {i+1}: {elapsed:.1f}s (Sharpe: {sharpe:.2f})")
    
    return cold_total, np.mean(warm_times), len(sig)


def main():
    parser = argparse.ArgumentParser(description="Test BacktestService persistence")
    parser.add_argument("--use-service", action="store_true", 
                        help="Use BacktestService.run_fast() instead of direct")
    parser.add_argument("--snapshot", default=None, help="Snapshot to use")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of warm runs")
    parser.add_argument("--start-year", type=int, default=2012, help="Start year")
    parser.add_argument("--end-year", type=int, default=2021, help="End year")
    args = parser.parse_args()
    
    # Run test
    if args.use_service:
        cold_total, mean_warm, sig_rows = test_with_service(args)
    else:
        cold_total, mean_warm, sig_rows = test_direct(args)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Persistence Benefit")
    print("="*70)
    print(f"")
    print(f"| Metric                | Value   |")
    print(f"|-----------------------|---------|")
    print(f"| Cold start (load+run) | {cold_total:.1f}s   |")
    print(f"| Warm run (per signal) | {mean_warm:.1f}s    |")
    print(f"| Speedup               | {cold_total/mean_warm:.1f}x     |")
    print(f"| Break-even            | 1 signal |")
    print(f"")
    print(f"Signal size: {sig_rows:,} rows ({args.start_year}-{args.end_year})")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
