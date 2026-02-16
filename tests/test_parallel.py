"""
Test parallel backtest execution with joblib.

Measures speedup from running multiple signals in parallel.
Uses same setup as test_server.py.

Run:
    python tests/test_parallel.py
    python tests/test_parallel.py --n-jobs 4 --n-signals 8
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SIGNAL_FILE = "data/reversal_signal_analyst.csv"
SNAPSHOT = "2026-02-10-v1"
# Use 2 years to reduce signal size (1.3M rows vs 6.4M rows)
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"


def load_signal():
    """Load and prepare signal data."""
    sig = pd.read_csv(SIGNAL_FILE)
    sig['date_sig'] = pd.to_datetime(sig['date_sig'])
    sig['date_avail'] = pd.to_datetime(sig['date_avail'])
    sig = sig[
        (sig['date_sig'] >= START_DATE) &
        (sig['date_sig'] <= END_DATE)
    ]
    sig['date_ret'] = sig['date_sig'] + pd.Timedelta(days=1)
    return sig


def run_backtest(signal_with_noise):
    """Run a single backtest (worker function)."""
    from api import BacktestService
    
    # Get service (loads data if not already loaded in this process)
    service = BacktestService.get(
        SNAPSHOT,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    result = service.run(signal_with_noise, sigvar='signal', byvar_list=['overall'])
    return result[0].iloc[0]['sharpe_ret']


def main():
    parser = argparse.ArgumentParser(description="Test parallel backtest execution")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--n-signals", type=int, default=4, help="Number of signals to test")
    args = parser.parse_args()
    
    print("="*70)
    print(f"PARALLEL BACKTEST TEST (joblib)")
    print(f"  Workers: {args.n_jobs}, Signals: {args.n_signals}")
    print("="*70)
    
    # Load base signal
    base_signal = load_signal()
    print(f"\nBase signal: {len(base_signal):,} rows")
    
    # Create variant signals (add small noise to make them different)
    signals = []
    for i in range(args.n_signals):
        sig = base_signal.copy()
        sig['signal'] = sig['signal'] + np.random.randn(len(sig)) * 0.01
        signals.append(sig)
    print(f"Created {len(signals)} signal variants")
    
    # Warm up: load data in main process first
    print("\nWarm-up: Loading data in main process...")
    from api import BacktestService
    BacktestService.reset()
    t0 = time.perf_counter()
    service = BacktestService.get(SNAPSHOT, start_date=START_DATE, end_date=END_DATE)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")
    
    # Sequential baseline
    print(f"\n1. SEQUENTIAL ({args.n_signals} signals):")
    t0 = time.perf_counter()
    seq_results = []
    for sig in signals:
        sharpe = run_backtest(sig)
        seq_results.append(sharpe)
    seq_time = time.perf_counter() - t0
    print(f"   Time: {seq_time:.1f}s ({seq_time/args.n_signals:.2f}s per signal)")
    print(f"   Sharpes: {[f'{s:.2f}' for s in seq_results]}")
    
    # Parallel with joblib (threading backend - shares memory)
    print(f"\n2. PARALLEL ({args.n_signals} signals, {args.n_jobs} workers):")
    from joblib import Parallel, delayed
    
    # DON'T reset - threads share the already-loaded service
    # Use threading backend to share memory and avoid data copying
    print("   Using threading backend (shared memory, no data reload)...")
    t0 = time.perf_counter()
    par_results = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(run_backtest)(sig) for sig in signals
    )
    par_time = time.perf_counter() - t0
    print(f"   Time: {par_time:.1f}s ({par_time/args.n_signals:.2f}s per signal)")
    print(f"   Sharpes: {[f'{s:.2f}' for s in par_results]}")
    
    # Summary
    speedup = seq_time / par_time
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"")
    print(f"| Metric | Sequential | Parallel ({args.n_jobs}w) |")
    print(f"|--------|------------|------------|")
    print(f"| Total time | {seq_time:.1f}s | {par_time:.1f}s |")
    print(f"| Per signal | {seq_time/args.n_signals:.2f}s | {par_time/args.n_signals:.2f}s |")
    print(f"| Speedup | 1.0x | **{speedup:.1f}x** |")
    print(f"")
    print(f"Note: Threading backend shares memory - no data copying or reload")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
