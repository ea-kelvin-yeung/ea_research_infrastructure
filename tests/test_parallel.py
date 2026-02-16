"""
Test parallel backtest execution with joblib.

Measures speedup from running multiple signals in parallel.
Workers read signal CSV directly to avoid DataFrame serialization/copying.

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

SNAPSHOT = "2026-02-10-v1"
START_DATE = "2012-01-01"
END_DATE = "2021-12-31"  # 10 years
SIGNAL_FILE = "data/reversal_signal_analyst.csv"


def load_signal_from_csv(signal_path: str, start_date: str, end_date: str, noise_seed: int = None) -> pd.DataFrame:
    """
    Load signal from CSV file directly.
    Each worker reads the file - avoids DataFrame serialization/copying.
    Optionally adds noise for variant signals.
    """
    sig = pd.read_csv(signal_path)
    sig['date_sig'] = pd.to_datetime(sig['date_sig'])
    sig['date_avail'] = pd.to_datetime(sig['date_avail'])
    
    # Filter to date range
    sig = sig[
        (sig['date_sig'] >= start_date) &
        (sig['date_sig'] <= end_date)
    ].copy()
    
    # Add date_ret (next day - will be aligned by engine)
    sig['date_ret'] = sig['date_sig'] + pd.Timedelta(days=1)
    
    # Add noise if seed provided (for variant signals)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        sig['signal'] = sig['signal'] + rng.normal(0, 0.01, size=len(sig))
    
    return sig


def run_backtest_from_csv(noise_seed: int = None):
    """
    Worker function: reads signal CSV directly, runs backtest.
    No DataFrame passed in = no serialization overhead.
    """
    from api import BacktestService
    
    # Get service (already loaded in main thread, shared via threading)
    service = BacktestService.get(
        SNAPSHOT,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    # Each worker reads CSV directly - no data copying
    signal = load_signal_from_csv(SIGNAL_FILE, START_DATE, END_DATE, noise_seed)
    
    result = service.run(signal, sigvar='signal', byvar_list=['overall'])
    return result[0].iloc[0]['sharpe_ret']


def main():
    parser = argparse.ArgumentParser(description="Test parallel backtest execution")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--n-signals", type=int, default=4, help="Number of signals to test")
    args = parser.parse_args()
    
    print("="*70)
    print(f"PARALLEL BACKTEST TEST (joblib) - 10 years of data")
    print(f"  Workers: {args.n_jobs}, Signals: {args.n_signals}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Signal file: {SIGNAL_FILE}")
    print("="*70)
    
    # Load master data via BacktestService (shared by all threads)
    print("\nLoading master data...")
    from api import BacktestService
    BacktestService.reset()
    t0 = time.perf_counter()
    service = BacktestService.get(SNAPSHOT, start_date=START_DATE, end_date=END_DATE)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Master rows: {len(service._master):,}")
    
    # Check signal file exists and show info
    base_signal = load_signal_from_csv(SIGNAL_FILE, START_DATE, END_DATE)
    print(f"\nSignal info:")
    print(f"  Rows: {len(base_signal):,}")
    print(f"  Workers will read CSV directly (no DataFrame copying)")
    
    # Create noise seeds for variant signals
    noise_seeds = [None] + [42 + i for i in range(1, args.n_signals)]
    print(f"  Created {len(noise_seeds)} signal variants (1 base + {args.n_signals - 1} with noise)")
    
    # Sequential baseline
    print(f"\n1. SEQUENTIAL ({args.n_signals} signals):")
    t0 = time.perf_counter()
    seq_results = []
    for seed in noise_seeds:
        sharpe = run_backtest_from_csv(seed)
        seq_results.append(sharpe)
    seq_time = time.perf_counter() - t0
    print(f"   Time: {seq_time:.1f}s ({seq_time/args.n_signals:.2f}s per signal)")
    print(f"   Sharpes: {[f'{s:.2f}' for s in seq_results]}")
    
    # Parallel with joblib (threading backend - shares memory)
    print(f"\n2. PARALLEL ({args.n_signals} signals, {args.n_jobs} workers):")
    from joblib import Parallel, delayed
    
    # DON'T reset - threads share the already-loaded service
    # Workers read CSV directly - no DataFrame serialization
    print("   Threading backend + CSV read per worker (zero copy)...")
    t0 = time.perf_counter()
    par_results = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(run_backtest_from_csv)(seed) for seed in noise_seeds
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
    print(f"Notes:")
    print(f"  - Threading backend shares master data in memory")
    print(f"  - Each worker reads signal CSV directly (no DataFrame serialization)")
    print(f"  - GIL limits true parallelism for CPU-bound Python code")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
