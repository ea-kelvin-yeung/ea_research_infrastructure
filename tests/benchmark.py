#!/usr/bin/env python
"""
Benchmark Script: Measure and compare performance of original vs optimized backtest engine.

Runs multiple iterations with different configurations and reports timing statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import statistics

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'poc'))

from backtest_engine import Backtest
from backtest_engine_fast import BacktestFast
from poc.catalog import load_catalog, list_snapshots


def load_test_signal():
    """Load the 1-year test signal."""
    signal_path = Path(__file__).parent / 'fixtures' / 'test_signal_1year.parquet'
    if not signal_path.exists():
        raise FileNotFoundError(f"Test signal not found at {signal_path}. Run generate_test_signal.py first.")
    return pd.read_parquet(signal_path)


def load_catalog_data():
    """Load catalog for benchmarking."""
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        raise RuntimeError("No snapshots found. Create one first.")
    return load_catalog(f'snapshots/{snapshots[0]}')


def prepare_signal_for_backtest(signal_df, catalog):
    """Prepare signal DataFrame with required columns."""
    signal_df = signal_df.copy()
    signal_df['date_avail'] = pd.to_datetime(signal_df['date_avail'])
    signal_df['date_sig'] = pd.to_datetime(signal_df['date_sig'])
    signal_df['date_ret'] = signal_df['date_avail']
    return signal_df


def run_benchmark(
    signal_df: pd.DataFrame,
    catalog: dict,
    engine_class,
    engine_name: str,
    n_runs: int = 3,
    **kwargs
) -> Dict:
    """
    Run benchmark for a specific engine.
    
    Returns dict with timing statistics.
    """
    times = []
    
    # Default config
    default_config = {
        'sigvar': 'signal',
        'method': 'long_short',
        'byvar_list': ['overall'],
        'from_open': False,
        'input_type': 'value',
        'mincos': 10,
        'fractile': [10, 90],
        'weight': 'equal',
        'tc_model': 'naive',
        'resid': False,
        'output': 'simple',
        'verbose': False,
    }
    # Apply kwargs overrides
    default_config.update(kwargs)
    
    for i in range(n_runs):
        # Create new instance each time to avoid caching
        bt = engine_class(
            infile=signal_df.copy(),
            retfile=catalog['ret'].copy(),
            otherfile=catalog['risk'].copy(),
            datefile=catalog['dates'].copy(),
            **default_config,
        )
        
        t0 = time.perf_counter()
        result = bt.gen_result()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
        # Validate result is not empty
        assert len(result) >= 2, f"Invalid result from {engine_name}"
    
    return {
        'engine': engine_name,
        'n_runs': n_runs,
        'times': times,
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }


def run_benchmark_suite(
    signal_df: pd.DataFrame,
    catalog: dict,
    n_runs: int = 3,
    configurations: List[Dict] = None
) -> List[Tuple[Dict, Dict]]:
    """
    Run benchmark suite with multiple configurations.
    
    Returns list of (config, {original_stats, fast_stats}) tuples.
    """
    if configurations is None:
        configurations = [
            {'name': 'basic', 'resid': False},
            {'name': 'resid_factor', 'resid': True, 'resid_style': 'factor'},
        ]
    
    results = []
    
    for config in configurations:
        config_name = config.pop('name')
        print(f"\n  Configuration: {config_name}")
        
        # Original
        print(f"    Running Original ({n_runs} runs)...", end=" ", flush=True)
        orig_stats = run_benchmark(
            signal_df, catalog, Backtest, 'original', n_runs, **config
        )
        print(f"done ({orig_stats['mean']:.2f}s)")
        
        # Fast
        print(f"    Running Fast ({n_runs} runs)...", end=" ", flush=True)
        fast_stats = run_benchmark(
            signal_df, catalog, BacktestFast, 'fast', n_runs, **config
        )
        print(f"done ({fast_stats['mean']:.2f}s)")
        
        # Restore name
        config['name'] = config_name
        results.append((config, {'original': orig_stats, 'fast': fast_stats}))
    
    return results


def print_benchmark_report(results: List[Tuple[Dict, Dict]]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80)
    
    print("\n{:<20} {:>12} {:>12} {:>12} {:>10}".format(
        "Configuration", "Original", "Fast", "Speedup", "Status"
    ))
    print("-" * 80)
    
    total_orig_time = 0
    total_fast_time = 0
    
    for config, stats in results:
        config_name = config['name']
        orig_mean = stats['original']['mean']
        fast_mean = stats['fast']['mean']
        speedup = orig_mean / fast_mean if fast_mean > 0 else float('inf')
        
        total_orig_time += orig_mean
        total_fast_time += fast_mean
        
        status = "OK" if speedup > 1 else "SLOWER"
        
        print("{:<20} {:>10.3f}s {:>10.3f}s {:>10.2f}x {:>10}".format(
            config_name, orig_mean, fast_mean, speedup, status
        ))
    
    print("-" * 80)
    
    overall_speedup = total_orig_time / total_fast_time if total_fast_time > 0 else float('inf')
    print("{:<20} {:>10.3f}s {:>10.3f}s {:>10.2f}x".format(
        "TOTAL", total_orig_time, total_fast_time, overall_speedup
    ))
    
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    for config, stats in results:
        print(f"\n{config['name']}:")
        for engine in ['original', 'fast']:
            s = stats[engine]
            print(f"  {engine}: mean={s['mean']:.3f}s, std={s['std']:.3f}s, "
                  f"min={s['min']:.3f}s, max={s['max']:.3f}s")


def main():
    """Main benchmark entry point."""
    print("=" * 80)
    print("BACKTEST ENGINE PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark backtest engines')
    parser.add_argument('-n', '--runs', type=int, default=3, help='Number of runs per configuration')
    parser.add_argument('--resid', action='store_true', help='Include residualization benchmarks')
    args = parser.parse_args()
    
    # Load data
    print("\nLoading data...")
    signal = load_test_signal()
    print(f"  Test signal: {len(signal):,} rows")
    
    catalog = load_catalog_data()
    print("  Catalog loaded")
    
    # Prepare signal
    print("\nPreparing signal...")
    prepared = prepare_signal_for_backtest(signal, catalog)
    print("  Signal prepared")
    
    # Define configurations
    configurations = [{'name': 'basic', 'resid': False}]
    if args.resid:
        configurations.append({'name': 'resid_factor', 'resid': True, 'resid_style': 'factor'})
    
    # Run benchmarks
    print(f"\nRunning benchmarks ({args.runs} runs each)...")
    results = run_benchmark_suite(
        prepared, catalog, 
        n_runs=args.runs,
        configurations=configurations
    )
    
    # Print report
    print_benchmark_report(results)
    
    # Summary
    _, basic_stats = results[0]
    speedup = basic_stats['original']['mean'] / basic_stats['fast']['mean']
    
    print("\n" + "=" * 80)
    if speedup > 1.5:
        print(f"SUCCESS: Achieved {speedup:.2f}x speedup")
    elif speedup > 1.0:
        print(f"MARGINAL: Achieved {speedup:.2f}x speedup (target: >1.5x)")
    else:
        print(f"REGRESSION: Optimized version is {1/speedup:.2f}x SLOWER")
    print("=" * 80)
    
    return speedup > 1.0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
