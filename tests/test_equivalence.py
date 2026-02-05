#!/usr/bin/env python
"""
Equivalence Tests: Verify that BacktestFast produces identical results to Backtest.

This test suite ensures that the optimized backtest engine produces the same
output as the original, within floating-point tolerance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'poc'))

from backtest_engine import Backtest
from backtest_engine_fast import BacktestFast
from poc.catalog import load_catalog, list_snapshots


# Tolerance for floating-point comparison
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance


def load_test_signal():
    """Load the 1-year test signal."""
    signal_path = Path(__file__).parent / 'fixtures' / 'test_signal_1year.parquet'
    if not signal_path.exists():
        if HAS_PYTEST:
            pytest.skip(f"Test signal not found at {signal_path}. Run generate_test_signal.py first.")
        else:
            raise FileNotFoundError(f"Test signal not found at {signal_path}. Run generate_test_signal.py first.")
    return pd.read_parquet(signal_path)


def load_test_catalog():
    """Load catalog for testing."""
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        if HAS_PYTEST:
            pytest.skip("No snapshots found. Create one first.")
        else:
            raise RuntimeError("No snapshots found. Create one first.")
    return load_catalog(f'snapshots/{snapshots[0]}')


def prepare_signal_for_backtest(signal_df, catalog):
    """Prepare signal DataFrame with required columns."""
    # The backtest engine expects date_ret column
    # Map date_avail to trading days
    trading_dates = catalog['dates'][['date', 'n']].sort_values('date')
    
    # Merge to get date_ret (next trading day after date_avail)
    signal_df = signal_df.copy()
    signal_df['date_avail'] = pd.to_datetime(signal_df['date_avail'])
    signal_df['date_sig'] = pd.to_datetime(signal_df['date_sig'])
    
    # Use date_avail as date_ret for simplicity
    signal_df['date_ret'] = signal_df['date_avail']
    
    return signal_df


def run_backtest_original(signal_df, catalog, **kwargs):
    """Run the original Backtest engine."""
    bt = Backtest(
        infile=signal_df,
        retfile=catalog['ret'],
        otherfile=catalog['risk'],
        datefile=catalog['dates'],
        sigvar='signal',
        method='long_short',
        byvar_list=['overall'],  # Simplified for testing
        from_open=False,
        input_type='value',
        mincos=10,
        fractile=[10, 90],
        weight='equal',
        tc_model='naive',
        resid=False,
        output='simple',
        verbose=False,
        **kwargs,
    )
    return bt.gen_result()


def run_backtest_fast(signal_df, catalog, **kwargs):
    """Run the optimized BacktestFast engine."""
    bt = BacktestFast(
        infile=signal_df,
        retfile=catalog['ret'],
        otherfile=catalog['risk'],
        datefile=catalog['dates'],
        sigvar='signal',
        method='long_short',
        byvar_list=['overall'],  # Simplified for testing
        from_open=False,
        input_type='value',
        mincos=10,
        fractile=[10, 90],
        weight='equal',
        tc_model='naive',
        resid=False,
        output='simple',
        verbose=False,
        **kwargs,
    )
    return bt.gen_result()


def compare_dataframes(df1, df2, name, rtol=RTOL, atol=ATOL):
    """Compare two DataFrames for equivalence."""
    # Check shapes
    assert df1.shape == df2.shape, f"{name}: Shape mismatch {df1.shape} vs {df2.shape}"
    
    # Check columns
    assert list(df1.columns) == list(df2.columns), f"{name}: Column mismatch"
    
    # Compare numeric columns
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals1 = df1[col].values
        vals2 = df2[col].values
        
        # Handle NaN values
        nan1 = np.isnan(vals1)
        nan2 = np.isnan(vals2)
        assert np.array_equal(nan1, nan2), f"{name}[{col}]: NaN positions differ"
        
        # Compare non-NaN values
        mask = ~nan1
        if mask.any():
            close = np.allclose(vals1[mask], vals2[mask], rtol=rtol, atol=atol)
            if not close:
                max_diff = np.max(np.abs(vals1[mask] - vals2[mask]))
                max_rel_diff = np.max(np.abs((vals1[mask] - vals2[mask]) / (np.abs(vals1[mask]) + 1e-10)))
                raise AssertionError(
                    f"{name}[{col}]: Values differ. "
                    f"Max absolute diff: {max_diff:.2e}, "
                    f"Max relative diff: {max_rel_diff:.2e}"
                )


# Pytest test class (only defined when pytest is available)
if HAS_PYTEST:
    class TestEquivalence:
        """Test equivalence between original and optimized backtest engines."""
        
        @pytest.fixture(scope="class")
        def signal(self):
            return load_test_signal()
        
        @pytest.fixture(scope="class")
        def catalog(self):
            return load_test_catalog()
        
        @pytest.fixture(scope="class")
        def prepared_signal(self, signal, catalog):
            return prepare_signal_for_backtest(signal, catalog)
        
        def test_basic_backtest(self, prepared_signal, catalog):
            """Test basic backtest without residualization."""
            result_orig = run_backtest_original(prepared_signal, catalog)
            result_fast = run_backtest_fast(prepared_signal, catalog)
            
            # Both should return same structure
            assert len(result_orig) == len(result_fast), "Result tuple length mismatch"
            
            # Compare summary (first element)
            summary_orig, summary_fast = result_orig[0], result_fast[0]
            
            # Sort both by group to ensure same order
            summary_orig = summary_orig.sort_values('group').reset_index(drop=True)
            summary_fast = summary_fast.sort_values('group').reset_index(drop=True)
            
            compare_dataframes(summary_orig, summary_fast, "summary")
            
            # Compare daily stats (second element)
            daily_orig, daily_fast = result_orig[1], result_fast[1]
            
            # Sort by date to ensure same order  
            daily_orig = daily_orig.sort_values('date').reset_index(drop=True)
            daily_fast = daily_fast.sort_values('date').reset_index(drop=True)
            
            compare_dataframes(daily_orig, daily_fast, "daily")
        
        def test_with_residualization_factor(self, prepared_signal, catalog):
            """Test backtest with factor residualization."""
            result_orig = run_backtest_original(
                prepared_signal, catalog, 
                resid=True, resid_style='factor'
            )
            result_fast = run_backtest_fast(
                prepared_signal, catalog,
                resid=True, resid_style='factor'
            )
            
            # Compare summaries
            summary_orig = result_orig[0].sort_values('group').reset_index(drop=True)
            summary_fast = result_fast[0].sort_values('group').reset_index(drop=True)
            
            # Use slightly higher tolerance for residualization (different numerical methods)
            compare_dataframes(summary_orig, summary_fast, "summary_resid", rtol=1e-3)
        
        def test_key_metrics(self, prepared_signal, catalog):
            """Test that key metrics (Sharpe, return, turnover) match."""
            result_orig = run_backtest_original(prepared_signal, catalog)
            result_fast = run_backtest_fast(prepared_signal, catalog)
            
            summary_orig = result_orig[0]
            summary_fast = result_fast[0]
            
            # Extract overall row
            overall_orig = summary_orig[summary_orig['group'] == 'overall'].iloc[0]
            overall_fast = summary_fast[summary_fast['group'] == 'overall'].iloc[0]
            
            # Compare key metrics with tolerance
            metrics = ['ret_ann', 'ret_std', 'sharpe_ret', 'turnover', 'maxdraw']
            for metric in metrics:
                if metric in overall_orig.index:
                    val_orig = overall_orig[metric]
                    val_fast = overall_fast[metric]
                    
                    if not np.isnan(val_orig):
                        rel_diff = abs(val_orig - val_fast) / (abs(val_orig) + 1e-10)
                        assert rel_diff < 1e-3, f"{metric}: {val_orig} vs {val_fast} (rel_diff: {rel_diff:.2e})"


def run_equivalence_check():
    """Run a quick equivalence check (for non-pytest use)."""
    print("=" * 60)
    print("Running Equivalence Check: Original vs Optimized Backtest")
    print("=" * 60)
    
    # Load data
    print("\nLoading test signal...")
    signal = load_test_signal()
    print(f"  Loaded {len(signal):,} rows")
    
    print("\nLoading catalog...")
    catalog = load_test_catalog()
    print("  Catalog loaded")
    
    # Prepare signal
    print("\nPreparing signal...")
    prepared = prepare_signal_for_backtest(signal, catalog)
    print("  Signal prepared")
    
    # Run original
    print("\nRunning ORIGINAL backtest...")
    import time
    t0 = time.time()
    result_orig = run_backtest_original(prepared, catalog)
    time_orig = time.time() - t0
    print(f"  Completed in {time_orig:.2f}s")
    
    # Run fast
    print("\nRunning FAST backtest...")
    t0 = time.time()
    result_fast = run_backtest_fast(prepared, catalog)
    time_fast = time.time() - t0
    print(f"  Completed in {time_fast:.2f}s")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    summary_orig = result_orig[0]
    summary_fast = result_fast[0]
    
    overall_orig = summary_orig[summary_orig['group'] == 'overall'].iloc[0]
    overall_fast = summary_fast[summary_fast['group'] == 'overall'].iloc[0]
    
    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Metric", "Original", "Fast", "Diff"
    ))
    print("-" * 65)
    
    metrics = ['ret_ann', 'ret_std', 'sharpe_ret', 'turnover', 'maxdraw']
    all_match = True
    for metric in metrics:
        if metric in overall_orig.index:
            val_orig = overall_orig[metric]
            val_fast = overall_fast[metric]
            diff = abs(val_orig - val_fast)
            match = diff < 1e-6
            all_match = all_match and match
            status = "OK" if match else "DIFF"
            print(f"{metric:<20} {val_orig:>15.6f} {val_fast:>15.6f} {diff:>15.2e} {status}")
    
    print("\n" + "=" * 60)
    if all_match:
        print("EQUIVALENCE CHECK PASSED")
    else:
        print("EQUIVALENCE CHECK FAILED - Results differ beyond tolerance")
    print("=" * 60)
    
    # Speedup
    speedup = time_orig / time_fast if time_fast > 0 else float('inf')
    print(f"\nPerformance: {speedup:.2f}x speedup ({time_orig:.2f}s -> {time_fast:.2f}s)")
    
    return all_match


if __name__ == '__main__':
    success = run_equivalence_check()
    sys.exit(0 if success else 1)
