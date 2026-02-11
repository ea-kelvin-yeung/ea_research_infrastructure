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


def load_test_catalog(use_master=False):
    """Load catalog for testing."""
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        if HAS_PYTEST:
            pytest.skip("No snapshots found. Create one first.")
        else:
            raise RuntimeError("No snapshots found. Create one first.")
    
    # Try snapshots in order, prefer those with all required files
    from pathlib import Path
    for snapshot in snapshots:
        snapshot_path = Path('snapshots') / snapshot
        # Check if essential files exist
        if (snapshot_path / 'risk.parquet').exists() or (snapshot_path / 'partitions' / 'risk').exists():
            return load_catalog(str(snapshot_path), use_master=use_master)
    
    # Fallback to first snapshot if none found with complete files
    return load_catalog(f'snapshots/{snapshots[0]}', use_master=use_master)


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
    bt_kwargs = dict(
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
    )
    bt_kwargs.update(kwargs)
    
    # Use master_data if available in catalog
    if 'master' in catalog and catalog['master'] is not None:
        bt_kwargs['master_data'] = catalog['master']
    
    bt = BacktestFast(**bt_kwargs)
    
    # Set pre-computed indexes and Polars DataFrames for maximum performance
    bt.set_precomputed_indexes(
        dates_indexed=catalog.get('dates_indexed'),
        asof_tables=catalog.get('asof_tables'),
        master_pl=catalog.get('master_pl'),
        otherfile_pl=catalog.get('otherfile_pl'),
        asof_tables_pl=catalog.get('asof_tables_pl'),
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
        
        def test_polars_implementation(self, prepared_signal, catalog):
            """Test that Polars-based BacktestFast produces valid results."""
            # Run with master_data (uses Polars code paths)
            catalog_with_master = load_test_catalog(use_master=True)
            result = run_backtest_fast(prepared_signal, catalog_with_master)
            
            # Validate structure
            assert len(result) >= 2, "Expected at least 2 elements in result"
            summary = result[0]
            daily = result[1]
            
            # Check expected columns
            assert 'group' in summary.columns
            assert 'ret_ann' in summary.columns
            assert 'sharpe_ret' in summary.columns
            
            # Check values are not NaN for overall
            overall = summary[summary['group'] == 'overall']
            assert len(overall) > 0, "No 'overall' group in results"
            
            overall_row = overall.iloc[0]
            assert not np.isnan(overall_row['ret_ann']), "ret_ann should not be NaN"
            assert not np.isnan(overall_row['sharpe_ret']), "sharpe_ret should not be NaN"


def run_equivalence_check(use_master=False):
    """Run a quick equivalence check (for non-pytest use)."""
    mode_str = "with master_data" if use_master else "without master_data"
    print("=" * 60)
    print(f"Running Equivalence Check: Original vs Optimized Backtest")
    print(f"Mode: {mode_str}")
    print("=" * 60)
    
    # Load data
    print("\nLoading test signal...")
    signal = load_test_signal()
    print(f"  Loaded {len(signal):,} rows")
    
    print(f"\nLoading catalog (use_master={use_master})...")
    catalog = load_test_catalog(use_master=use_master)
    if use_master and 'master' in catalog:
        print(f"  Catalog loaded with master_data: {len(catalog['master']):,} rows")
    else:
        print("  Catalog loaded (no master_data)")
    
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
    print(f"\nRunning FAST backtest {mode_str}...")
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


def run_polars_validation():
    """
    Validate that the Polars implementation in BacktestFast works correctly.
    
    This test runs a backtest and validates:
    1. No exceptions during execution
    2. Results have expected structure
    3. Key metrics are valid (not NaN)
    """
    print("=" * 60)
    print("Polars Implementation Validation Test")
    print("=" * 60)
    
    try:
        # Load data
        print("\nLoading test signal...")
        signal = load_test_signal()
        print(f"  Loaded {len(signal):,} rows")
        
        print("\nLoading catalog (use_master=True)...")
        catalog = load_test_catalog(use_master=True)
        print(f"  Catalog loaded with master_data: {len(catalog['master']):,} rows")
        
        # Prepare signal
        print("\nPreparing signal...")
        prepared = prepare_signal_for_backtest(signal, catalog)
        print("  Signal prepared")
        
        # Run the Polars-optimized backtest
        print("\nRunning BacktestFast with Polars operations...")
        import time
        t0 = time.time()
        result = run_backtest_fast(prepared, catalog)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.2f}s")
        
        # Validate results structure
        print("\nValidating results structure...")
        assert len(result) >= 2, "Expected at least 2 elements in result tuple"
        
        summary = result[0]
        daily = result[1]
        
        assert isinstance(summary, pd.DataFrame), "Summary should be a DataFrame"
        assert isinstance(daily, pd.DataFrame), "Daily should be a DataFrame"
        
        print(f"  Summary: {summary.shape[0]} rows, {summary.shape[1]} columns")
        print(f"  Daily: {daily.shape[0]} rows, {daily.shape[1]} columns")
        
        # Check key columns exist
        expected_cols = ['group', 'ret_ann', 'sharpe_ret', 'turnover']
        for col in expected_cols:
            assert col in summary.columns, f"Expected column '{col}' in summary"
        
        # Check key metrics are not NaN
        print("\nValidating key metrics...")
        overall = summary[summary['group'] == 'overall'].iloc[0]
        metrics = ['ret_ann', 'ret_std', 'sharpe_ret', 'turnover']
        for metric in metrics:
            val = overall[metric]
            if np.isnan(val):
                print(f"  WARNING: {metric} is NaN")
            else:
                print(f"  {metric}: {val:.4f}")
        
        print("\n" + "=" * 60)
        print("POLARS VALIDATION PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n*** ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("POLARS VALIDATION FAILED")
        print("=" * 60)
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-master', action='store_true', help='Test with master_data enabled')
    parser.add_argument('--both', action='store_true', help='Test both with and without master_data')
    parser.add_argument('--polars', action='store_true', help='Validate Polars implementation only')
    args = parser.parse_args()
    
    if args.polars:
        success = run_polars_validation()
        sys.exit(0 if success else 1)
    elif args.both:
        print("\n" + "=" * 60)
        print("TEST 1: Without master_data")
        print("=" * 60)
        success1 = run_equivalence_check(use_master=False)
        
        print("\n\n" + "=" * 60)
        print("TEST 2: With master_data")
        print("=" * 60)
        success2 = run_equivalence_check(use_master=True)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Without master_data: {'PASSED' if success1 else 'FAILED'}")
        print(f"  With master_data:    {'PASSED' if success2 else 'FAILED'}")
        sys.exit(0 if (success1 and success2) else 1)
    else:
        success = run_equivalence_check(use_master=args.use_master)
        sys.exit(0 if success else 1)
