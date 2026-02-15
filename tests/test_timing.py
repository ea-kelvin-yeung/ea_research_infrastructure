# test_timing.py
# Timing benchmark for backtest engines - measures execution time using REAL data
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Default snapshot and date range
DEFAULT_SNAPSHOT = "2026-02-10-v1"
DEFAULT_START = "2014-01-01"  # 10 years of data
DEFAULT_END = "2023-12-31"


def filter_catalog(catalog: dict, start_date: str, end_date: str) -> dict:
    """Filter catalog data to date range."""
    filtered = catalog.copy()
    filtered['ret'] = catalog['ret'][
        (catalog['ret']['date'] >= start_date) & 
        (catalog['ret']['date'] <= end_date)
    ].copy()
    filtered['risk'] = catalog['risk'][
        (catalog['risk']['date'] >= start_date) & 
        (catalog['risk']['date'] <= end_date)
    ].copy()
    filtered['dates'] = catalog['dates'][
        (catalog['dates']['date'] >= start_date) & 
        (catalog['dates']['date'] <= end_date)
    ].copy()
    # Also filter master if present
    if 'master' in catalog and catalog['master'] is not None:
        md = catalog['master']
        if isinstance(md.index, pd.MultiIndex):
            # MultiIndex with (security_id, date)
            md = md.reset_index()
            md = md[(md['date'] >= start_date) & (md['date'] <= end_date)]
            md = md.set_index(['security_id', 'date'])
        else:
            md = md[(md['date'] >= start_date) & (md['date'] <= end_date)]
        filtered['master'] = md.copy()
    return filtered


def load_real_data(snapshot: str = DEFAULT_SNAPSHOT, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Load real data from a snapshot for timing tests."""
    from poc.catalog import load_catalog
    
    snapshot_path = f"snapshots/{snapshot}"
    print(f"Loading snapshot: {snapshot_path}")
    print(f"Date range: {start} to {end}")
    
    # Load the catalog (this loads ret, risk, dates, and creates master)
    catalog = load_catalog(snapshot_path, universe_only=False)
    
    # Filter to date range
    catalog = filter_catalog(catalog, start, end)
    
    # Extract components
    master = catalog["master"]
    datefile = catalog["dates"]
    ret_df = catalog["ret"]
    risk_df = catalog["risk"]
    
    print(f"Master rows: {len(master):,}")
    print(f"Date range loaded: {datefile['date'].min()} to {datefile['date'].max()}")
    
    # Create a simple signal file for testing (using momentum as signal)
    # Reset index if master has MultiIndex
    if isinstance(master.index, pd.MultiIndex):
        master_flat = master.reset_index()
    else:
        master_flat = master.copy()
    
    # Create signal DataFrame
    sig = master_flat[["security_id", "date"]].copy()
    sig.rename(columns={"date": "date_sig"}, inplace=True)
    
    # Map date_sig -> date_ret (next trading day)
    dates = datefile["date"].values
    n_days = len(dates)
    date_to_n = pd.Series(np.arange(n_days), index=pd.to_datetime(dates))
    
    sig_dates = pd.to_datetime(sig["date_sig"])
    n_sig = date_to_n.reindex(sig_dates).fillna(-1).astype(int).to_numpy()
    n_ret = n_sig + 1
    
    date_ret = np.empty_like(sig["date_sig"].values, dtype="datetime64[ns]")
    date_ret[:] = np.datetime64("NaT")
    ok = (n_ret >= 0) & (n_ret < n_days)
    date_ret[ok] = dates[n_ret[ok]]
    
    sig["date_ret"] = pd.to_datetime(date_ret)
    sig["date_openret"] = sig["date_sig"]
    
    # Use momentum as signal (or random if momentum not available)
    if "momentum" in master_flat.columns:
        rng = np.random.default_rng(42)
        sig["my_signal"] = (0.2 * master_flat["momentum"].to_numpy() + 
                           rng.normal(0, 1, size=len(master_flat))).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        sig["my_signal"] = rng.normal(0, 1, size=len(master_flat)).astype(np.float32)
    
    # Drop rows where date_ret is missing
    sig = sig.dropna(subset=["date_ret"]).reset_index(drop=True)
    
    print(f"Signal rows: {len(sig):,}")
    
    return master_flat, datefile, sig


class TimingResult:
    """Container for timing results with statistics."""
    def __init__(self, name: str, times: list):
        self.name = name
        self.times = np.array(times)
        self.mean = self.times.mean()
        self.min = self.times.min()
        self.max = self.times.max()
        self.std = self.times.std()
    
    def __repr__(self):
        return f"{self.name}: mean={self.mean:.3f}s, min={self.min:.3f}s, max={self.max:.3f}s, std={self.std:.3f}s"


def time_function(fn, name: str = "", repeats: int = 5, warmup: int = 2) -> TimingResult:
    """Time a function and return statistics."""
    # Warmup runs
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return TimingResult(name, times)


def test_timing_old_engine(snapshot: str = DEFAULT_SNAPSHOT, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Test timing for OLD BacktestFast engine using real data."""
    from backtest_engine_fast import BacktestFast
    
    master, datefile, sig = load_real_data(snapshot, start, end)
    
    def run():
        bt = BacktestFast(
            infile=sig,
            retfile=master,
            otherfile=master,
            datefile=datefile,
            sigvar="my_signal",
            method="long_short",
            from_open=False,
            input_type="value",
            weight="equal",
            tc_model="naive",
            byvar_list=["overall"],
            resid=False,
            verbose=False,
            byvix=False,
            earnings_window=False,
        )
        return bt.gen_result()
    
    result = time_function(run, name="OLD BacktestFast", repeats=5, warmup=2)
    print(result)
    return result


def test_timing_new_engine(snapshot: str = DEFAULT_SNAPSHOT, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Test timing for NEW BacktestFastV2 engine using real data."""
    from backtest_engine_minimal_fast import BacktestFastV2
    
    master, datefile, sig = load_real_data(snapshot, start, end)
    
    def run():
        bt = BacktestFastV2(
            infile=sig,
            retfile=master,
            otherfile=master,
            datefile=datefile,
            sigvar="my_signal",
            method="long_short",
            from_open=False,
            input_type="value",
            weight="equal",
            tc_model="naive",
            byvar_list=["overall"],
            resid=False,
            verbose=False,
        )
        return bt.gen_result()
    
    result = time_function(run, name="NEW BacktestFastV2", repeats=5, warmup=2)
    print(result)
    return result


def test_timing_comparison(snapshot: str = DEFAULT_SNAPSHOT, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Compare timing between old and new engines using real data."""
    print("\n" + "=" * 70)
    print("BACKTEST ENGINE TIMING COMPARISON (REAL DATA)")
    print("=" * 70)
    
    master, datefile, sig = load_real_data(snapshot, start, end)
    print("-" * 70)
    
    from backtest_engine_fast import BacktestFast
    from backtest_engine_minimal_fast import BacktestFastV2
    
    bt_params = dict(
        infile=sig,
        retfile=master,
        otherfile=master,
        datefile=datefile,
        sigvar="my_signal",
        method="long_short",
        from_open=False,
        input_type="value",
        weight="equal",
        tc_model="naive",
        byvar_list=["overall"],
        resid=False,
        verbose=False,
    )
    
    def run_old():
        bt = BacktestFast(**bt_params, byvix=False, earnings_window=False)
        return bt.gen_result()
    
    def run_new():
        bt = BacktestFastV2(**bt_params)
        return bt.gen_result()
    
    old_timing = time_function(run_old, name="OLD BacktestFast", repeats=5, warmup=2)
    new_timing = time_function(run_new, name="NEW BacktestFastV2", repeats=5, warmup=2)
    
    print("\nRESULTS:")
    print("-" * 70)
    print(f"{'Engine':<25} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print("-" * 70)
    print(f"{'OLD BacktestFast':<25} {old_timing.mean:>10.3f}s {old_timing.min:>10.3f}s {old_timing.max:>10.3f}s {old_timing.std:>10.3f}s")
    print(f"{'NEW BacktestFastV2':<25} {new_timing.mean:>10.3f}s {new_timing.min:>10.3f}s {new_timing.max:>10.3f}s {new_timing.std:>10.3f}s")
    print("-" * 70)
    
    speedup = old_timing.mean / new_timing.mean
    print(f"\nSpeedup: {speedup:.2f}x faster")
    print(f"Time saved per run: {old_timing.mean - new_timing.mean:.3f}s")
    
    return old_timing, new_timing


def test_timing_with_multiple_byvars(snapshot: str = DEFAULT_SNAPSHOT, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """Test timing with multiple byvars (overall, year, cap) using real data."""
    print("\n" + "=" * 70)
    print("TIMING WITH MULTIPLE BYVARS (REAL DATA)")
    print("=" * 70)
    
    master, datefile, sig = load_real_data(snapshot, start, end)
    print("-" * 70)
    
    from backtest_engine_fast import BacktestFast
    from backtest_engine_minimal_fast import BacktestFastV2
    
    bt_params = dict(
        infile=sig,
        retfile=master,
        otherfile=master,
        datefile=datefile,
        sigvar="my_signal",
        method="long_short",
        from_open=False,
        input_type="value",
        weight="equal",
        tc_model="naive",
        byvar_list=["overall", "year", "cap"],  # Multiple byvars
        resid=False,
        verbose=False,
    )
    
    def run_old():
        bt = BacktestFast(**bt_params, byvix=False, earnings_window=False)
        return bt.gen_result()
    
    def run_new():
        bt = BacktestFastV2(**bt_params)
        return bt.gen_result()
    
    old_timing = time_function(run_old, name="OLD (3 byvars)", repeats=3, warmup=1)
    new_timing = time_function(run_new, name="NEW (3 byvars)", repeats=3, warmup=1)
    
    print(f"\nOLD: {old_timing.mean:.3f}s")
    print(f"NEW: {new_timing.mean:.3f}s")
    print(f"Speedup: {old_timing.mean / new_timing.mean:.2f}x")
    
    return old_timing, new_timing


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Timing tests for backtest engines using real data")
    parser.add_argument("--test", choices=["old", "new", "compare", "byvars", "all"], 
                        default="compare", help="Which test to run")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, 
                        help=f"Data snapshot to use (default: {DEFAULT_SNAPSHOT})")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})")
    args = parser.parse_args()
    
    print(f"\nUsing snapshot: {args.snapshot}")
    print(f"Date range: {args.start} to {args.end} (10 years)")
    print()
    
    if args.test == "old":
        test_timing_old_engine(args.snapshot, args.start, args.end)
    elif args.test == "new":
        test_timing_new_engine(args.snapshot, args.start, args.end)
    elif args.test == "compare":
        test_timing_comparison(args.snapshot, args.start, args.end)
    elif args.test == "byvars":
        test_timing_with_multiple_byvars(args.snapshot, args.start, args.end)
    elif args.test == "all":
        test_timing_comparison(args.snapshot, args.start, args.end)
        test_timing_with_multiple_byvars(args.snapshot, args.start, args.end)
