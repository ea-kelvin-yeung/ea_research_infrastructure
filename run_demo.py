#!/usr/bin/env python
"""
Demo script: Run a signal through the full PoC pipeline.

Usage:
    python run_demo.py

Prerequisites:
    1. Create a snapshot: python -c "from poc.catalog import create_snapshot; create_snapshot()"
    2. Have a signal file ready (or use the sample)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

from poc.catalog import load_catalog, create_snapshot, list_snapshots
from poc.suite import run_suite
from poc.tearsheet import generate_tearsheet
# Note: poc.tracking imported lazily below to avoid mlflow import noise

# Demo date range (3 months for fast testing)
DEMO_START = '2018-01-01'
DEMO_END = '2018-03-31'


def filter_catalog(catalog: dict, start_date: str, end_date: str) -> dict:
    """Filter catalog data to date range for faster testing."""
    filtered = catalog.copy()
    filtered['ret'] = catalog['ret'][
        (catalog['ret']['date'] >= start_date) & 
        (catalog['ret']['date'] <= end_date)
    ].copy()
    filtered['risk'] = catalog['risk'][
        (catalog['risk']['date'] >= start_date) & 
        (catalog['risk']['date'] <= end_date)
    ].copy()
    return filtered


def main():
    print("=" * 60)
    print("Backtest PoC Demo (Fast Mode: 3 months)")
    print("=" * 60)
    
    # Step 1: Ensure snapshot exists
    snapshots = list_snapshots('snapshots')
    if not snapshots:
        print("\n[1/5] Creating snapshot from data/ folder...")
        try:
            create_snapshot('data', 'snapshots')
            snapshots = list_snapshots('snapshots')
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure data/ folder contains ret.parquet, risk.parquet, trading_date.pkl")
            return
    else:
        print(f"\n[1/5] Found existing snapshots: {snapshots}")
    
    snapshot_path = f"snapshots/{snapshots[0]}"
    
    # Step 2: Load and filter catalog
    print(f"\n[2/5] Loading catalog from {snapshot_path}...")
    catalog_full = load_catalog(snapshot_path)
    print(f"  Full data: ret={len(catalog_full['ret']):,}, risk={len(catalog_full['risk']):,}")
    
    # Filter to demo date range
    print(f"  Filtering to {DEMO_START} - {DEMO_END}...")
    catalog = filter_catalog(catalog_full, DEMO_START, DEMO_END)
    print(f"  Filtered: ret={len(catalog['ret']):,}, risk={len(catalog['risk']):,}, dates={len(catalog['dates']):,}")
    
    # Step 3: Load or create sample signal
    print("\n[3/5] Loading sample signal...")
    signal_path = Path('data/signal_sample.pkl')
    if signal_path.exists():
        raw_signal = pd.read_pickle(signal_path)
        print(f"  Loaded raw signal: {len(raw_signal):,} rows")
        
        # Transform to match signal contract
        signal_cols = [c for c in raw_signal.columns if c not in ['security_id', 'date_avail', 'date_sig']]
        if signal_cols:
            sig_col = signal_cols[0]  # Use first signal column
            print(f"  Using signal column: {sig_col}")
            
            signal_df = raw_signal[['security_id', 'date_avail', sig_col]].copy()
            signal_df = signal_df.rename(columns={sig_col: 'signal'})
            
            # Ensure date_avail is datetime
            signal_df['date_avail'] = pd.to_datetime(signal_df['date_avail'])
            
            # Add date_sig (assume signal is known 1 day before available)
            signal_df['date_sig'] = signal_df['date_avail'] - pd.Timedelta(days=1)
            
            # Filter to demo date range
            signal_df = signal_df[
                (signal_df['date_avail'] >= DEMO_START) & 
                (signal_df['date_avail'] <= DEMO_END)
            ]
            
            print(f"  Transformed & filtered signal: {len(signal_df):,} rows")
        else:
            raise ValueError("No signal columns found in sample file")
    else:
        # Create a simple reversal signal as demo
        print("  No sample signal found, generating reversal signal...")
        from poc.baselines import generate_reversal_signal
        signal_df = generate_reversal_signal(catalog, lookback=5, start_date=DEMO_START, end_date=DEMO_END)
        print(f"  Generated reversal signal: {len(signal_df):,} rows")
    
    signal_name = "demo_signal"
    
    # Step 4: Run suite (simplified for demo - fewer configs)
    print("\n[4/5] Running backtest suite...")
    print("  Configs: lag=[0,1] x residualize=[off] (simplified for speed)")
    
    result = run_suite(
        signal_df, 
        catalog,
        grid={'lags': [0, 1], 'residualize': ['off']},  # Simplified
        include_baselines=True,
        n_jobs=1,
        baseline_start_date=DEMO_START,
        baseline_end_date=DEMO_END,
    )
    
    print("\n  Suite Summary:")
    print(result.summary.to_string(index=False))
    
    # Step 5: Generate tearsheet
    print("\n[5/5] Generating tear sheet...")
    Path('artifacts').mkdir(exist_ok=True)
    tearsheet_path = generate_tearsheet(
        result, 
        signal_name, 
        catalog,
        output_path=f"artifacts/{signal_name}_tearsheet.html"
    )
    
    # Optional: Log to MLflow
    try:
        print("\nLogging to MLflow...")
        from poc.tracking import log_run  # Late import to avoid startup noise
        run_id = log_run(result, signal_name, catalog, tearsheet_path)
        print(f"  Run ID: {run_id}")
        print("  View at: http://localhost:5000 (run 'mlflow ui' first)")
    except Exception as e:
        print(f"  MLflow logging skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Tear sheet: {tearsheet_path}")
    print("  To launch UI: streamlit run poc/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
