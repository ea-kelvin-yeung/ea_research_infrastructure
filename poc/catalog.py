"""
Data Catalog: Load data from snapshots with manifest tracking.

Includes:
- master_data: pre-merged ret+risk DataFrame for fast backtest joins
- factors: pre-indexed risk factor DataFrame for fast factor correlation

Performance optimizations:
- All data is deduplicated on join keys (security_id, date) to enable fast joins
- security_id is converted to int32 for efficient hashing
- Group columns (cap, industry_id, sector_id) are converted to category
- Dates are normalized to datetime64[ns] without timezone
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# Standard risk factors used for signal quality assessment
RISK_FACTORS = ['size', 'value', 'growth', 'leverage', 'volatility', 'momentum']


def _normalize_dataframe(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Normalize DataFrame for optimal merge performance.
    
    1. Deduplicate on (security_id, date) - keeps last
    2. Convert security_id to int32
    3. Convert group columns to category
    4. Normalize dates to datetime64[ns]
    """
    df = df.copy()
    
    # 1. Deduplicate on join keys
    if 'security_id' in df.columns and 'date' in df.columns:
        original_len = len(df)
        df = df.sort_values(['security_id', 'date']).drop_duplicates(
            ['security_id', 'date'], keep='last'
        )
        if len(df) < original_len:
            print(f"  {name}: deduplicated {original_len - len(df)} rows")
    
    # 2. Convert security_id to int32 (cheaper hashing)
    if 'security_id' in df.columns:
        df['security_id'] = df['security_id'].astype(np.int32)
    
    # 3. Convert group columns to category (faster groupby/factorize)
    category_cols = ['cap', 'industry_id', 'sector_id']
    for col in category_cols:
        if col in df.columns and df[col].dtype != 'category':
            df[col] = df[col].astype('category')
    
    # 4. Normalize dates to datetime64[ns] (no timezone)
    date_cols = ['date', 'date_sig', 'date_avail', 'date_ret', 'date_openret']
    for col in date_cols:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
            # Remove timezone if present
            if hasattr(df[col].dtype, 'tz') and df[col].dtype.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    
    return df


def _normalize_datefile(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize datefile - deduplicate on date."""
    df = df.copy()
    
    # Deduplicate on date
    if 'date' in df.columns:
        original_len = len(df)
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        if len(df) < original_len:
            print(f"  datefile: deduplicated {original_len - len(df)} rows")
    
    # Normalize date column
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
    
    # Ensure n is int32
    if 'n' in df.columns:
        df['n'] = df['n'].astype(np.int32)
    
    return df


def _create_master_data(ret_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-merge ret and risk DataFrames into a single master DataFrame.
    
    This eliminates repeated merges during backtest (13+ seconds -> ~1 second).
    The master data is indexed on (security_id, date) for fast joins.
    
    Columns included:
    - From ret: ret, resret, openret, resopenret, vol, adv, close_adj
    - From risk: mcap, cap, industry_id, sector_id, size, value, growth, 
                 leverage, volatility, momentum, yield
    """
    # Select columns from ret
    ret_cols = ['security_id', 'date', 'ret', 'resret']
    optional_ret_cols = ['openret', 'resopenret', 'vol', 'adv', 'close_adj']
    for col in optional_ret_cols:
        if col in ret_df.columns:
            ret_cols.append(col)
    
    # Select columns from risk
    risk_cols = ['security_id', 'date']
    optional_risk_cols = ['mcap', 'adv', 'cap', 'industry_id', 'sector_id', 
                          'size', 'value', 'growth', 'leverage', 
                          'volatility', 'momentum', 'yield']
    for col in optional_risk_cols:
        if col in risk_df.columns:
            risk_cols.append(col)
    
    # Handle adv column naming conflict (may exist in both)
    ret_subset = ret_df[ret_cols].copy()
    risk_subset = risk_df[risk_cols].copy()
    
    if 'adv' in ret_subset.columns and 'adv' in risk_subset.columns:
        # Keep ret's adv, drop risk's
        risk_subset = risk_subset.drop(columns=['adv'])
    
    # Merge
    master = ret_subset.merge(
        risk_subset,
        on=['security_id', 'date'],
        how='inner'
    )
    
    # Set index for fast joins
    master = master.set_index(['security_id', 'date']).sort_index()
    
    return master


def _create_factor_data(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pre-indexed factor DataFrame for fast factor correlation.
    
    This speeds up factor exposure computation in suite.py by:
    1. Selecting only the factor columns needed
    2. Pre-indexing by (security_id, date) for fast joins
    
    Columns included:
    - security_id, date (as index)
    - size, value, growth, leverage, volatility, momentum
    """
    # Select available factor columns
    cols = ['security_id', 'date']
    for factor in RISK_FACTORS:
        if factor in risk_df.columns:
            cols.append(factor)
    
    if len(cols) <= 2:
        # No factor columns available
        return pd.DataFrame()
    
    # Create subset and index
    factors = risk_df[cols].copy()
    factors = factors.set_index(['security_id', 'date']).sort_index()
    
    return factors


def load_catalog(snapshot_path: str = "snapshots/default", use_master: bool = True) -> dict:
    """
    Load data catalog from a snapshot directory.
    
    Fast load path: All heavy lifting was done at snapshot creation time.
    This function just reads pre-computed files and restores indexes.
    
    Args:
        snapshot_path: Path to snapshot directory containing parquet files
        use_master: If True, load master_data for fast backtest joins
        
    Returns:
        dict with keys: 'ret', 'risk', 'dates', 'snapshot_id', 'manifest'
        If use_master=True, also includes:
        - 'master': pre-merged ret+risk indexed by (security_id, date)
        - 'dates_indexed': dict with 'by_date' and 'by_n' indexed DataFrames
        - 'asof_tables': dict with pre-sorted tables for merge_asof
    """
    path = Path(snapshot_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    
    # Load manifest first to check version
    manifest_path = path / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {'snapshot_id': path.name, 'created_at': 'unknown', 'version': 1}
    
    # Check if this is a v2 snapshot (all precomputation done at snapshot time)
    is_v2 = manifest.get('version', 1) >= 2
    
    # Load base data files (already normalized in v2 snapshots)
    ret_df = pd.read_parquet(path / 'ret.parquet')
    risk_df = pd.read_parquet(path / 'risk.parquet')
    dates_df = pd.read_parquet(path / 'trading_date.parquet')
    
    # For v1 snapshots, normalize on load (backwards compatibility)
    if not is_v2:
        ret_df = _normalize_dataframe(ret_df, 'ret')
        risk_df = _normalize_dataframe(risk_df, 'risk')
        dates_df = _normalize_datefile(dates_df)
    
    catalog = {
        'ret': ret_df,
        'risk': risk_df,
        'dates': dates_df,
        'snapshot_id': path.name,
        'snapshot_path': str(path.resolve()),
        'manifest': manifest,
    }
    
    if use_master:
        # Load master_data (pre-merged ret+risk)
        master_path = path / 'master.parquet'
        if master_path.exists():
            master_df = pd.read_parquet(master_path)
            # Restore MultiIndex (cheap operation - skip sort_index for speed)
            if 'security_id' in master_df.columns:
                master_df = master_df.set_index(['security_id', 'date'])
            catalog['master'] = master_df
        elif not is_v2:
            # Fallback for v1: create master on the fly
            catalog['master'] = _create_master_data(ret_df, risk_df)
            catalog['master'].reset_index().to_parquet(master_path, index=False)
            print(f"Created master.parquet: {len(catalog['master'])} rows")
        
        # Load factors (pre-indexed)
        factors_path = path / 'factors.parquet'
        if factors_path.exists():
            factors_df = pd.read_parquet(factors_path)
            if 'security_id' in factors_df.columns:
                factors_df = factors_df.set_index(['security_id', 'date'])
            catalog['factors'] = factors_df
        elif not is_v2:
            catalog['factors'] = _create_factor_data(risk_df)
            if len(catalog['factors']) > 0:
                catalog['factors'].reset_index().to_parquet(factors_path, index=False)
        
        # Create indexed dates (small data, fast to create in-memory)
        catalog['dates_indexed'] = {
            'by_date': dates_df.set_index('date').sort_index(),
            'by_n': dates_df.set_index('n').sort_index() if 'n' in dates_df.columns else None,
        }
        
        # Load pre-sorted asof tables (or create for v1 compatibility)
        asof_resid_path = path / 'asof_resid.parquet'
        asof_cap_path = path / 'asof_cap.parquet'
        
        if asof_resid_path.exists():
            resid_table = pd.read_parquet(asof_resid_path)
        else:
            # Fallback: create on the fly
            resid_cols = ["security_id", "date", "industry_id", "sector_id",
                          "size", "value", "growth", "leverage", "volatility", "momentum"]
            if 'yield' in risk_df.columns:
                resid_cols.append('yield')
            available = [c for c in resid_cols if c in risk_df.columns]
            resid_table = risk_df[available].copy()
            resid_table = resid_table.rename(columns={"date": "date_sig"})
            if 'yield' in resid_table.columns:
                resid_table = resid_table.rename(columns={"yield": "yields"})
            resid_table = resid_table.sort_values("date_sig")
        
        if asof_cap_path.exists():
            cap_table = pd.read_parquet(asof_cap_path)
        else:
            cap_cols = ["security_id", "date", "cap"]
            available = [c for c in cap_cols if c in risk_df.columns]
            cap_table = risk_df[available].copy()
            cap_table = cap_table.rename(columns={"date": "date_sig"})
            cap_table = cap_table.sort_values("date_sig")
        
        catalog['asof_tables'] = {
            'resid': resid_table,
            'byvars_cap': cap_table,
        }
    
    return catalog


def create_snapshot(
    source_dir: str = 'data',
    output_dir: str = 'snapshots',
    snapshot_id: Optional[str] = None,
) -> Path:
    """
    Create a snapshot from existing data files with ALL precomputation done upfront.
    
    This is where ALL heavy lifting happens:
    1. Normalize data (deduplicate, convert dtypes for fast hashing)
    2. Create master_data (pre-merged ret+risk)
    3. Create pre-sorted asof tables for merge_asof
    4. Create factors table
    
    After this, load_catalog() is just fast file reads.
    
    Args:
        source_dir: Directory containing ret.parquet, risk.parquet, trading_date.pkl
        output_dir: Directory to store snapshots
        snapshot_id: Optional custom snapshot ID (default: date-based)
        
    Returns:
        Path to created snapshot directory
    """
    import time
    start_time = time.time()
    
    source = Path(source_dir)
    snapshot_id = snapshot_id or f"{datetime.now().strftime('%Y-%m-%d')}-v1"
    snapshot_path = Path(output_dir) / snapshot_id
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating snapshot: {snapshot_id}")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Load raw data
    # =========================================================================
    print("Step 1: Loading raw data...")
    step_start = time.time()
    
    ret_raw = pd.read_parquet(source / 'ret.parquet')
    risk_raw = pd.read_parquet(source / 'risk.parquet')
    
    date_file = source / 'trading_date.pkl'
    if date_file.exists():
        dates_raw = pd.read_pickle(date_file)
    else:
        dates_raw = pd.read_parquet(source / 'trading_date.parquet')
    
    print(f"  Loaded: ret={len(ret_raw):,} rows, risk={len(risk_raw):,} rows, dates={len(dates_raw):,} rows")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 2: Normalize data (deduplicate, convert dtypes)
    # =========================================================================
    print("\nStep 2: Normalizing data...")
    step_start = time.time()
    
    ret_df = _normalize_dataframe(ret_raw, 'ret')
    risk_df = _normalize_dataframe(risk_raw, 'risk')
    datefile = _normalize_datefile(dates_raw)
    
    print(f"  After dedup: ret={len(ret_df):,} rows, risk={len(risk_df):,} rows")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 3: Save normalized base files
    # =========================================================================
    print("\nStep 3: Saving normalized base files...")
    step_start = time.time()
    
    ret_df.to_parquet(snapshot_path / 'ret.parquet', index=False)
    risk_df.to_parquet(snapshot_path / 'risk.parquet', index=False)
    datefile.to_parquet(snapshot_path / 'trading_date.parquet', index=False)
    
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 4: Create master_data (pre-merged ret+risk, indexed)
    # =========================================================================
    print("\nStep 4: Creating master_data (pre-merged ret+risk)...")
    step_start = time.time()
    
    master_df = _create_master_data(ret_df, risk_df)
    master_df.reset_index().to_parquet(snapshot_path / 'master.parquet', index=False)
    
    print(f"  master_data: {len(master_df):,} rows")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 5: Create pre-sorted asof tables for merge_asof
    # =========================================================================
    print("\nStep 5: Creating pre-sorted asof tables...")
    step_start = time.time()
    
    # Resid table: for signal residualization
    resid_cols = ["security_id", "date", "industry_id", "sector_id",
                  "size", "value", "growth", "leverage", "volatility", "momentum"]
    if 'yield' in risk_df.columns:
        resid_cols.append('yield')
    available_resid = [c for c in resid_cols if c in risk_df.columns]
    resid_table = risk_df[available_resid].copy()
    resid_table = resid_table.rename(columns={"date": "date_sig"})
    if 'yield' in resid_table.columns:
        resid_table = resid_table.rename(columns={"yield": "yields"})
    resid_table = resid_table.sort_values("date_sig")
    resid_table.to_parquet(snapshot_path / 'asof_resid.parquet', index=False)
    
    # Cap table: for byvar cap lookup
    cap_cols = ["security_id", "date", "cap"]
    available_cap = [c for c in cap_cols if c in risk_df.columns]
    if len(available_cap) == 3:
        cap_table = risk_df[available_cap].copy()
        cap_table = cap_table.rename(columns={"date": "date_sig"})
        cap_table = cap_table.sort_values("date_sig")
        cap_table.to_parquet(snapshot_path / 'asof_cap.parquet', index=False)
    
    print(f"  asof_resid: {len(resid_table):,} rows")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 6: Create factors table
    # =========================================================================
    print("\nStep 6: Creating factors table...")
    step_start = time.time()
    
    factor_df = _create_factor_data(risk_df)
    if len(factor_df) > 0:
        factor_df.reset_index().to_parquet(snapshot_path / 'factors.parquet', index=False)
        print(f"  factors: {len(factor_df):,} rows")
    else:
        print("  No factor columns found")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 7: Create manifest
    # =========================================================================
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': datetime.now().isoformat(),
        'version': 2,  # Version 2 = all precomputation done at snapshot time
        'files': {
            'ret': {
                'rows': len(ret_df),
                'securities': int(ret_df['security_id'].nunique()),
                'date_range': [str(ret_df['date'].min()), str(ret_df['date'].max())],
                'normalized': True,
            },
            'risk': {
                'rows': len(risk_df),
                'columns': list(risk_df.columns),
                'normalized': True,
            },
            'trading_date': {
                'rows': len(datefile),
            },
            'master': {
                'rows': len(master_df),
            },
            'factors': {
                'rows': len(factor_df),
                'columns': list(factor_df.columns) if len(factor_df) > 0 else [],
            },
            'asof_resid': {
                'rows': len(resid_table),
            },
        },
    }
    
    with open(snapshot_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Snapshot created: {snapshot_path}")
    print(f"Total time: {total_time:.1f}s")
    
    return snapshot_path


def list_snapshots(snapshots_dir: str = 'snapshots') -> list:
    """List all available snapshots."""
    path = Path(snapshots_dir)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir() and (d / 'manifest.json').exists()]
