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
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple


# Standard risk factors used for signal quality assessment
RISK_FACTORS = ['size', 'value', 'growth', 'leverage', 'volatility', 'momentum']


# =============================================================================
# Content-Based Hashing for Reproducibility
# =============================================================================

def compute_table_hash(df: pd.DataFrame, sample_rows: int = 10000) -> str:
    """
    Compute deterministic hash of DataFrame content.
    
    Uses sampling for speed on large tables while ensuring reproducibility.
    Same data content → same hash, regardless of when snapshot was created.
    
    Components hashed:
    - Schema (column names + dtypes)
    - Shape (rows x cols)
    - Date range (if date column exists)
    - Sampled content (deterministic sampling)
    
    Args:
        df: DataFrame to hash
        sample_rows: Max rows to sample for content hashing (default 10000)
        
    Returns:
        16-character hex hash string
    """
    # Include schema in hash (column names + dtypes, sorted for determinism)
    schema = str([(c, str(df[c].dtype)) for c in sorted(df.columns)])
    
    # Include shape
    shape = f"{len(df)}x{len(df.columns)}"
    
    # Include date range if present
    date_range = ""
    if 'date' in df.columns:
        date_range = f"{df['date'].min()}:{df['date'].max()}"
    
    # Deterministic content sampling
    if len(df) > sample_rows:
        # Sample: first 1000 + last 1000 + evenly spaced
        n = len(df)
        indices = (
            list(range(min(1000, n))) +  # First 1000
            list(range(max(0, n - 1000), n)) +  # Last 1000
            list(range(0, n, max(1, n // (sample_rows - 2000))))  # Evenly spaced
        )
        sample = df.iloc[sorted(set(indices))]
    else:
        sample = df
    
    # Convert to CSV for deterministic string representation
    # Use consistent date format and handle NaN consistently
    content = sample.to_csv(index=False, date_format='%Y-%m-%d', na_rep='NA')
    
    # Combine all components
    fingerprint = f"SCHEMA:{schema}|SHAPE:{shape}|DATES:{date_range}|CONTENT:{content}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def compute_snapshot_fingerprint(component_hashes: Dict[str, str]) -> str:
    """
    Combine component table hashes into a single snapshot fingerprint.
    
    Args:
        component_hashes: Dict mapping table name to hash
            e.g., {'ret': 'abc123...', 'risk': 'def456...', 'dates': 'ghi789...'}
    
    Returns:
        12-character hex meta-hash
    """
    # Sort keys for determinism
    combined = "|".join(f"{k}:{v}" for k, v in sorted(component_hashes.items()))
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def verify_snapshot_integrity(catalog: dict) -> Tuple[bool, Dict[str, str]]:
    """
    Verify that loaded data matches the hashes stored in manifest.
    
    Args:
        catalog: Loaded catalog dict
        
    Returns:
        (is_valid, details) - is_valid is True if all hashes match
    """
    manifest = catalog.get('manifest', {})
    fingerprint = manifest.get('fingerprint', {})
    stored_hashes = fingerprint.get('components', {})
    
    if not stored_hashes:
        return True, {'status': 'no_hashes_stored', 'message': 'Legacy snapshot without hashes'}
    
    details = {}
    all_match = True
    
    # Check each component
    for table_name, stored_info in stored_hashes.items():
        stored_hash = stored_info.get('hash', '')
        if table_name in catalog and catalog[table_name] is not None:
            current_hash = compute_table_hash(catalog[table_name])
            matches = current_hash == stored_hash
            details[table_name] = {
                'stored': stored_hash,
                'current': current_hash,
                'matches': matches
            }
            if not matches:
                all_match = False
        else:
            details[table_name] = {'status': 'not_loaded'}
    
    return all_match, details


# =============================================================================
# Content-Addressable Storage (Component Reuse)
# =============================================================================

OBJECTS_DIR = "objects"  # Shared storage for deduplicated components


def _get_object_path(base_dir: Path, table_hash: str, table_name: str) -> Path:
    """Get path to a content-addressed object file."""
    # Use first 2 chars as subdirectory (like git) for filesystem efficiency
    return base_dir / OBJECTS_DIR / table_hash[:2] / f"{table_hash}_{table_name}.parquet"


def _store_object(df: pd.DataFrame, base_dir: Path, table_name: str) -> Tuple[str, Path, bool]:
    """
    Store a DataFrame in content-addressable storage.
    
    Returns:
        (hash, path, is_new) - is_new=False if object already existed (reused)
    """
    table_hash = compute_table_hash(df)
    object_path = _get_object_path(base_dir, table_hash, table_name)
    
    if object_path.exists():
        # Object already exists - reuse it!
        return table_hash, object_path, False
    
    # New object - store it
    object_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(object_path, index=False)
    return table_hash, object_path, True


def _load_object(base_dir: Path, table_hash: str, table_name: str) -> Optional[pd.DataFrame]:
    """Load a DataFrame from content-addressable storage."""
    object_path = _get_object_path(base_dir, table_hash, table_name)
    if object_path.exists():
        return pd.read_parquet(object_path)
    return None


def get_object_stats(base_dir: str = "snapshots") -> Dict:
    """
    Get statistics about the content-addressable object store.
    
    Returns:
        Dict with counts of unique objects, total size, etc.
    """
    objects_path = Path(base_dir) / OBJECTS_DIR
    if not objects_path.exists():
        return {'exists': False, 'unique_objects': 0, 'total_size_mb': 0}
    
    total_size = 0
    object_count = 0
    objects_by_type = {}
    
    for subdir in objects_path.iterdir():
        if subdir.is_dir():
            for obj_file in subdir.glob("*.parquet"):
                object_count += 1
                total_size += obj_file.stat().st_size
                # Extract table name from filename
                parts = obj_file.stem.split('_', 1)
                if len(parts) == 2:
                    table_name = parts[1]
                    objects_by_type[table_name] = objects_by_type.get(table_name, 0) + 1
    
    return {
        'exists': True,
        'unique_objects': object_count,
        'total_size_mb': total_size / (1024 * 1024),
        'by_type': objects_by_type,
    }


# =============================================================================
# Partitioned Storage (Incremental Updates)
# =============================================================================

def _partition_by_year(df: pd.DataFrame, date_col: str = 'date') -> Dict[str, pd.DataFrame]:
    """Split DataFrame into yearly partitions."""
    if date_col not in df.columns:
        return {'all': df}
    
    df = df.copy()
    df['_year'] = pd.to_datetime(df[date_col]).dt.year
    
    partitions = {}
    for year, group in df.groupby('_year'):
        partitions[str(year)] = group.drop(columns=['_year'])
    
    return partitions


def _store_partitions(
    df: pd.DataFrame,
    base_dir: Path,
    table_name: str,
    date_col: str = 'date',
) -> Dict[str, Dict]:
    """
    Store DataFrame as yearly partitions with deduplication.
    
    Returns:
        Dict mapping year -> {hash, rows, path, is_new}
    """
    partitions = _partition_by_year(df, date_col)
    result = {}
    
    for year, part_df in partitions.items():
        part_name = f"{table_name}_{year}"
        part_hash, obj_path, is_new = _store_object(part_df, base_dir, part_name)
        result[year] = {
            'hash': part_hash,
            'rows': len(part_df),
            'path': str(obj_path.relative_to(base_dir)),
            'is_new': is_new,
        }
    
    return result


def _load_partitions(
    base_dir: Path,
    table_name: str,
    partitions: Dict[str, Dict],
) -> pd.DataFrame:
    """Load and concatenate yearly partitions."""
    dfs = []
    for year, part_info in sorted(partitions.items()):
        part_path = base_dir / part_info['path']
        if part_path.exists():
            dfs.append(pd.read_parquet(part_path))
        else:
            # Fallback to object store lookup
            part_name = f"{table_name}_{year}"
            part_df = _load_object(base_dir, part_info['hash'], part_name)
            if part_df is not None:
                dfs.append(part_df)
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def update_snapshot_incremental(
    snapshot_path: str,
    new_data: Dict[str, pd.DataFrame],
    output_dir: str = 'snapshots',
) -> Path:
    """
    Update a snapshot incrementally by only replacing changed partitions.
    
    Args:
        snapshot_path: Path to existing snapshot
        new_data: Dict of {table_name: DataFrame} with new/updated data
        output_dir: Where to store objects
        
    Returns:
        Path to updated snapshot
    """
    import time
    from datetime import datetime
    
    path = Path(snapshot_path)
    base_dir = Path(output_dir)
    
    # Load existing manifest
    with open(path / 'manifest.json') as f:
        manifest = json.load(f)
    
    print(f"Updating snapshot: {manifest['snapshot_id']}")
    print("=" * 60)
    
    updated_components = []
    
    for table_name, new_df in new_data.items():
        if table_name not in manifest['components']:
            print(f"  Skipping unknown component: {table_name}")
            continue
        
        comp = manifest['components'][table_name]
        old_partitions = comp.get('partitions', {})
        
        # Partition new data
        new_partitions = _store_partitions(new_df, base_dir, table_name)
        
        # Merge: keep old partitions, update/add new ones
        merged_partitions = dict(old_partitions)
        reused = 0
        updated = 0
        
        for year, part_info in new_partitions.items():
            if year in old_partitions and old_partitions[year]['hash'] == part_info['hash']:
                reused += 1
            else:
                merged_partitions[year] = {
                    'hash': part_info['hash'],
                    'rows': part_info['rows'],
                    'path': part_info['path'],
                    'updated': datetime.now().isoformat()[:10],
                }
                updated += 1
        
        # Update component
        total_rows = sum(p['rows'] for p in merged_partitions.values())
        comp_hash = compute_snapshot_fingerprint({y: p['hash'] for y, p in merged_partitions.items()})
        
        manifest['components'][table_name] = {
            'source': comp.get('source', f'data/{table_name}.parquet'),
            'hash': comp_hash,
            'rows': total_rows,
            'partitions': merged_partitions,
        }
        
        if 'securities' in comp:
            manifest['components'][table_name]['securities'] = int(new_df['security_id'].nunique())
        if 'date_range' in comp:
            manifest['components'][table_name]['date_range'] = [
                str(new_df['date'].min().date()),
                str(new_df['date'].max().date()),
            ]
        
        print(f"  {table_name}: {reused} partitions reused, {updated} updated")
        if updated > 0:
            updated_components.append(table_name)
    
    # Recompute fingerprint
    comp_hashes = {k: v['hash'] for k, v in manifest['components'].items() if 'hash' in v}
    manifest['fingerprint'] = compute_snapshot_fingerprint(comp_hashes)
    manifest['updated_at'] = datetime.now().isoformat()
    
    # Save updated manifest
    with open(path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Updated components: {updated_components}")
    print(f"  New fingerprint: {manifest['fingerprint']}")
    
    return path


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


def load_catalog(
    snapshot_path: str = "snapshots/default",
    use_master: bool = True,
    verify_integrity: bool = False,
) -> dict:
    """
    Load data catalog from a snapshot directory.
    
    Fast load path: All heavy lifting was done at snapshot creation time.
    This function just reads pre-computed files and restores indexes.
    
    Args:
        snapshot_path: Path to snapshot directory containing parquet files
        use_master: If True, load master_data for fast backtest joins
        verify_integrity: If True, verify content hashes match manifest (slower)
        
    Returns:
        dict with keys: 'ret', 'risk', 'dates', 'snapshot_id', 'manifest', 'fingerprint'
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
    is_v4 = manifest.get('version', 1) >= 4  # V4 = content-addressable with object references
    
    components = manifest.get('components', {})
    
    # Helper to load a table (supports partitions, object store, or direct files)
    def _load_table(table_name: str, default_filename: str) -> pd.DataFrame:
        """Load table from partitions, object store, or direct file."""
        comp = components.get(table_name, {})
        
        # Check if partitioned
        if 'partitions' in comp:
            return _load_partitions(path.parent, table_name, comp['partitions'])
        
        # Check object store (V4+)
        if is_v4:
            objects = manifest.get('objects', {})
            if table_name in objects:
                obj_path = path.parent / objects[table_name]['object_path']
                if obj_path.exists():
                    return pd.read_parquet(obj_path)
        
        # Direct file in snapshot dir
        default_path = path / default_filename
        if default_path.exists():
            return pd.read_parquet(default_path)
        
        raise FileNotFoundError(f"Cannot find {table_name} in snapshot")
    
    # Load base data files (already normalized in v2 snapshots)
    ret_df = _load_table('ret', 'ret.parquet')
    risk_df = _load_table('risk', 'risk.parquet')
    dates_df = _load_table('dates', 'trading_date.parquet')
    
    # For v1 snapshots, normalize on load (backwards compatibility)
    if not is_v2:
        ret_df = _normalize_dataframe(ret_df, 'ret')
        risk_df = _normalize_dataframe(risk_df, 'risk')
        dates_df = _normalize_datefile(dates_df)
    
    # Extract fingerprint from manifest (v3+)
    fingerprint = manifest.get('fingerprint', None)
    # Handle both old format (dict with meta_hash) and new format (string)
    if isinstance(fingerprint, dict):
        meta_hash = fingerprint.get('meta_hash', None)
    else:
        meta_hash = fingerprint  # V5: fingerprint is directly the hash string
    
    catalog = {
        'ret': ret_df,
        'risk': risk_df,
        'dates': dates_df,
        'snapshot_id': path.name,
        'snapshot_path': str(path.resolve()),
        'manifest': manifest,
        'fingerprint': meta_hash,  # Quick access to content fingerprint
    }
    
    # Verify integrity if requested
    if verify_integrity and fingerprint:
        print("Verifying snapshot integrity...")
        is_valid, details = verify_snapshot_integrity(catalog)
        if not is_valid:
            import warnings
            warnings.warn(f"Snapshot integrity check failed: {details}")
        catalog['integrity_verified'] = is_valid
        catalog['integrity_details'] = details
    
    if use_master:
        # Load master_data (pre-merged ret+risk)
        master_path = path / 'master.parquet'
        try:
            master_df = _load_table('master', 'master.parquet')
            # Restore MultiIndex (cheap operation - skip sort_index for speed)
            if 'security_id' in master_df.columns:
                master_df = master_df.set_index(['security_id', 'date'])
            catalog['master'] = master_df
        except FileNotFoundError:
            if not is_v2:
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
        
        # Pre-compute Polars DataFrames to avoid expensive runtime conversions
        # Load directly from parquet as Polars to avoid memory duplication
        try:
            import polars as pl
            
            # Master as Polars - load directly from parquet (avoids reset_index)
            master_parquet = path / 'master.parquet'
            if master_parquet.exists():
                catalog['master_pl'] = pl.read_parquet(master_parquet)
            elif 'master' in catalog and catalog['master'] is not None:
                # Fallback: convert from Pandas
                catalog['master_pl'] = pl.from_pandas(catalog['master'].reset_index())
            
            # Otherfile (risk) as Polars - load from parquet
            risk_parquet = path / 'risk.parquet'
            if risk_parquet.exists():
                catalog['otherfile_pl'] = pl.read_parquet(risk_parquet)
            else:
                # Check partitioned
                risk_partitions = path / 'partitions' / 'risk'
                if risk_partitions.exists():
                    catalog['otherfile_pl'] = pl.read_parquet(risk_partitions)
                else:
                    catalog['otherfile_pl'] = pl.from_pandas(risk_df)
            
            # Retfile as Polars - for ret column joins that need retfile rows
            ret_parquet = path / 'ret.parquet'
            if ret_parquet.exists():
                catalog['retfile_pl'] = pl.read_parquet(ret_parquet)
            else:
                ret_partitions = path / 'partitions' / 'ret'
                if ret_partitions.exists():
                    catalog['retfile_pl'] = pl.read_parquet(ret_partitions)
                else:
                    catalog['retfile_pl'] = pl.from_pandas(ret_df)
            
            # Datefile as Polars - for date/n lookups in turnover
            catalog['datefile_pl'] = pl.from_pandas(dates_df)
            
            # Pre-sorted asof tables as Polars
            catalog['asof_tables_pl'] = {
                'resid': pl.from_pandas(resid_table).sort('date_sig'),
                'byvars_cap': pl.from_pandas(cap_table).sort('date_sig'),
            }
            
        except ImportError:
            # Polars not installed, skip pre-computation
            pass
    
    return catalog


def create_snapshot(
    source_dir: str = 'data',
    output_dir: str = 'snapshots',
    snapshot_id: Optional[str] = None,
    use_content_hash: bool = False,
    deduplicate: bool = False,
    sources: Optional[Dict[str, Dict]] = None,
) -> Path:
    """
    Create a snapshot from existing data files with ALL precomputation done upfront.
    
    This is where ALL heavy lifting happens:
    1. Normalize data (deduplicate, convert dtypes for fast hashing)
    2. Create master_data (pre-merged ret+risk)
    3. Create pre-sorted asof tables for merge_asof
    4. Create factors table
    5. Compute content fingerprints for reproducibility
    6. (Optional) Store components in shared object store for reuse
    
    After this, load_catalog() is just fast file reads.
    
    Args:
        source_dir: Directory containing ret.parquet, risk.parquet, trading_date.pkl
        output_dir: Directory to store snapshots
        snapshot_id: Optional custom snapshot ID (default: date-based, or content-hash if use_content_hash=True)
        use_content_hash: If True, use content-based hash as snapshot ID (enables deduplication)
        deduplicate: If True, store components in shared object store by hash (saves disk space)
        sources: Optional dict with source lineage for each component, e.g.:
            {
                'ret': {'type': 'prod_export', 'uri': 's3://...', 'extraction_id': '...'},
                'risk': {'type': 'prod_export', 'uri': 's3://...'},
                'dates': {'type': 'static_file', 'uri': 's3://...'},
            }
        
    Returns:
        Path to created snapshot directory
    """
    import time
    start_time = time.time()
    
    source = Path(source_dir)
    
    # Defer snapshot_id assignment if using content hash (computed after loading data)
    temp_snapshot_id = snapshot_id or f"{datetime.now().strftime('%Y-%m-%d')}-v1"
    snapshot_path = Path(output_dir) / temp_snapshot_id
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
    # Step 3: Save normalized base files (with optional deduplication)
    # =========================================================================
    print("\nStep 3: Saving normalized base files...")
    step_start = time.time()
    
    base_dir = Path(output_dir)
    object_refs = {}  # Track object references for manifest
    reused_count = 0
    
    if deduplicate:
        # Use content-addressable storage - store once, reference by hash
        for table_name, df in [('ret', ret_df), ('risk', risk_df), ('dates', datefile)]:
            table_hash, obj_path, is_new = _store_object(df, base_dir, table_name)
            object_refs[table_name] = {
                'hash': table_hash,
                'object_path': str(obj_path.relative_to(base_dir)),
            }
            # Create symlink in snapshot dir for easy access
            snapshot_link = snapshot_path / f'{table_name}.parquet'
            if snapshot_link.exists():
                snapshot_link.unlink()
            snapshot_link.symlink_to(obj_path.resolve())
            
            status = "NEW" if is_new else "REUSED"
            if not is_new:
                reused_count += 1
            print(f"  {table_name}: {table_hash} [{status}]")
    else:
        # Traditional: copy files directly to snapshot dir
        ret_df.to_parquet(snapshot_path / 'ret.parquet', index=False)
        risk_df.to_parquet(snapshot_path / 'risk.parquet', index=False)
        datefile.to_parquet(snapshot_path / 'trading_date.parquet', index=False)
    
    if reused_count > 0:
        print(f"  Reused {reused_count} existing components (saved disk space)")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # =========================================================================
    # Step 4: Create master_data (pre-merged ret+risk, indexed)
    # =========================================================================
    print("\nStep 4: Creating master_data (pre-merged ret+risk)...")
    step_start = time.time()
    
    master_df = _create_master_data(ret_df, risk_df)
    master_flat = master_df.reset_index()
    
    if deduplicate:
        master_hash, obj_path, is_new = _store_object(master_flat, base_dir, 'master')
        object_refs['master'] = {
            'hash': master_hash,
            'object_path': str(obj_path.relative_to(base_dir)),
        }
        snapshot_link = snapshot_path / 'master.parquet'
        if snapshot_link.exists():
            snapshot_link.unlink()
        snapshot_link.symlink_to(obj_path.resolve())
        status = "NEW" if is_new else "REUSED"
        if not is_new:
            reused_count += 1
        print(f"  master: {master_hash} [{status}]")
    else:
        master_flat.to_parquet(snapshot_path / 'master.parquet', index=False)
    
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
    # Step 7: Compute content hashes for reproducibility
    # =========================================================================
    print("\nStep 7: Computing content fingerprints...")
    step_start = time.time()
    
    ret_hash = compute_table_hash(ret_df)
    risk_hash = compute_table_hash(risk_df)
    dates_hash = compute_table_hash(datefile)
    master_hash = compute_table_hash(master_df.reset_index())
    
    component_hashes = {
        'ret': ret_hash,
        'risk': risk_hash,
        'dates': dates_hash,
        'master': master_hash,
    }
    meta_hash = compute_snapshot_fingerprint(component_hashes)
    
    print(f"  ret:    {ret_hash}")
    print(f"  risk:   {risk_hash}")
    print(f"  dates:  {dates_hash}")
    print(f"  master: {master_hash}")
    print(f"  META:   {meta_hash}")
    print(f"  Time: {time.time() - step_start:.1f}s")
    
    # If using content-based ID, rename snapshot directory
    if use_content_hash and snapshot_id is None:
        content_snapshot_id = f"snap_{meta_hash}"
        new_snapshot_path = Path(output_dir) / content_snapshot_id
        
        # Check if this exact data already exists
        if new_snapshot_path.exists():
            print(f"\n  Snapshot with identical content already exists: {content_snapshot_id}")
            # Clean up temporary directory
            import shutil
            shutil.rmtree(snapshot_path)
            return new_snapshot_path
        
        # Rename to content-based ID
        snapshot_path.rename(new_snapshot_path)
        snapshot_path = new_snapshot_path
        snapshot_id = content_snapshot_id
    else:
        snapshot_id = temp_snapshot_id
    
    # =========================================================================
    # Step 8: Create manifest (simplified)
    # =========================================================================
    
    source_info = sources or {}
    
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': datetime.now().isoformat(),
        'version': 5,
        'fingerprint': meta_hash,
        'components': {
            'ret': {
                'source': source_info.get('ret', {}).get('source', str(source / 'ret.parquet')),
                'hash': ret_hash,
                'rows': len(ret_df),
                'securities': int(ret_df['security_id'].nunique()),
                'date_range': [str(ret_df['date'].min().date()), str(ret_df['date'].max().date())],
            },
            'risk': {
                'source': source_info.get('risk', {}).get('source', str(source / 'risk.parquet')),
                'hash': risk_hash,
                'rows': len(risk_df),
            },
            'dates': {
                'source': source_info.get('dates', {}).get('source', str(source / 'trading_date.pkl')),
                'hash': dates_hash,
                'rows': len(datefile),
            },
            'master': {
                'derived_from': ['ret', 'risk'],
                'hash': master_hash,
                'rows': len(master_df),
            },
            'factors': {
                'derived_from': ['risk'],
                'rows': len(factor_df),
            },
            'asof_resid': {
                'derived_from': ['risk'],
                'rows': len(resid_table),
            },
        },
    }
    
    # Add object store refs if using deduplication
    if deduplicate and object_refs:
        manifest['objects'] = object_refs
    
    with open(snapshot_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Snapshot created: {snapshot_path}")
    print(f"Fingerprint: {meta_hash}")
    print(f"Total time: {total_time:.1f}s")
    
    return snapshot_path


def list_snapshots(snapshots_dir: str = 'snapshots') -> list:
    """List all available snapshots."""
    path = Path(snapshots_dir)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir() and (d / 'manifest.json').exists()]
