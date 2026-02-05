"""
Data Catalog: Load data from snapshots with manifest tracking.

Includes master_data: pre-merged ret+risk DataFrame for fast backtest joins.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


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
    optional_risk_cols = ['mcap', 'cap', 'industry_id', 'sector_id', 
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


def load_catalog(snapshot_path: str = "snapshots/default", use_master: bool = True) -> dict:
    """
    Load data catalog from a snapshot directory.
    
    Args:
        snapshot_path: Path to snapshot directory containing parquet files
        use_master: If True, load/create master_data for fast backtest joins
        
    Returns:
        dict with keys: 'ret', 'risk', 'dates', 'snapshot_id', 'manifest'
        If use_master=True, also includes 'master' (indexed DataFrame)
    """
    path = Path(snapshot_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    
    catalog = {
        'ret': pd.read_parquet(path / 'ret.parquet'),
        'risk': pd.read_parquet(path / 'risk.parquet'),
        'dates': pd.read_parquet(path / 'trading_date.parquet'),
        'snapshot_id': path.name,
    }
    
    # Load manifest if exists
    manifest_path = path / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            catalog['manifest'] = json.load(f)
    else:
        catalog['manifest'] = {'snapshot_id': path.name, 'created_at': 'unknown'}
    
    # Load or create master data for fast joins
    if use_master:
        master_path = path / 'master.parquet'
        if master_path.exists():
            # Load cached master data
            catalog['master'] = pd.read_parquet(master_path)
            # Restore index
            if 'security_id' in catalog['master'].columns:
                catalog['master'] = catalog['master'].set_index(['security_id', 'date']).sort_index()
        else:
            # Create and cache master data
            catalog['master'] = _create_master_data(catalog['ret'], catalog['risk'])
            # Save to disk (reset index for parquet)
            catalog['master'].reset_index().to_parquet(master_path, index=False)
            print(f"Created master.parquet cache: {len(catalog['master'])} rows")
    
    return catalog


def create_snapshot(
    source_dir: str = 'data',
    output_dir: str = 'snapshots',
    snapshot_id: Optional[str] = None,
) -> Path:
    """
    Create a snapshot from existing data files.
    
    Args:
        source_dir: Directory containing ret.parquet, risk.parquet, trading_date.pkl
        output_dir: Directory to store snapshots
        snapshot_id: Optional custom snapshot ID (default: date-based)
        
    Returns:
        Path to created snapshot directory
    """
    source = Path(source_dir)
    snapshot_id = snapshot_id or f"{datetime.now().strftime('%Y-%m-%d')}-v1"
    snapshot_path = Path(output_dir) / snapshot_id
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    # Read source files
    ret_df = pd.read_parquet(source / 'ret.parquet')
    risk_df = pd.read_parquet(source / 'risk.parquet')
    
    # Handle both pkl and parquet for trading_date
    date_file = source / 'trading_date.pkl'
    if date_file.exists():
        datefile = pd.read_pickle(date_file)
    else:
        datefile = pd.read_parquet(source / 'trading_date.parquet')
    
    # Write to snapshot
    ret_df.to_parquet(snapshot_path / 'ret.parquet', index=False)
    risk_df.to_parquet(snapshot_path / 'risk.parquet', index=False)
    datefile.to_parquet(snapshot_path / 'trading_date.parquet', index=False)
    
    # Create master data (pre-merged for fast backtest)
    master_df = _create_master_data(ret_df, risk_df)
    master_df.reset_index().to_parquet(snapshot_path / 'master.parquet', index=False)
    
    # Create manifest
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': datetime.now().isoformat(),
        'files': {
            'ret': {
                'rows': len(ret_df),
                'securities': int(ret_df['security_id'].nunique()),
                'date_range': [str(ret_df['date'].min()), str(ret_df['date'].max())],
            },
            'risk': {
                'rows': len(risk_df),
                'columns': list(risk_df.columns),
            },
            'trading_date': {
                'rows': len(datefile),
            },
        },
    }
    
    with open(snapshot_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Created snapshot: {snapshot_path}")
    return snapshot_path


def list_snapshots(snapshots_dir: str = 'snapshots') -> list:
    """List all available snapshots."""
    path = Path(snapshots_dir)
    if not path.exists():
        return []
    return [d.name for d in path.iterdir() if d.is_dir() and (d / 'manifest.json').exists()]
