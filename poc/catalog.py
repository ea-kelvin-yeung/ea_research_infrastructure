"""
Data Catalog: Load data from snapshots with manifest tracking.
~80 lines - keep it simple.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_catalog(snapshot_path: str = "snapshots/default") -> dict:
    """
    Load data catalog from a snapshot directory.
    
    Args:
        snapshot_path: Path to snapshot directory containing parquet files
        
    Returns:
        dict with keys: 'ret', 'risk', 'dates', 'snapshot_id', 'manifest'
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
