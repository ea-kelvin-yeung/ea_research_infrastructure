"""
BacktestService: Singleton service for running backtests with persistent data.

This module provides a singleton service that loads data once and keeps it in memory
for fast, repeated backtest execution. It's designed for:
- Jupyter notebooks (data persists across cells)
- Batch scripts (data persists across signal iterations)
- Server backends (data shared across requests)

Usage:
    from api import BacktestService
    
    # Get singleton (loads data on first call, instant after)
    service = BacktestService.get()
    
    # Run backtest
    result = service.run(signal_df, lag=0, resid="off")
    print(result.sharpe)
"""

import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import warnings

import numpy as np
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from poc.catalog import load_catalog, list_snapshots
from poc.wrapper import BacktestConfig, BacktestResult, run_backtest
from poc.contract import prepare_signal


def load_config_from_yaml(config_path: str = "api/backtest_config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict with configuration values, or empty dict if file not found
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    try:
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        warnings.warn("PyYAML not installed. Config file ignored. Install with: pip install pyyaml")
        return {}
    except Exception as e:
        warnings.warn(f"Failed to load config file {config_path}: {e}")
        return {}


@dataclass
class ServiceConfig:
    """Configuration for BacktestService."""
    snapshot: str = "latest"  # Snapshot name or "latest"
    use_mmap: bool = True  # Use memory-mapped files for large data
    cache_size: int = 10  # LRU cache size for recent signals
    preload_factors: bool = False  # Preload risk factors (uses more RAM)
    
    # Default backtest settings
    default_lag: int = 0
    default_resid: str = "off"
    default_byvar_list: List[str] = field(default_factory=lambda: ["overall", "year", "cap"])


class BacktestService:
    """
    Singleton service for running backtests with persistent data.
    
    The service loads data once and keeps it in memory for fast, repeated execution.
    Thread-safe for concurrent backtest requests.
    
    Example:
        service = BacktestService.get("2026-02-10-v1")
        result = service.run(signal_df, lag=0, resid="off")
    """
    
    _instance: Optional["BacktestService"] = None
    _lock = threading.Lock()
    
    @classmethod
    def get(
        cls,
        snapshot: Optional[str] = None,
        config: Optional[ServiceConfig] = None,
        config_file: Optional[str] = "api/backtest_config.yaml",
        force_reload: bool = False,
    ) -> "BacktestService":
        """
        Get or create the singleton BacktestService instance.
        
        Args:
            snapshot: Snapshot name (e.g., "2026-02-10-v1") or None for config/latest
            config: Service configuration (uses defaults if None)
            config_file: Path to YAML config file (default: backtest_config.yaml)
            force_reload: Force reload data even if already loaded
            
        Returns:
            BacktestService singleton instance
        """
        with cls._lock:
            # Load from config file if available
            file_config = {}
            if config_file:
                file_config = load_config_from_yaml(config_file)
            
            # Build config from file + defaults
            if config is None:
                memory_cfg = file_config.get("memory", {})
                defaults_cfg = file_config.get("defaults", {})
                config = ServiceConfig(
                    snapshot=file_config.get("snapshot", "latest"),
                    use_mmap=memory_cfg.get("use_mmap", True),
                    cache_size=memory_cfg.get("cache_size", 10),
                    preload_factors=memory_cfg.get("preload_factors", False),
                    default_lag=defaults_cfg.get("lag", 0),
                    default_resid=defaults_cfg.get("resid", "off"),
                    default_byvar_list=defaults_cfg.get("byvar_list", ["overall", "year", "cap"]),
                )
            
            # Resolve snapshot (CLI arg > config file > latest)
            if snapshot is None:
                snapshot = config.snapshot
            if snapshot is None or snapshot == "latest":
                snapshots = list_snapshots("snapshots")
                if not snapshots:
                    raise FileNotFoundError("No snapshots found in snapshots/")
                snapshot = snapshots[0]  # Most recent
            
            # Check if we need to create/reload
            needs_reload = (
                cls._instance is None or
                cls._instance.snapshot != snapshot or
                force_reload
            )
            
            if needs_reload:
                if cls._instance is not None:
                    print(f"Reloading data (snapshot: {snapshot})...")
                cls._instance = cls(snapshot, config)
            
            return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing or memory cleanup)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._catalog = None
                cls._instance = None
    
    def __init__(self, snapshot: str, config: ServiceConfig):
        """Initialize service (use get() instead of calling directly)."""
        self.snapshot = snapshot
        self.config = config
        self._catalog: Optional[dict] = None
        self._load_time: float = 0
        self._run_count: int = 0
        self._run_lock = threading.Lock()
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from snapshot."""
        start = time.perf_counter()
        
        snapshot_path = f"snapshots/{self.snapshot}"
        print(f"Loading snapshot: {snapshot_path}")
        
        # Load catalog with memory optimization
        self._catalog = load_catalog(
            snapshot_path,
            use_master=True,
            verify_integrity=False,
            universe_only=False,  # Load full universe for flexibility
        )
        
        self._load_time = time.perf_counter() - start
        
        # Report stats
        master_rows = len(self._catalog.get('master', pd.DataFrame()))
        dates_rows = len(self._catalog.get('dates', pd.DataFrame()))
        print(f"Loaded in {self._load_time:.1f}s: {master_rows:,} master rows, {dates_rows:,} dates")
    
    @property
    def is_ready(self) -> bool:
        """Check if data is loaded and ready."""
        return self._catalog is not None
    
    @property
    def catalog(self) -> dict:
        """Get the loaded catalog (for advanced usage)."""
        if not self.is_ready:
            raise RuntimeError("Service not initialized. Call BacktestService.get() first.")
        return self._catalog
    
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "snapshot": self.snapshot,
            "is_ready": self.is_ready,
            "load_time_seconds": self._load_time,
            "run_count": self._run_count,
            "master_rows": len(self._catalog.get('master', pd.DataFrame())) if self._catalog else 0,
            "dates_count": len(self._catalog.get('dates', pd.DataFrame())) if self._catalog else 0,
        }
    
    def run(
        self,
        signal: pd.DataFrame,
        lag: Optional[int] = None,
        resid: Optional[str] = None,
        byvar_list: Optional[List[str]] = None,
        fractile: tuple = (10, 90),
        weight: str = "equal",
        tc_model: str = "naive",
        from_open: bool = False,
        mincos: int = 10,
        validate: bool = True,
    ) -> BacktestResult:
        """
        Run a single backtest on the given signal.
        
        Args:
            signal: Signal DataFrame with columns [security_id, date_sig, signal]
                    Optional: date_avail, date_ret, date_openret
            lag: Trading lag in days (default: config.default_lag)
            resid: Residualization mode: "off", "industry", "all" (default: config.default_resid)
            byvar_list: Analysis slices (default: config.default_byvar_list)
            fractile: Long/short percentile thresholds (default: (10, 90))
            weight: Weighting method: "equal", "value", "volume" (default: "equal")
            tc_model: Transaction cost model: "naive", "power_law" (default: "naive")
            from_open: Trade at open vs close (default: False)
            mincos: Minimum companies per side (default: 10)
            validate: Validate signal format (default: True)
            
        Returns:
            BacktestResult with summary, daily series, and fractile analysis
        """
        if not self.is_ready:
            raise RuntimeError("Service not initialized. Call BacktestService.get() first.")
        
        # Apply defaults
        if lag is None:
            lag = self.config.default_lag
        if resid is None:
            resid = self.config.default_resid
        if byvar_list is None:
            byvar_list = self.config.default_byvar_list.copy()
        
        # Build config
        bt_config = BacktestConfig(
            lag=lag,
            residualize=resid,
            byvar_list=byvar_list,
            fractile=fractile,
            weight=weight,
            tc_model=tc_model,
            from_open=from_open,
            mincos=mincos,
        )
        
        # Thread-safe run
        with self._run_lock:
            self._run_count += 1
        
        # Run backtest using wrapper
        return run_backtest(signal, self._catalog, bt_config, validate=validate)
    
    def run_suite(
        self,
        signal: pd.DataFrame,
        lags: Optional[List[int]] = None,
        resid_modes: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Run a suite of backtests with different configurations.
        
        Args:
            signal: Signal DataFrame
            lags: List of lags to test (default: [0])
            resid_modes: List of residualization modes (default: ["off"])
            **kwargs: Additional arguments passed to run()
            
        Returns:
            Dict mapping config keys (e.g., "lag0_residoff") to BacktestResult
        """
        if lags is None:
            lags = [0]
        if resid_modes is None:
            resid_modes = ["off"]
        
        results = {}
        for lag in lags:
            for resid in resid_modes:
                key = f"lag{lag}_resid{resid}"
                results[key] = self.run(signal, lag=lag, resid=resid, **kwargs)
        
        return results
    
    def filter_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> "BacktestService":
        """
        Filter the loaded data to a date range (modifies in place).
        
        Note: This modifies the cached data. Call reset() and get() to reload full data.
        
        Args:
            start_date: Start date string (e.g., "2020-01-01")
            end_date: End date string (e.g., "2023-12-31")
            
        Returns:
            self for chaining
        """
        if not self.is_ready:
            raise RuntimeError("Service not initialized.")
        
        if start_date:
            self._catalog['ret'] = self._catalog['ret'][
                self._catalog['ret']['date'] >= start_date
            ]
            self._catalog['risk'] = self._catalog['risk'][
                self._catalog['risk']['date'] >= start_date
            ]
            self._catalog['dates'] = self._catalog['dates'][
                self._catalog['dates']['date'] >= start_date
            ]
            if 'master' in self._catalog and self._catalog['master'] is not None:
                master = self._catalog['master']
                if isinstance(master.index, pd.MultiIndex):
                    master = master.reset_index()
                    master = master[master['date'] >= start_date]
                    master = master.set_index(['security_id', 'date'])
                else:
                    master = master[master['date'] >= start_date]
                self._catalog['master'] = master
        
        if end_date:
            self._catalog['ret'] = self._catalog['ret'][
                self._catalog['ret']['date'] <= end_date
            ]
            self._catalog['risk'] = self._catalog['risk'][
                self._catalog['risk']['date'] <= end_date
            ]
            self._catalog['dates'] = self._catalog['dates'][
                self._catalog['dates']['date'] <= end_date
            ]
            if 'master' in self._catalog and self._catalog['master'] is not None:
                master = self._catalog['master']
                if isinstance(master.index, pd.MultiIndex):
                    master = master.reset_index()
                    master = master[master['date'] <= end_date]
                    master = master.set_index(['security_id', 'date'])
                else:
                    master = master[master['date'] <= end_date]
                self._catalog['master'] = master
        
        return self


# Convenience function for quick access
def get_service(snapshot: Optional[str] = None, **kwargs) -> BacktestService:
    """
    Convenience function to get the BacktestService singleton.
    
    Args:
        snapshot: Snapshot name or None for latest
        **kwargs: Additional ServiceConfig options
        
    Returns:
        BacktestService instance
    """
    config = ServiceConfig(**kwargs) if kwargs else None
    return BacktestService.get(snapshot, config)
