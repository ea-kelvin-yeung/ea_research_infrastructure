"""
BacktestService: Fast singleton service for running backtests.

Loads data once, keeps only what's needed in memory (~4GB instead of ~12GB),
and provides a fast `run()` method for batch signal testing.

Usage:
    from api import BacktestService
    
    # Load data (first call ~30s, instant after)
    service = BacktestService.get(start_date='2012-01-01', end_date='2021-12-31')
    
    # Run backtest (~1.8s per signal)
    result = service.run(signal_df)
    print(result[0].iloc[0]['sharpe_ret'])  # summary DataFrame
"""

import gc
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.catalog import load_catalog, list_snapshots


class BacktestService:
    """
    Singleton service for fast backtests with persistent data.
    
    - Loads master + dates only (minimal memory: ~4GB vs ~12GB full)
    - Thread-safe for concurrent requests
    - Use run() for pre-aligned signals (fastest path)
    
    Example:
        service = BacktestService.get('2026-02-10-v1', start_date='2020-01-01')
        result = service.run(signal_df, sigvar='my_signal')
        sharpe = result[0].iloc[0]['sharpe_ret']
    """
    
    _instance: Optional["BacktestService"] = None
    _lock = threading.Lock()
    
    @classmethod
    def get(
        cls,
        snapshot: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_reload: bool = False,
        compact: bool = True,
    ) -> "BacktestService":
        """
        Get or create the singleton BacktestService.
        
        Args:
            snapshot: Snapshot name (e.g., "2026-02-10-v1") or None for latest
            start_date: Filter data from this date (e.g., "2012-01-01")
            end_date: Filter data to this date (e.g., "2021-12-31")
            force_reload: Force reload even if already loaded
            compact: Downcast dtypes to save ~50% memory (default: True)
            
        Returns:
            BacktestService singleton
        """
        with cls._lock:
            # Resolve snapshot
            if snapshot is None:
                snapshots = list_snapshots("snapshots")
                if not snapshots:
                    raise FileNotFoundError("No snapshots found in snapshots/")
                snapshot = snapshots[0]
            
            # Check if reload needed
            needs_reload = (
                cls._instance is None or
                cls._instance.snapshot != snapshot or
                force_reload
            )
            
            if needs_reload:
                cls._instance = cls(snapshot, start_date, end_date, compact)
            elif start_date or end_date:
                cls._instance._filter_dates(start_date, end_date)
            
            return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton (for testing or memory cleanup)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._master = None
                cls._instance._dates = None
                cls._instance = None
            gc.collect()
    
    def __init__(
        self,
        snapshot: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compact: bool = True,
    ):
        """Initialize service (use get() instead)."""
        self.snapshot = snapshot
        self._master: Optional[pd.DataFrame] = None
        self._dates: Optional[pd.DataFrame] = None
        self._load_time: float = 0
        self._run_count: int = 0
        self._compact: bool = compact
        self._lock = threading.Lock()
        
        self._load_data(start_date, end_date, compact)
    
    def _load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compact: bool = True,
    ):
        """
        Load master + dates with optional date filtering and compaction.
        
        LEAN LOADING: Only loads master.parquet and dates.parquet directly,
        skipping ret/risk to avoid ~6GB of intermediate memory usage.
        """
        t0 = time.perf_counter()
        
        date_range = ""
        if start_date or end_date:
            date_range = f" [{start_date or '*'} to {end_date or '*'}]"
        mode = "compact" if compact else "full"
        print(f"Loading: snapshots/{self.snapshot}{date_range} ({mode})")
        
        snapshot_path = Path(f"snapshots/{self.snapshot}")
        
        # LEAN LOAD: Only load master and dates directly (skip ret/risk)
        # This saves ~6GB of intermediate memory vs load_catalog
        master_path = snapshot_path / "master.parquet"
        dates_path = snapshot_path / "trading_date.parquet"
        
        if not master_path.exists():
            # Fallback to full catalog load if no master.parquet
            self._load_data_full_catalog(start_date, end_date, compact)
            return
        
        # Load master with Polars for predicate pushdown (only loads matching rows)
        import polars as pl
        
        # Build filter expression
        filters = []
        if start_date:
            filters.append(pl.col("date") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            filters.append(pl.col("date") <= pl.lit(end_date).str.to_datetime())
        
        if filters:
            filter_expr = filters[0]
            for f in filters[1:]:
                filter_expr = filter_expr & f
            master_pl = pl.scan_parquet(master_path).filter(filter_expr).collect()
        else:
            master_pl = pl.read_parquet(master_path)
        
        # Convert to Pandas
        master = master_pl.to_pandas()
        del master_pl
        gc.collect()
        
        # Flatten if MultiIndex
        if isinstance(master.index, pd.MultiIndex):
            master = master.reset_index()
        
        # Downcast dtypes to save ~50% memory
        if compact:
            master = self._compact_dtypes(master)
        
        self._master = master
        
        # Load dates (small, no filtering needed at load time)
        dates = pd.read_parquet(dates_path)
        if start_date or end_date:
            mask = np.ones(len(dates), dtype=bool)
            if start_date:
                mask &= (dates['date'] >= start_date).values
            if end_date:
                mask &= (dates['date'] <= end_date).values
            dates = dates[mask]
        self._dates = dates
        
        gc.collect()
        
        self._load_time = time.perf_counter() - t0
        
        rows = len(self._master) if self._master is not None else 0
        mem_mb = self._master.memory_usage(deep=True).sum() / 1024 / 1024 if self._master is not None else 0
        print(f"Loaded in {self._load_time:.1f}s: {rows:,} rows, {mem_mb:.0f} MB")
    
    def _load_data_full_catalog(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compact: bool = True,
    ):
        """Fallback: Load via full catalog (slower, more memory)."""
        print("  (fallback: using full catalog load)")
        
        catalog = load_catalog(
            f"snapshots/{self.snapshot}",
            use_master=True,
            verify_integrity=False,
        )
        
        # Extract and flatten master
        master = catalog.get('master')
        if master is not None:
            if isinstance(master.index, pd.MultiIndex):
                master = master.reset_index()
            
            # Filter by date range
            if start_date or end_date:
                mask = np.ones(len(master), dtype=bool)
                if start_date:
                    mask &= (master['date'] >= start_date).values
                if end_date:
                    mask &= (master['date'] <= end_date).values
                master = master[mask]
            
            if compact:
                master = self._compact_dtypes(master)
            
            self._master = master
        
        # Extract and filter dates
        dates = catalog.get('dates')
        if dates is not None and (start_date or end_date):
            mask = np.ones(len(dates), dtype=bool)
            if start_date:
                mask &= (dates['date'] >= start_date).values
            if end_date:
                mask &= (dates['date'] <= end_date).values
            dates = dates[mask]
        self._dates = dates
        
        del catalog
        gc.collect()
    
    @staticmethod
    def _compact_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast dtypes to save memory (~50% reduction)."""
        # Columns safe to downcast to float32
        float_cols = [
            'ret', 'resret', 'openret', 'resopenret',
            'size', 'value', 'growth', 'leverage', 'volatility', 'momentum', 'yield',
            'mcap', 'adv',
        ]
        
        # Columns safe to downcast to int32
        int_cols = ['industry_id', 'sector_id', 'cap']
        
        for col in float_cols:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        
        for col in int_cols:
            if col in df.columns and df[col].dtype == np.int64:
                df[col] = df[col].astype(np.int32)
        
        return df
    
    def _filter_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Filter loaded data by date range (in place)."""
        if start_date is None and end_date is None:
            return
        
        if self._master is not None:
            mask = np.ones(len(self._master), dtype=bool)
            if start_date:
                mask &= (self._master['date'] >= start_date).values
            if end_date:
                mask &= (self._master['date'] <= end_date).values
            self._master = self._master[mask]
        
        if self._dates is not None:
            mask = np.ones(len(self._dates), dtype=bool)
            if start_date:
                mask &= (self._dates['date'] >= start_date).values
            if end_date:
                mask &= (self._dates['date'] <= end_date).values
            self._dates = self._dates[mask]
    
    @property
    def is_ready(self) -> bool:
        """Check if data is loaded."""
        return self._master is not None
    
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "snapshot": self.snapshot,
            "is_ready": self.is_ready,
            "load_time_seconds": self._load_time,
            "run_count": self._run_count,
            "master_rows": len(self._master) if self._master is not None else 0,
            "dates_count": len(self._dates) if self._dates is not None else 0,
        }
    
    def run(
        self,
        signal: pd.DataFrame,
        sigvar: str = "signal",
        byvar_list: Optional[List[str]] = None,
        fractile: tuple = (10, 90),
        weight: str = "equal",
        resid: bool = False,
        resid_style: str = "all",
        from_open: bool = False,
        mincos: int = 10,
        calc_turnover: bool = True,
    ) -> tuple:
        """
        Run backtest on a signal.
        
        Args:
            signal: DataFrame with columns [security_id, date_sig, date_ret, <sigvar>]
                   date_ret is required (pre-aligned signal)
            sigvar: Signal column name (default: "signal")
            byvar_list: Analysis groups (default: ["overall"])
            fractile: (short_pct, long_pct) thresholds (default: (10, 90))
            weight: "equal", "value", or "volume" (default: "equal")
            resid: Enable residualization (default: False)
            resid_style: "all" or "industry" (default: "all")
            from_open: Trade at open vs close (default: False)
            mincos: Min companies per side (default: 10)
            calc_turnover: Compute turnover/TC (default: True, False for speed)
            
        Returns:
            Tuple: (summary_df, daily_df, turnover_df, ff_df)
        """
        if not self.is_ready:
            raise RuntimeError("Service not ready. Call BacktestService.get() first.")
        
        if byvar_list is None:
            byvar_list = ["overall"]
        
        # Import here to avoid circular imports
        from backtest_wrapper import BacktestFastV2
        
        with self._lock:
            self._run_count += 1
        
        bt = BacktestFastV2(
            infile=signal,
            retfile=self._master,
            otherfile=self._master,
            datefile=self._dates,
            sigvar=sigvar,
            method="long_short",
            byvar_list=byvar_list,
            fractile=list(fractile),
            weight=weight,
            resid=resid,
            resid_style=resid_style,
            from_open=from_open,
            mincos=mincos,
            calc_turnover=calc_turnover,
            tc_model="naive",
            input_type="value",
            verbose=False,
        )
        
        return bt.gen_result()


# Convenience alias
def get_service(
    snapshot: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestService:
    """Convenience function to get BacktestService singleton."""
    return BacktestService.get(snapshot, start_date, end_date)
