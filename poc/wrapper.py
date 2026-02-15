"""
Backtest Wrapper: Thin wrapper around Backtest/BacktestFastV2 classes.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import backtest engines
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest_engine import Backtest
from backtest_engine_minimal_fast import BacktestFastV2

from .contract import prepare_signal


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    lag: int = 0
    residualize: str = 'off'  # 'off', 'industry', 'factor', 'all'
    tc_model: str = 'naive'
    weight: str = 'equal'
    fractile: tuple = (10, 90)
    from_open: bool = False
    mincos: int = 10
    byvar_list: list = field(default_factory=lambda: ['overall', 'year', 'cap'])
    
    def to_dict(self) -> dict:
        return {
            'lag': self.lag,
            'residualize': self.residualize,
            'tc_model': self.tc_model,
            'weight': self.weight,
            'fractile': self.fractile,
            'from_open': self.from_open,
            'mincos': self.mincos,
        }
    
    def config_key(self) -> str:
        """Unique string key for this config."""
        return f"lag{self.lag}_resid{self.residualize}"


@dataclass
class BacktestResult:
    """Result from a single backtest run."""
    summary: pd.DataFrame           # Overall + slices stats
    daily: pd.DataFrame             # Daily series (pnl, cumret, drawdown, turnover)
    fractile: Optional[pd.DataFrame]  # Fractile analysis (can be None)
    config: Dict[str, Any]          # Config used
    
    @property
    def sharpe(self) -> float:
        """Get overall Sharpe ratio."""
        overall = self.summary[self.summary['group'] == 'overall']
        if len(overall) > 0 and 'sharpe_ret' in overall.columns:
            return float(overall['sharpe_ret'].iloc[0])
        return np.nan
    
    @property
    def annual_return(self) -> float:
        """Get overall annualized return."""
        overall = self.summary[self.summary['group'] == 'overall']
        if len(overall) > 0 and 'ret_ann' in overall.columns:
            return float(overall['ret_ann'].iloc[0])
        return np.nan
    
    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        overall = self.summary[self.summary['group'] == 'overall']
        if len(overall) > 0 and 'maxdraw' in overall.columns:
            return float(overall['maxdraw'].iloc[0])
        return np.nan
    
    @property
    def turnover(self) -> float:
        """Get average turnover."""
        overall = self.summary[self.summary['group'] == 'overall']
        if len(overall) > 0 and 'turnover' in overall.columns:
            return float(overall['turnover'].iloc[0])
        return np.nan


def run_backtest(
    signal_df: pd.DataFrame,
    catalog: dict,
    config: Optional[BacktestConfig] = None,
    validate: bool = True,
) -> BacktestResult:
    """
    Run a backtest with the given signal and configuration.
    
    Args:
        signal_df: Signal DataFrame with columns [security_id, date_sig, date_avail, signal]
        catalog: Data catalog dict with keys ['ret', 'risk', 'dates']
        config: Backtest configuration (default: BacktestConfig())
        validate: Whether to validate the signal (default: True)
        
    Returns:
        BacktestResult with summary, daily series, and fractile analysis
    """
    if config is None:
        config = BacktestConfig()
    
    # Prepare signal (validate + align dates)
    aligned_signal = prepare_signal(
        signal_df,
        catalog['dates'],
        lag=config.lag,
        validate=validate,
    )
    
    # Normalize signal security_id to int32 to match catalog data
    # (catalog data was normalized at snapshot creation time)
    if aligned_signal['security_id'].dtype != np.int32:
        aligned_signal = aligned_signal.copy()
        aligned_signal['security_id'] = aligned_signal['security_id'].astype(np.int32)
    
    # Map residualize setting to Backtest parameters
    resid = config.residualize != 'off'
    resid_style = config.residualize if resid else 'all'
    
    # Use BacktestFastV2 if master data is available (1.8x faster than BacktestFast)
    use_fast = 'master' in catalog and catalog['master'] is not None
    BacktestClass = BacktestFastV2 if use_fast else Backtest
    
    # Instantiate and run backtest
    bt_kwargs = {
        'infile': aligned_signal,
        'retfile': catalog['ret'],
        'otherfile': catalog['risk'],
        'datefile': catalog['dates'],
        'sigvar': 'signal',
        'method': 'long_short',
        'byvar_list': config.byvar_list,
        'from_open': config.from_open,
        'input_type': 'value',
        'mincos': config.mincos,
        'fractile': list(config.fractile),
        'weight': config.weight,
        'tc_model': config.tc_model,
        'resid': resid,
        'resid_style': resid_style,
        'output': 'simple',
        'verbose': False,
    }
    
    if use_fast:
        bt_kwargs['master_data'] = catalog['master']
    
    bt = BacktestClass(**bt_kwargs)
    
    # Inject pre-computed indexes and Polars DataFrames for maximum performance
    if use_fast and hasattr(bt, 'set_precomputed_indexes'):
        bt.set_precomputed_indexes(
            dates_indexed=catalog.get('dates_indexed'),
            asof_tables=catalog.get('asof_tables'),
            master_pl=catalog.get('master_pl'),
            otherfile_pl=catalog.get('otherfile_pl'),
            retfile_pl=catalog.get('retfile_pl'),
            datefile_pl=catalog.get('datefile_pl'),
            asof_tables_pl=catalog.get('asof_tables_pl'),
        )
    
    # Run and extract results
    result = bt.gen_result()
    
    # Unpack results (gen_result returns different tuple sizes)
    if len(result) == 4:
        summary, daily, fractile, ff_result = result
    elif len(result) == 3:
        summary, daily, fractile = result
    else:
        summary, daily = result[:2]
        fractile = None
    
    return BacktestResult(
        summary=summary,
        daily=daily,
        fractile=fractile,
        config=config.to_dict(),
    )


def run_backtest_simple(
    signal_df: pd.DataFrame,
    catalog: dict,
    lag: int = 0,
    residualize: str = 'off',
    **kwargs,
) -> BacktestResult:
    """
    Convenience function: run backtest with keyword arguments.
    """
    config = BacktestConfig(lag=lag, residualize=residualize, **kwargs)
    return run_backtest(signal_df, catalog, config)
