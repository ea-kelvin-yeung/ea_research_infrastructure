"""
Backtest Wrapper: Thin wrapper around existing Backtest class.
~150 lines - stable API with clean result dataclass.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import backtest_engine
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest_engine import Backtest

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
    
    # Map residualize setting to Backtest parameters
    resid = config.residualize != 'off'
    resid_style = config.residualize if resid else 'all'
    
    # Instantiate and run backtest
    bt = Backtest(
        infile=aligned_signal,
        retfile=catalog['ret'],
        otherfile=catalog['risk'],
        datefile=catalog['dates'],
        sigvar='signal',
        method='long_short',
        byvar_list=config.byvar_list,
        from_open=config.from_open,
        input_type='value',
        mincos=config.mincos,
        fractile=list(config.fractile),
        weight=config.weight,
        tc_model=config.tc_model,
        resid=resid,
        resid_style=resid_style,
        output='simple',
        verbose=False,
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
