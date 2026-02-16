"""
Backtest API Package

Fast programmatic interface for backtesting signals.

Quick Start:
    from api import BacktestService
    
    # Load data once (~30s, then instant)
    service = BacktestService.get(start_date='2012-01-01', end_date='2021-12-31')
    
    # Run backtest (~1.8s per signal)
    result = service.run(signal_df)
    sharpe = result[0].iloc[0]['sharpe_ret']
"""

from .service import BacktestService, get_service

__all__ = [
    "BacktestService",
    "get_service",
]
