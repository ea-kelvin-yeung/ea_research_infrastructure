"""
Backtest API Package

Programmatic interface for running backtests without the Streamlit UI.

Quick Start:
    from api import BacktestClient
    
    client = BacktestClient()
    result = client.run(signal_df, lag=0, resid="off")
    print(result.sharpe)

For server mode:
    from api import BacktestClient
    client = BacktestClient(server_url="http://localhost:8000")
"""

from .service import BacktestService, ServiceConfig, get_service
from .client import BacktestClient, quick_backtest

__all__ = [
    "BacktestService",
    "ServiceConfig", 
    "BacktestClient",
    "get_service",
    "quick_backtest",
]
