"""
BacktestClient: Simple Python SDK for running backtests programmatically.

This module provides a user-friendly API for running backtests without dealing
with data loading or configuration details. It can operate in two modes:

1. Direct mode (default): Uses BacktestService singleton, loads data in-process
2. Server mode: Connects to a running backtest server via HTTP

Usage (Direct mode - recommended for Jupyter/scripts):
    from api import BacktestClient
    
    client = BacktestClient()  # Loads data once (~30s first time)
    result = client.run(signal_df)
    print(result.sharpe)

Usage (Server mode - for shared server):
    from api import BacktestClient
    client = BacktestClient(server_url="http://localhost:8000")
    result = client.run(signal_df)
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .service import BacktestService, ServiceConfig, get_service
from poc.wrapper import BacktestResult, BacktestConfig


class BacktestClient:
    """
    Simple client for running backtests programmatically.
    
    Example:
        client = BacktestClient()
        result = client.run(signal_df, lag=0, resid="off")
        print(f"Sharpe: {result.sharpe:.2f}")
    """
    
    def __init__(
        self,
        snapshot: Optional[str] = None,
        server_url: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **service_kwargs,
    ):
        """
        Initialize the backtest client.
        
        Args:
            snapshot: Data snapshot to use (e.g., "2026-02-10-v1"). None = latest.
            server_url: If provided, connect to a backtest server instead of loading locally.
                       Example: "http://localhost:8000"
            start_date: Filter data to start date (e.g., "2020-01-01")
            end_date: Filter data to end date (e.g., "2023-12-31")
            **service_kwargs: Additional options for ServiceConfig (use_mmap, cache_size, etc.)
        """
        self.server_url = server_url
        self._service: Optional[BacktestService] = None
        
        if server_url:
            # Server mode - validate URL
            self._mode = "server"
            try:
                import requests
                self._requests = requests
            except ImportError:
                raise ImportError("requests package required for server mode: pip install requests")
        else:
            # Direct mode - get service singleton
            self._mode = "direct"
            config = ServiceConfig(**service_kwargs) if service_kwargs else None
            self._service = BacktestService.get(snapshot, config)
            
            # Apply date filters if specified
            if start_date or end_date:
                self._service.filter_dates(start_date, end_date)
    
    @property
    def mode(self) -> str:
        """Get client mode: 'direct' or 'server'."""
        return self._mode
    
    @property
    def is_ready(self) -> bool:
        """Check if client is ready to run backtests."""
        if self._mode == "server":
            try:
                response = self._requests.get(f"{self.server_url}/health", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        else:
            return self._service is not None and self._service.is_ready
    
    def stats(self) -> Dict[str, Any]:
        """Get client/service statistics."""
        if self._mode == "server":
            try:
                response = self._requests.get(f"{self.server_url}/stats", timeout=5)
                return response.json()
            except Exception as e:
                return {"error": str(e), "mode": "server", "server_url": self.server_url}
        else:
            stats = self._service.stats()
            stats["mode"] = "direct"
            return stats
    
    def run(
        self,
        signal: pd.DataFrame,
        lag: int = 0,
        resid: str = "off",
        byvar_list: Optional[List[str]] = None,
        fractile: tuple = (10, 90),
        weight: str = "equal",
        tc_model: str = "naive",
        from_open: bool = False,
        mincos: int = 10,
        validate: bool = True,
    ) -> BacktestResult:
        """
        Run a single backtest.
        
        Args:
            signal: Signal DataFrame with columns [security_id, date_sig, signal]
                   Optional: date_avail, date_ret, date_openret
            lag: Trading lag in days (default: 0)
            resid: Residualization mode: "off", "industry", "all" (default: "off")
            byvar_list: Analysis slices (default: ["overall", "year", "cap"])
            fractile: Long/short percentile thresholds (default: (10, 90))
            weight: Weighting method: "equal", "value", "volume" (default: "equal")
            tc_model: Transaction cost model: "naive", "power_law" (default: "naive")
            from_open: Trade at open vs close (default: False)
            mincos: Minimum companies per side (default: 10)
            validate: Validate signal format (default: True)
            
        Returns:
            BacktestResult with .sharpe, .annual_return, .max_drawdown, .summary, .daily
        """
        if byvar_list is None:
            byvar_list = ["overall", "year", "cap"]
        
        if self._mode == "server":
            return self._run_server(
                signal=signal,
                lag=lag,
                resid=resid,
                byvar_list=byvar_list,
                fractile=fractile,
                weight=weight,
                tc_model=tc_model,
                from_open=from_open,
                mincos=mincos,
            )
        else:
            return self._service.run(
                signal=signal,
                lag=lag,
                resid=resid,
                byvar_list=byvar_list,
                fractile=fractile,
                weight=weight,
                tc_model=tc_model,
                from_open=from_open,
                mincos=mincos,
                validate=validate,
            )
    
    def _run_server(self, signal: pd.DataFrame, **config) -> BacktestResult:
        """Run backtest via server."""
        # Convert signal to JSON-serializable format
        signal_dict = signal.copy()
        
        # Convert datetime columns to strings
        for col in signal_dict.columns:
            if pd.api.types.is_datetime64_any_dtype(signal_dict[col]):
                signal_dict[col] = signal_dict[col].dt.strftime('%Y-%m-%d')
        
        payload = {
            "signal": signal_dict.to_dict(orient="records"),
            "config": config,
        }
        
        response = self._requests.post(
            f"{self.server_url}/run",
            json=payload,
            timeout=120,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.status_code} - {response.text}")
        
        result_data = response.json()
        
        # Reconstruct BacktestResult from JSON
        return BacktestResult(
            summary=pd.DataFrame(result_data["summary"]),
            daily=pd.DataFrame(result_data["daily"]),
            fractile=pd.DataFrame(result_data["fractile"]) if result_data.get("fractile") else None,
            config=result_data.get("config", config),
        )
    
    def run_suite(
        self,
        signal: pd.DataFrame,
        lags: Optional[List[int]] = None,
        resid_modes: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Run multiple backtests with different configurations.
        
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
        
        if self._mode == "server":
            return self._run_suite_server(signal, lags, resid_modes, **kwargs)
        else:
            return self._service.run_suite(signal, lags, resid_modes, **kwargs)
    
    def _run_suite_server(
        self,
        signal: pd.DataFrame,
        lags: List[int],
        resid_modes: List[str],
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """Run suite via server."""
        # Convert signal to JSON-serializable format
        signal_dict = signal.copy()
        for col in signal_dict.columns:
            if pd.api.types.is_datetime64_any_dtype(signal_dict[col]):
                signal_dict[col] = signal_dict[col].dt.strftime('%Y-%m-%d')
        
        payload = {
            "signal": signal_dict.to_dict(orient="records"),
            "lags": lags,
            "resid_modes": resid_modes,
            "config": kwargs,
        }
        
        response = self._requests.post(
            f"{self.server_url}/suite",
            json=payload,
            timeout=300,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.status_code} - {response.text}")
        
        results_data = response.json()
        
        results = {}
        for key, result_data in results_data.items():
            results[key] = BacktestResult(
                summary=pd.DataFrame(result_data["summary"]),
                daily=pd.DataFrame(result_data["daily"]),
                fractile=pd.DataFrame(result_data["fractile"]) if result_data.get("fractile") else None,
                config=result_data.get("config", {}),
            )
        
        return results
    
    def batch(
        self,
        signals: List[pd.DataFrame],
        **kwargs,
    ) -> List[BacktestResult]:
        """
        Run backtests on multiple signals.
        
        Args:
            signals: List of signal DataFrames
            **kwargs: Arguments passed to run()
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        for i, signal in enumerate(signals):
            if (i + 1) % 10 == 0:
                print(f"Processing signal {i + 1}/{len(signals)}")
            results.append(self.run(signal, **kwargs))
        return results
    
    def compare(
        self,
        signals: Dict[str, pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compare multiple signals and return a summary table.
        
        Args:
            signals: Dict mapping signal names to DataFrames
            **kwargs: Arguments passed to run()
            
        Returns:
            DataFrame with comparison metrics (Sharpe, Ann. Return, etc.) per signal
        """
        rows = []
        for name, signal in signals.items():
            result = self.run(signal, **kwargs)
            rows.append({
                "signal": name,
                "sharpe": result.sharpe,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown,
                "turnover": result.turnover,
            })
        return pd.DataFrame(rows)


# Convenience function
def quick_backtest(signal: pd.DataFrame, **kwargs) -> BacktestResult:
    """
    One-liner for running a quick backtest.
    
    Example:
        from api import quick_backtest
        result = quick_backtest(my_signal, lag=0)
        print(result.sharpe)
    """
    client = BacktestClient()
    return client.run(signal, **kwargs)
