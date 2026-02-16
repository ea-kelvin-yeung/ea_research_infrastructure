"""
BacktestServer: FastAPI server for running backtests via HTTP API.

This server loads data once at startup and keeps it in memory, allowing
multiple clients to run backtests without reloading data each time.

Start the server:
    python -m api.server --port 8000
    
Or with uvicorn directly:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /run     - Run a single backtest
    POST /suite   - Run a suite of backtests
    GET  /health  - Health check
    GET  /stats   - Server statistics
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .service import BacktestService, ServiceConfig
from poc.wrapper import BacktestResult


# ============================================================================
# Pydantic Models for API
# ============================================================================

class SignalRecord(BaseModel):
    """Single row in the signal DataFrame."""
    security_id: int
    date_sig: str
    signal: float
    date_avail: Optional[str] = None
    date_ret: Optional[str] = None
    date_openret: Optional[str] = None


class BacktestConfigModel(BaseModel):
    """Backtest configuration."""
    lag: int = 0
    resid: str = "off"
    byvar_list: List[str] = ["overall", "year", "cap"]
    fractile: List[int] = [10, 90]
    weight: str = "equal"
    tc_model: str = "naive"
    from_open: bool = False
    mincos: int = 10


class RunRequest(BaseModel):
    """Request body for /run endpoint."""
    signal: List[Dict[str, Any]]  # List of signal records
    config: Optional[BacktestConfigModel] = None


class SuiteRequest(BaseModel):
    """Request body for /suite endpoint."""
    signal: List[Dict[str, Any]]
    lags: List[int] = [0]
    resid_modes: List[str] = ["off"]
    config: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    snapshot: str
    is_ready: bool


class StatsResponse(BaseModel):
    """Response for /stats endpoint."""
    snapshot: str
    is_ready: bool
    load_time_seconds: float
    run_count: int
    master_rows: int
    dates_count: int
    uptime_seconds: float


# ============================================================================
# Server State
# ============================================================================

class ServerState:
    """Global server state."""
    service: Optional[BacktestService] = None
    start_time: float = 0
    snapshot: str = ""
    

state = ServerState()


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: Load data
    print(f"Starting backtest server...")
    state.start_time = time.time()
    
    # Service is initialized via command line args before uvicorn starts
    # If not initialized, use defaults
    if state.service is None:
        print("Initializing service with default snapshot...")
        state.service = BacktestService.get()
        state.snapshot = state.service.snapshot
    
    print(f"Server ready! Snapshot: {state.snapshot}")
    yield
    # Shutdown
    print("Shutting down backtest server...")


app = FastAPI(
    title="Backtest Server",
    description="HTTP API for running backtests with persistent data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def signal_to_dataframe(signal_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert signal list to DataFrame with proper types."""
    df = pd.DataFrame(signal_data)
    
    # Convert date columns
    date_cols = ['date_sig', 'date_avail', 'date_ret', 'date_openret']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure security_id is int
    if 'security_id' in df.columns:
        df['security_id'] = df['security_id'].astype(np.int32)
    
    return df


def result_to_dict(result: BacktestResult) -> Dict[str, Any]:
    """Convert BacktestResult to JSON-serializable dict."""
    return {
        "summary": result.summary.to_dict(orient="records"),
        "daily": result.daily.to_dict(orient="records") if result.daily is not None else [],
        "fractile": result.fractile.to_dict(orient="records") if result.fractile is not None else None,
        "config": result.config,
        "sharpe": result.sharpe if not np.isnan(result.sharpe) else None,
        "annual_return": result.annual_return if not np.isnan(result.annual_return) else None,
        "max_drawdown": result.max_drawdown if not np.isnan(result.max_drawdown) else None,
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if state.service and state.service.is_ready else "not_ready",
        snapshot=state.snapshot,
        is_ready=state.service is not None and state.service.is_ready,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get server statistics."""
    if state.service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    svc_stats = state.service.stats()
    return StatsResponse(
        snapshot=svc_stats.get("snapshot", ""),
        is_ready=svc_stats.get("is_ready", False),
        load_time_seconds=svc_stats.get("load_time_seconds", 0),
        run_count=svc_stats.get("run_count", 0),
        master_rows=svc_stats.get("master_rows", 0),
        dates_count=svc_stats.get("dates_count", 0),
        uptime_seconds=time.time() - state.start_time,
    )


@app.post("/run")
async def run_backtest(request: RunRequest):
    """
    Run a single backtest.
    
    Request body:
    {
        "signal": [{"security_id": 1, "date_sig": "2020-01-02", "signal": 0.5}, ...],
        "config": {"lag": 0, "resid": "off", ...}
    }
    """
    if state.service is None or not state.service.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Convert signal to DataFrame
        signal_df = signal_to_dataframe(request.signal)
        
        # Get config
        config = request.config or BacktestConfigModel()
        
        # Run backtest
        result = state.service.run(
            signal=signal_df,
            lag=config.lag,
            resid=config.resid,
            byvar_list=config.byvar_list,
            fractile=tuple(config.fractile),
            weight=config.weight,
            tc_model=config.tc_model,
            from_open=config.from_open,
            mincos=config.mincos,
        )
        
        return result_to_dict(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suite")
async def run_suite(request: SuiteRequest):
    """
    Run a suite of backtests with multiple configurations.
    
    Request body:
    {
        "signal": [...],
        "lags": [0, 1, 2],
        "resid_modes": ["off", "all"],
        "config": {...}
    }
    """
    if state.service is None or not state.service.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        signal_df = signal_to_dataframe(request.signal)
        extra_config = request.config or {}
        
        results = state.service.run_suite(
            signal=signal_df,
            lags=request.lags,
            resid_modes=request.resid_modes,
            **extra_config,
        )
        
        return {key: result_to_dict(result) for key, result in results.items()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Start the backtest server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Data snapshot to load (default: latest)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (use 1 for shared state)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    # Initialize service before starting server
    print(f"Loading data snapshot: {args.snapshot or 'latest'}...")
    state.service = BacktestService.get(args.snapshot)
    state.snapshot = state.service.snapshot
    
    # Start server
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
