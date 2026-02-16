# Programmatic Backtest API

Run backtests programmatically without the Streamlit UI. The API loads data once and keeps it in memory for fast, repeated execution.

## Quick Start

### Option 1: Direct Python (Recommended for Jupyter/Scripts)

```python
from api import BacktestClient

# Initialize (loads data once, ~30s first time, instant after)
client = BacktestClient()

# Run a backtest
result = client.run(signal_df, lag=0, resid="off")
print(f"Sharpe: {result.sharpe:.2f}")
print(f"Annual Return: {result.annual_return:.2%}")

# Access detailed results
result.summary  # DataFrame with stats by group (overall, year, cap)
result.daily    # DataFrame with daily returns
```

### Option 2: Local Server (For multiple scripts/notebooks)

Start the server once to keep data in memory:

```bash
# Terminal 1: Start server
python -m api.server --port 8000

# Terminal 2+: Run backtests from any script
```

```python
from api import BacktestClient

# Connect to server (no data loading needed)
client = BacktestClient(server_url="http://localhost:8000")
result = client.run(signal_df)
```

### Option 3: Docker

```bash
# Build and run (from project root)
docker-compose -f docker/docker-compose.yml up -d

# Or build manually
docker build -f docker/Dockerfile -t backtest-server .
docker run -d -p 8000:8000 -v $(pwd)/snapshots:/app/snapshots:ro backtest-server

# Use from any client
curl http://localhost:8000/health
```

## API Reference

### BacktestClient

```python
from api import BacktestClient

client = BacktestClient(
    snapshot="2026-02-10-v1",  # Data snapshot (default: latest)
    server_url=None,           # HTTP server URL (default: direct mode)
    start_date="2020-01-01",   # Filter data start (optional)
    end_date="2023-12-31",     # Filter data end (optional)
)
```

#### Methods

**`run(signal, **config)`** - Run a single backtest

```python
result = client.run(
    signal=signal_df,
    lag=0,              # Trading lag in days
    resid="off",        # "off", "industry", "all"
    byvar_list=["overall", "year", "cap"],
    fractile=(10, 90),  # Long/short percentiles
    weight="equal",     # "equal", "value", "volume"
    tc_model="naive",   # "naive", "power_law"
)
```

**`run_suite(signal, lags, resid_modes)`** - Run multiple configs

```python
results = client.run_suite(
    signal=signal_df,
    lags=[0, 1, 2],
    resid_modes=["off", "all"],
)
# Returns dict: {"lag0_residoff": BacktestResult, "lag0_residall": BacktestResult, ...}
```

**`batch(signals)`** - Run on multiple signals

```python
signals = [signal1, signal2, signal3]
results = client.batch(signals, lag=0)  # List of BacktestResult
```

**`compare(signals)`** - Compare multiple signals

```python
signals = {"momentum": mom_signal, "value": val_signal}
comparison = client.compare(signals)  # DataFrame with Sharpe, Return, etc.
```

### BacktestResult

```python
result.sharpe         # Overall Sharpe ratio (float)
result.annual_return  # Annualized return (float)
result.max_drawdown   # Maximum drawdown (float)
result.turnover       # Average turnover (float)
result.summary        # DataFrame with all metrics by group
result.daily          # DataFrame with daily returns
result.fractile       # DataFrame with fractile analysis
result.config         # Dict of config used
```

### Server API (HTTP)

When running `python -m api.server`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Server statistics |
| `/run` | POST | Run single backtest |
| `/suite` | POST | Run backtest suite |

**Example request:**
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [{"security_id": 1, "date_sig": "2020-01-02", "signal": 0.5}],
    "config": {"lag": 0, "resid": "off"}
  }'
```

## Signal DataFrame Format

Your signal DataFrame must have these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `security_id` | int | Yes | Security identifier |
| `date_sig` | datetime | Yes | Signal date |
| `signal` | float | Yes | Signal value (higher = more bullish) |
| `date_avail` | datetime | No | When signal becomes available |
| `date_ret` | datetime | No | Return measurement date |

Example:
```python
signal_df = pd.DataFrame({
    'security_id': [1, 2, 3],
    'date_sig': pd.to_datetime(['2020-01-02', '2020-01-02', '2020-01-02']),
    'signal': [0.5, -0.3, 0.1],
})
```

## Configuration

Edit `api/backtest_config.yaml` to customize defaults:

```yaml
snapshot: "2026-02-10-v1"

server:
  host: "127.0.0.1"
  port: 8000

memory:
  use_mmap: true
  cache_size: 10

defaults:
  lag: 0
  resid: "off"
  byvar_list: ["overall", "year", "cap"]
```

## Memory Usage

| Mode | RAM Usage | Notes |
|------|-----------|-------|
| Direct (default) | ~500MB | Hot data in memory |
| With residualization | ~1.2GB | Risk factors loaded |
| Memory-mapped | ~100MB | Uses disk, slower |

## Troubleshooting

**"No snapshots found"**
- Ensure you have data in `snapshots/` directory
- Create a snapshot: see `poc/catalog.py`

**Server connection refused**
- Check server is running: `curl http://localhost:8000/health`
- Check port is not in use: `lsof -i :8000`

**Slow first run**
- First call loads data (~30s)
- Subsequent calls use cached data (<1s)

## Project Structure

```
api/
├── __init__.py            # Package exports (BacktestClient, BacktestService)
├── client.py              # Python SDK
├── service.py             # Core singleton service
├── server.py              # FastAPI HTTP server
└── backtest_config.yaml   # Configuration file

docker/
├── Dockerfile        # Container definition
└── docker-compose.yml # Docker orchestration
```
