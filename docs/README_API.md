# Programmatic Backtest API

Run backtests programmatically without the Streamlit UI. The API loads data once and keeps it in memory for fast, repeated execution.

## Quick Start

### Option 1: Direct Python (Recommended for Jupyter/Scripts)

```python
from api import BacktestService

# Initialize (loads data once, ~30s first time, instant after)
service = BacktestService.get(
    '2026-02-10-v1',
    start_date='2020-01-01',
    end_date='2021-12-31',
)

# Run a backtest (~0.5s per signal)
result = service.run(signal_df, sigvar='signal')
summary, daily, turnover, tc = result

# Access results
sharpe = summary.iloc[0]['sharpe_ret']
print(f"Sharpe: {sharpe:.2f}")
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

### BacktestService

```python
from api import BacktestService

service = BacktestService.get(
    snapshot="2026-02-10-v1",  # Data snapshot (default: latest)
    start_date="2020-01-01",   # Filter data start (optional)
    end_date="2023-12-31",     # Filter data end (optional)
    compact=True,              # Use float32 to save memory (default)
)
```

#### Methods

**`run(signal, **config)`** - Run a backtest

```python
result = service.run(
    signal=signal_df,
    sigvar="signal",            # Signal column name
    byvar_list=["overall"],     # Analysis groups
    fractile=(10, 90),          # Long/short percentiles
    weight="equal",             # "equal", "value", "volume"
    resid=False,                # Enable residualization
    resid_style="all",          # "all" or "industry"
    from_open=False,            # Trade at open vs close
    mincos=10,                  # Min companies per side
    calc_turnover=True,         # Set False for faster screening
)
# Returns: (summary_df, daily_df, turnover_df, tc_df)
```

**`stats()`** - Get service statistics

```python
service.stats()
# {'snapshot': '2026-02-10-v1', 'run_count': 5, 'master_rows': 2522327, ...}
```

**`reset()`** - Clear singleton and free memory

```python
BacktestService.reset()
```

### Result Format

`service.run()` returns a tuple of 4 DataFrames:

```python
summary, daily, turnover, tc = service.run(signal_df)

# Summary DataFrame - key metrics by group
summary.iloc[0]['sharpe_ret']    # Sharpe ratio
summary.iloc[0]['ret_ann']       # Annualized return
summary.iloc[0]['ret_std']       # Return std dev
summary.iloc[0]['maxdd']         # Max drawdown
summary.iloc[0]['turnover']      # Average turnover

# Daily DataFrame - time series
daily['date']      # Trading dates
daily['ret']       # Daily returns
daily['ret_net']   # Returns net of TC
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

## Parallel Execution

### Option 1: Threading (Simple, Shared Memory)

For quick parallelism with shared memory:

```python
from api import BacktestService
from joblib import Parallel, delayed

# Load data once
service = BacktestService.get(
    '2026-02-10-v1',
    start_date='2020-01-01',
    end_date='2021-12-31',
)

def run_backtest(signal):
    """Worker function - uses shared service."""
    result = service.run(signal, sigvar='signal', byvar_list=['overall'])
    return result[0].iloc[0]['sharpe_ret']

# Run in parallel (threading backend shares memory)
signals = [signal1, signal2, signal3, signal4]
results = Parallel(n_jobs=4, backend='threading')(
    delayed(run_backtest)(sig) for sig in signals
)
```

### Option 2: Multi-Process Worker Pool (True Parallelism)

For maximum throughput with true parallelism (bypasses GIL):

```python
# See tests/test_server_parallel.py for full implementation
from tests.test_server_parallel import BacktestWorkerPool, serialize_signal_arrow

# Start worker pool (each loads data independently)
pool = BacktestWorkerPool(n_workers=2, use_arrow=True)
pool.start(sequential=True)  # Sequential startup avoids memory spikes

# Prepare signals as Arrow bytes (fast transfer)
signals_arrow = [serialize_signal_arrow(sig) for sig in signals]

# Submit tasks
for arrow_bytes in signals_arrow:
    pool.submit(arrow_bytes)

# Collect results
results = pool.collect_all()
for task_id, worker_id, sharpe, elapsed, mem_delta in results:
    print(f"Task {task_id}: sharpe={sharpe:.2f}, time={elapsed:.2f}s")

pool.shutdown()
```

### Parallel Performance (10 years of data, 8 signals)

| Mode | Workers | Effective time/signal | Throughput | Memory |
|------|---------|----------------------|------------|--------|
| Sequential | 1 | 4.91s | 0.20/sec | 5GB |
| Threading | 2 | 2.64s | 0.38/sec | 10GB |
| **Multi-Process + Arrow** | 2 | **1.36s** | **0.74/sec** | 10GB |

### Signal Transfer Modes

| Mode | When to Use | Overhead |
|------|-------------|----------|
| **CSV path** | Signals on disk | ~1s read per signal |
| **Arrow IPC** | Signals in memory | ~0.1s serialize/deserialize |

Arrow mode is **2.2x faster** when signals are already in DataFrames.

### Key Points

- **Threading**: Simple, shares memory, GIL-limited (~1.8x speedup)
- **Multi-Process**: True parallelism, separate memory per worker
- **Sequential startup**: Start workers one-at-a-time to avoid memory spikes
- **Arrow IPC**: Fastest transfer for in-memory DataFrames

### Alternative: Skip Turnover for Screening

For fast signal screening, disable turnover calculation:

```python
# Fast screening pass (~0.15s per signal)
results = []
for signal in signals:
    result = service.run(signal, calc_turnover=False)
    results.append(result[0].iloc[0]['sharpe_ret'])

# Full backtest only on top candidates
top_signals = [s for s, r in zip(signals, results) if r > 1.5]
```

## Memory Usage

| Mode | RAM Usage | Notes |
|------|-----------|-------|
| Compact (default) | ~200 MB | float32 dtypes, 2-year data |
| Full 10-year data | ~1 GB | Compact mode |
| Full dtypes | ~2 GB | float64 dtypes |

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
├── __init__.py            # Package exports (BacktestService)
├── service.py             # Core singleton service (~300 lines)
├── server.py              # FastAPI HTTP server (optional)
└── backtest_config.yaml   # Configuration file

tests/
├── test_server_parallel.py  # Multi-process worker pool (true parallelism)
└── test_equivalence.py      # Result equivalence tests

docker/
├── Dockerfile             # Container definition
└── docker-compose.yml     # Docker orchestration
```
