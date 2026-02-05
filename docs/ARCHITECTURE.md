# Backtest PoC Infrastructure - Architecture & Design

> A standardized, reproducible backtesting platform with one-button signal evaluation.

## Overview

This infrastructure wraps an existing backtest engine to provide:
- **Signal contract validation** - Standardized input format
- **Data snapshots** - Versioned, reproducible data
- **Suite runner** - Grid search over lag/residualization configs
- **Baseline comparisons** - Compare against reversal, momentum, value signals
- **MLflow tracking** - Log runs, metrics, and artifacts
- **Tear sheet generation** - HTML report with verdict
- **Streamlit UI** - Drag-and-drop signal evaluation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │Run Suite │  │ Results  │  │ History  │                       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
└───────┼─────────────┼─────────────┼─────────────────────────────┘
        │             │             │
        ▼             │             │
┌───────────────┐     │             │
│   contract    │     │             │
│  (validate)   │     │             │
└───────┬───────┘     │             │
        │             │             │
        ▼             │             │
┌───────────────┐     │             │
│   catalog     │     │             │
│ (load data)   │     │             │
└───────┬───────┘     │             │
        │             │             │
        ▼             │             │
┌───────────────┐     │             │
│   wrapper     │     │             │
│ (run backtest)│     │             │
└───────┬───────┘     │             │
        │             │             │
        ▼             │             │
┌───────────────┐     │             │
│    suite      │◄────┴─────────────┤
│ (grid search) │                   │
└───────┬───────┘                   │
        │                           │
        ├──────────┐                │
        ▼          ▼                ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ tearsheet │ │ tracking  │ │  MLflow   │
│  (HTML)   │ │ (log run) │ │   (DB)    │
└───────────┘ └───────────┘ └───────────┘
```

---

## Project Structure

```
ea_research_infrastructure/
├── backtest_engine.py      # Core backtest engine (original)
├── backtest_engine_fast.py # Optimized backtest engine (3-4x faster)
├── run_demo.py             # Demo script
├── requirements.txt        # Dependencies
│
├── poc/                    # PoC modules (flat structure)
│   ├── __init__.py
│   ├── contract.py         # Signal schema validation
│   ├── catalog.py          # Data snapshot management
│   ├── wrapper.py          # Backtest API wrapper
│   ├── baselines.py        # Baseline signal generators
│   ├── suite.py            # Suite runner (grid search)
│   ├── tracking.py         # MLflow logging
│   ├── tearsheet.py        # HTML report generator
│   └── app.py              # Streamlit UI
│
├── data/                   # Raw data files
│   ├── ret.parquet         # Returns data
│   ├── risk.parquet        # Risk factors
│   ├── trading_date.pkl    # Trading calendar
│   ├── descriptor.parquet  # Universe flags
│   └── signal_sample.pkl   # Sample signal
│
├── snapshots/              # Versioned data snapshots
│   └── YYYY-MM-DD-vN/
│       ├── manifest.json
│       ├── ret.parquet
│       ├── risk.parquet
│       ├── trading_date.parquet
│       └── master.parquet  # Pre-merged for fast joins
│
├── artifacts/              # Generated outputs
│   ├── *_tearsheet.html
│   ├── *_summary.csv
│   └── *_daily.parquet
│
└── mlruns/                 # MLflow tracking data
```

---

## Module Details

### 1. `poc/contract.py` - Signal Contract

Validates and aligns input signals to the required schema.

**Required columns:**
| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Unique security identifier |
| `date_sig` | datetime | Signal calculation date |
| `date_avail` | datetime | When signal becomes available |
| `signal` | float | Signal value |

**Key functions:**
```python
validate_signal(df) -> (bool, List[str])  # Validate schema
align_dates(df, datefile, lag=0) -> df    # Align to trading dates
prepare_signal(df, datefile, lag=0) -> df # Validate + align
```

---

### 2. `poc/catalog.py` - Data Catalog

Manages versioned data snapshots for reproducibility.

**Key functions:**
```python
create_snapshot(source_dir, output_dir) -> Path  # Create new snapshot
load_catalog(snapshot_path) -> dict              # Load snapshot data
list_snapshots(snapshots_dir) -> List[str]       # List available snapshots
```

**Catalog dict structure:**
```python
{
    'ret': pd.DataFrame,      # Returns data
    'risk': pd.DataFrame,     # Risk factors
    'dates': pd.DataFrame,    # Trading calendar
    'master': pd.DataFrame,   # Pre-merged ret+risk (indexed)
    'snapshot_id': str,       # Snapshot identifier
}
```

**Master Data (Performance Optimization):**

The `master` DataFrame is a pre-merged, indexed version of `ret` and `risk` data, created automatically when loading a catalog. It provides **3-4x speedup** for backtests by avoiding repeated merge operations.

```python
# Master data columns (indexed on security_id, date):
# From ret: ret, resret, openret, resopenret, vol, adv, close_adj
# From risk: mcap, cap, industry_id, sector_id, size, value, growth, 
#            leverage, volatility, momentum, yield
```

The master data is cached to `master.parquet` in the snapshot directory for fast loading.

---

### 3. `poc/wrapper.py` - Backtest Wrapper

Thin, stable API wrapper around the core `Backtest` class.

**Config dataclass:**
```python
@dataclass
class BacktestConfig:
    lag: int = 0
    residualize: str = 'off'  # 'off', 'industry', 'all'
    tc_model: str = 'naive'
    weight: str = 'equal'
    fractile: tuple = (10, 90)
    from_open: bool = False
    mincos: int = 10
```

**Result dataclass:**
```python
@dataclass
class BacktestResult:
    summary: pd.DataFrame   # Stats by group (overall, year, cap)
    daily: pd.DataFrame     # Daily series (cumret, drawdown, turnover)
    fractile: pd.DataFrame  # Fractile analysis
    config: dict            # Config used

    @property sharpe -> float
    @property annual_return -> float
    @property max_drawdown -> float
    @property turnover -> float
```

---

### 4. `poc/baselines.py` - Baseline Library

Standard signals for comparison.

**Available baselines:**
| Signal | Description | Lookback |
|--------|-------------|----------|
| `reversal_5d` | -1 × past 5-day return | 5 days |
| `momentum_12_1` | 12-month return minus last month | 252 days |
| `value` | Value factor from risk file | N/A |

**Caching:** Results cached to `.cache/baselines/` using joblib.

**Key functions:**
```python
generate_reversal_signal(catalog, lookback=5, start_date, end_date) -> df
generate_momentum_signal(catalog, lookback=252, skip=21, ...) -> df
generate_value_signal(catalog, start_date, end_date) -> df
generate_all_baselines(catalog, start_date, end_date) -> Dict[str, df]
compute_signal_correlation(signal_df, baseline_df) -> float
clear_baseline_cache()  # Clear disk cache
```

---

### 5. `poc/suite.py` - Suite Runner

Runs grid of backtest configurations.

**Default grid:**
```python
DEFAULT_GRID = {
    'lags': [0, 1, 2],
    'residualize': ['off', 'industry'],
}
```

**SuiteResult dataclass:**
```python
@dataclass
class SuiteResult:
    results: Dict[str, BacktestResult]    # config_key -> result
    baselines: Dict[str, BacktestResult]  # baseline_name -> result
    summary: pd.DataFrame                  # All configs summarized
    correlations: pd.DataFrame             # Signal/PnL correlations

    @property best_sharpe -> float
    @property best_config -> str
```

**Key functions:**
```python
run_suite(signal_df, catalog, grid=None, include_baselines=True, n_jobs=1) -> SuiteResult
get_best_config(suite_result, metric='sharpe') -> str
```

---

### 6. `poc/tracking.py` - MLflow Tracking

Logs runs for reproducibility and comparison.

**Logged data:**
- **Tags:** signal_name, snapshot_id, git_sha, author
- **Params:** lag, residualize, tc_model, weight
- **Metrics:** sharpe, ann_ret, max_dd, turnover (per config + baselines)
- **Artifacts:** tearsheet.html, summary.csv, daily.parquet

**Key functions:**
```python
log_run(suite_result, signal_name, catalog, tearsheet_path) -> run_id
get_run_history(experiment_name, max_results=100) -> List[dict]
```

---

### 7. `poc/tearsheet.py` - Tear Sheet Generator

Creates HTML report with traffic-light verdict.

**Sections:**
1. **Header** - Signal name, date range, snapshot
2. **Summary Table** - All configs with key metrics
3. **Cumulative Return Chart** - Interactive Plotly chart
4. **Baseline Comparison** - Correlations table
5. **Verdict Panel** - Green/Yellow/Red with reasons

**Verdict logic:**
| Color | Criteria |
|-------|----------|
| 🟢 Green | Sharpe > 0.5, turnover < 50%, passes lag/resid tests |
| 🟡 Yellow | Sharpe > 0, some concerns (high decay, turnover, correlation) |
| 🔴 Red | Sharpe ≤ 0 or fails critical tests |

**Key functions:**
```python
compute_verdict(suite_result) -> {'color': str, 'reasons': List[str]}
generate_tearsheet(suite_result, signal_name, catalog, output_path) -> path
```

---

### 8. `poc/app.py` - Streamlit UI

Web interface for running and viewing backtests.

**Tabs:**
1. **Run Suite** - Upload signal, configure, run
2. **Results** - View current run results
3. **History** - Browse past MLflow runs

**Sidebar options:**
- Snapshot selection
- Lag configs (0, 1, 2, 3, 5)
- Residualization (off, industry, all)
- Include baselines toggle
- Date range filter
- Universe filter (All / Universe=1 / Universe=0)

**Usage:**
```bash
streamlit run poc/app.py
```

---

## Data Flow

```
1. User uploads signal.parquet
              │
              ▼
2. contract.validate_signal()  ──► Check schema
              │
              ▼
3. catalog.load_catalog()  ──► Load snapshot data
              │
              ▼
4. Filter by date range + universe
              │
              ▼
5. suite.run_suite()
   ├── For each (lag, resid) config:
   │   └── wrapper.run_backtest() ──► backtest_engine.Backtest
   │
   └── If include_baselines:
       └── baselines.generate_all_baselines()
           └── Run backtest for each baseline
              │
              ▼
6. tearsheet.generate_tearsheet()  ──► HTML report
              │
              ▼
7. tracking.log_run()  ──► MLflow
              │
              ▼
8. Display in Streamlit UI
```

---

## Quick Start

```bash
# 1. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create data snapshot
python -c "from poc.catalog import create_snapshot; create_snapshot()"

# 3. Run demo
python run_demo.py

# 4. Launch UI
streamlit run poc/app.py

# 5. View MLflow (optional)
mlflow ui
```

---

## Configuration

### Environment
- Python 3.11+
- Key dependencies: pandas, numpy, mlflow, streamlit, plotly, joblib

### Ignored files (`.gitignore`)
- `mlflow.db`, `mlruns/` - MLflow data
- `snapshots/`, `artifacts/` - Generated files
- `.cache/` - Baseline cache
- `*.parquet`, `*.pkl` - Data files

---

## Performance Optimizations

### BacktestFast Engine

`backtest_engine_fast.py` contains `BacktestFast`, an optimized version of the core `Backtest` class that provides **3-4x speedup** through several techniques:

#### 1. Master Data Pre-Merge

The biggest bottleneck in the original engine is pandas merge operations, which spend 70-80% of time on key hashing (`_factorize_keys`). The master data strategy eliminates most merges:

```python
# Original: 18+ merge operations per backtest
# Optimized: 2-3 merges using pre-indexed master data

# Usage:
from backtest_engine_fast import BacktestFast
from poc.catalog import load_catalog

catalog = load_catalog('snapshots/default', use_master=True)

bt = BacktestFast(
    infile=signal_df,
    retfile=catalog['ret'],
    datefile=catalog['dates'],
    otherfile=catalog['risk'],
    master_data=catalog['master'],  # Pass master for fast joins
    **config
)
result = bt.gen_result()
```

#### 2. Vectorized Industry Residualization

For `resid_style='industry'`, uses `transform('mean')` instead of `groupby().apply()`:

```python
# Original: ~30s (10ms per day × 3000 days)
resid = temp.groupby("date").apply(_residualize_industry, ...)

# Optimized: ~0.05s (single vectorized operation)
ind_means = temp.groupby(["date", "industry_id"]).transform("mean")
temp[sigvar] = temp[sigvar] - ind_means
```

#### 3. Numpy-Based Index Lookups

Uses `get_indexer()` + numpy `take()` instead of merge/join, bypassing pandas overhead:

```python
# Original merge (slow - hash-based key matching + is_unique checks)
df.merge(other, on=['security_id', 'date'])

# Optimized (fast - direct numpy array indexing)
idx = pd.MultiIndex.from_arrays([df['security_id'], df['date']])
positions = master.index.get_indexer(idx)
result[col] = np.take(master[col].values, positions)
```

#### 4. Pre-Indexed Datefile Lookups

Datefile merges use pre-indexed lookups via `reindex()`:

```python
# Original (slow - merge on every call)
df.merge(datefile[['date', 'n']], on='date')

# Optimized (fast - cached index + direct lookup)
datefile_by_date = datefile.set_index('date')  # cached
df['n'] = datefile_by_date.reindex(df['date'])['n'].values
```

### Benchmark Results

| Configuration | Original | Fast+Master | Speedup |
|---------------|----------|-------------|---------|
| Basic (resid=off) | 18-20s | 4.9-5.5s | **3.7x** |
| With residualization | 25-30s | 6-8s | **4x** |

### Profile Breakdown (5.9s total)

| Operation | Time | Notes |
|-----------|------|-------|
| `_fast_join_master` | 2.2s | Numpy get_indexer + take |
| `_factorize_keys` | 2.6s | Remaining 13 merges (datefile, small DFs) |
| groupby/transform | 0.5s | Weight calculations |
| backtest() | 0.6s | Portfolio return aggregation |

### Equivalence Guarantee

`BacktestFast` produces numerically identical results to the original `Backtest` class (differences at machine epsilon ~10⁻¹⁶). Run equivalence tests:

```bash
python tests/test_equivalence.py
```

---

## Extending the System

### Adding a new baseline signal

```python
# In poc/baselines.py
def generate_my_signal(catalog, start_date, end_date) -> pd.DataFrame:
    # Compute signal...
    result = df[['security_id', 'date', 'signal']].copy()
    result = result.rename(columns={'date': 'date_sig'})
    result['date_avail'] = result['date_sig'] + pd.Timedelta(days=1)
    return result.dropna()

# Add to BASELINES registry
BASELINES['my_signal'] = generate_my_signal
```

### Adding a new metric to tearsheet

```python
# In poc/tearsheet.py, update TEARSHEET_TEMPLATE
# Add new column to summary table or new section
```

### Running with custom config

```python
from poc.catalog import load_catalog
from poc.suite import run_suite
from poc.tearsheet import generate_tearsheet

catalog = load_catalog('snapshots/2024-01-01-v1')
result = run_suite(
    signal_df,
    catalog,
    grid={'lags': [0, 1, 5], 'residualize': ['off', 'all']},
    include_baselines=True,
)
generate_tearsheet(result, 'my_signal', catalog)
```
