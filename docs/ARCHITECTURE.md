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
│       ├── master.parquet              # Pre-merged for fast backtest joins
│       ├── factors.parquet             # Pre-indexed for fast factor correlation
│       ├── baseline_reversal_5d.parquet  # Pre-computed reversal baseline
│       └── baseline_value.parquet        # Pre-computed value baseline
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

Manages versioned data snapshots for reproducibility. **All heavy computation is done at snapshot creation time.**

**Key functions:**
```python
# Hashing functions for reproducibility
compute_table_hash(df) -> str                    # 16-char content hash of DataFrame
compute_snapshot_fingerprint(hashes) -> str      # 12-char meta-hash of all components
verify_snapshot_integrity(catalog) -> (bool, details)  # Verify loaded data matches manifest

# Snapshot management
create_snapshot(source_dir, output_dir, use_content_hash=False) -> Path
load_catalog(snapshot_path, use_master=True, verify_integrity=False) -> dict
list_snapshots(snapshots_dir) -> List[str]
```

**Catalog dict structure:**
```python
{
    'ret': pd.DataFrame,           # Normalized returns (deduplicated, int32 keys)
    'risk': pd.DataFrame,          # Normalized risk factors (category cols)
    'dates': pd.DataFrame,         # Trading calendar
    'master': pd.DataFrame,        # Pre-merged ret+risk (MultiIndex)
    'factors': pd.DataFrame,       # Pre-indexed factors (MultiIndex)
    'dates_indexed': dict,         # {'by_date': df, 'by_n': df}
    'asof_tables': dict,           # {'resid': df, 'byvars_cap': df}
    'snapshot_id': str,
    'fingerprint': str,            # 12-char content fingerprint (V3+)
    'manifest': dict,              # Includes version and fingerprint details
}
```

**V5 Snapshot Format (Content-Addressable with Lineage):**

V5 snapshots provide content hashing and optional source lineage:

| Feature | Description |
|---------|-------------|
| **Fingerprint** | 12-char meta-hash of all component hashes |
| **Source lineage** | Optional: where each table originated (path, URI) |
| **Per-table hashes** | 16-char SHA256 hash per table |
| **Component deduplication** | `deduplicate=True` stores tables once in shared object store |

**V5 Manifest:**
```json
{
  "snapshot_id": "full-history",
  "version": 5,
  "fingerprint": "e570278e9355",
  "components": {
    "ret": {
      "source": "data/ret.parquet",
      "hash": "7ed19eeda288aa6e",
      "rows": 30445515,
      "securities": 12482,
      "date_range": ["2001-01-02", "2025-10-07"]
    },
    "risk": {"source": "data/risk.parquet", "hash": "345677db35fb17bc", "rows": 30628078},
    "dates": {"source": "data/trading_date.pkl", "hash": "a65d479e1bf659f5", "rows": 15603},
    "master": {"derived_from": ["ret", "risk"], "hash": "dfa58371b1267141", "rows": 30434746},
    "factors": {"derived_from": ["risk"], "rows": 30628078},
    "asof_resid": {"derived_from": ["risk"], "rows": 30628078}
  }
}
```

**Creating Snapshots with Source Lineage:**
```python
# Basic (local paths)
create_snapshot('data/', 'snapshots/')

# With source tracking
sources = {
    'ret': {'uri': 's3://exports/returns/2026-02-08/', 'extraction_id': 'ret_001'},
    'risk': {'uri': 's3://exports/risk/2026-02-08/'},
}
create_snapshot('data/', 'snapshots/', sources=sources)

# With component deduplication (saves disk space)
create_snapshot('data/', 'snapshots/', deduplicate=True)
```

**Reproducibility Benefits:**

| Feature | Benefit |
|---------|---------|
| **Fingerprint** | Single hash to verify entire snapshot |
| **Source tracking** | Trace data back to original exports |
| **Same data → Same hash** | Detect when data changes |
| **Component deduplication** | Reuse unchanged tables across snapshots |

**V2 Snapshot Format (Legacy):**

V2 snapshots (version >= 2) have all precomputation done at creation time:

| File | Created At | Contents |
|------|------------|----------|
| `ret.parquet` | Snapshot creation | Normalized, deduplicated, security_id=int32 |
| `risk.parquet` | Snapshot creation | Normalized, group cols=category |
| `master.parquet` | Snapshot creation | Pre-merged ret+risk, ready for MultiIndex |
| `asof_resid.parquet` | Snapshot creation | Pre-sorted for residualization merge_asof |
| `asof_cap.parquet` | Snapshot creation | Pre-sorted for cap lookup merge_asof |
| `factors.parquet` | Snapshot creation | Pre-indexed for factor correlation |

**Loading Flow:**
```python
# V3 snapshots: Read files, restore indexes, optionally verify integrity
load_catalog('snapshots/snap_a1b2c3d4', verify_integrity=True)

# V2 snapshots: Just read files, restore indexes
load_catalog('snapshots/v2-snapshot')  # ~2-5 seconds

# V1 snapshots (backwards compat): Normalize on load
load_catalog('snapshots/v1-snapshot')  # Slower, normalizes each time
```

**Content-Addressable Snapshot Creation:**
```python
# Date-based ID (default)
create_snapshot('data/', 'snapshots/')  # Creates snapshots/2026-02-10-v1/

# Content-based ID (for deduplication)
create_snapshot('data/', 'snapshots/', use_content_hash=True)  # Creates snapshots/snap_a1b2c3d4e5f6/
# If identical data already exists, returns existing snapshot path
```

**V4 Format: Component Reuse Across Snapshots:**

V4 snapshots (`deduplicate=True`) store each table once in a shared object store, identified by content hash:

```python
# Create snapshot with component deduplication
create_snapshot('data/', 'snapshots/', deduplicate=True)

# Directory structure:
# snapshots/
#   objects/          # Shared content-addressable storage
#     44/
#       4479db9c06a95a78_ret.parquet   # Unique by content hash
#     a1/
#       a1b2c3d4e5f6_master.parquet
#   snap1/
#     manifest.json   # References objects by hash
#     ret.parquet     # Symlink → ../objects/44/4479db...parquet
#   snap2/
#     manifest.json   # May reference SAME objects if data unchanged
#     ret.parquet     # Symlink → ../objects/44/4479db...parquet (REUSED)
```

**Benefits:**
- **Disk savings**: Identical tables stored once, not duplicated per snapshot
- **Automatic deduplication**: If `ret` data hasn't changed, new snapshot reuses existing object
- **Fast snapshots**: Creating a snapshot with unchanged data is near-instant

**Object Store Functions:**
```python
# Check object store stats
get_object_stats('snapshots/')
# Returns: {'unique_objects': 12, 'total_size_mb': 1500.5, 'by_type': {'ret': 3, 'risk': 3, ...}}

# Manifest includes object references
manifest = {
    'version': 4,
    'objects': {
        'ret': {'hash': '4479db9c06a95a78', 'object_path': 'objects/44/4479db...parquet'},
        'master': {'hash': 'a1b2c3d4e5f6', 'object_path': 'objects/a1/a1b2c3...parquet'},
    }
}
```

**Master Data:**

Pre-merged, indexed version of `ret` and `risk` data:
```python
# Indexed on (security_id, date) for O(1) lookups via get_indexer
# Columns from ret: ret, resret, openret, resopenret, vol, adv, close_adj
# Columns from risk: mcap, cap, industry_id, sector_id, size, value, growth, 
#                    leverage, volatility, momentum, yield
```

**Pre-sorted Asof Tables:**

For `merge_asof` operations (which require sorted keys):
```python
asof_tables = {
    'resid': risk_df with date→date_sig, sorted by date_sig,
    'byvars_cap': cap table with date→date_sig, sorted by date_sig,
}
```

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

**Annualization note:** All slices (overall, year, cap) use 252-day annualization for Sharpe ratio and annual return calculations. This ensures metrics are comparable across different time periods and market cap groups.

---

### 4. `poc/baselines.py` - Baseline Library

Standard signals for comparison.

**Available baselines:**
| Signal | Description | Lookback |
|--------|-------------|----------|
| `reversal_5d` | -1 × past 5-day return | 5 days |
| `momentum_12_1` | 12-month return minus last month | 252 days |
| `value` | Value factor from risk file | N/A |

**Pre-computed Baselines:**

Baselines can be pre-computed and stored in the snapshot folder for instant loading:

```python
from poc.catalog import load_catalog
from poc.baselines import precompute_baselines_to_snapshot

catalog = load_catalog('snapshots/my-snapshot', use_master=False)
precompute_baselines_to_snapshot(catalog, 'snapshots/my-snapshot')
# Creates: baseline_reversal_5d.parquet, baseline_value.parquet
```

When `generate_all_baselines()` is called with a `snapshot_path`, it loads from these files and filters to the requested date range (instant vs. ~60s to compute).

**Fallback caching:** If pre-computed files don't exist, results are cached to `.cache/baselines/` using joblib.

**Key functions:**
```python
generate_reversal_signal(catalog, lookback=5, start_date, end_date) -> df
generate_value_signal(catalog, start_date, end_date) -> df
generate_all_baselines(catalog, start_date, end_date, snapshot_path=None) -> Dict[str, df]
precompute_baselines_to_snapshot(catalog, snapshot_path) -> Dict[str, Path]
load_precomputed_baselines(snapshot_path) -> Optional[Dict[str, df]]
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
    factor_exposures: pd.DataFrame         # Signal correlation to risk factors
    coverage: Dict                         # avg_securities_per_day, coverage_pct, etc.

    @property best_sharpe -> float
    @property best_config -> str
```

**Key functions:**
```python
run_suite(signal_df, catalog, grid=None, include_baselines=True, n_jobs=1) -> SuiteResult
get_best_config(suite_result, metric='sharpe') -> str
```

**Risk Factor Exposures:**
Correlates signal values to common risk factors (size, value, growth, leverage, volatility, momentum) to identify factor overlap.

**Coverage Metrics:**
- `avg_securities_per_day`: Average number of securities with signal per day
- `coverage_pct`: Percentage of universe securities covered by signal
- `unique_securities`: Total unique securities in signal
- `total_days`: Number of trading days with signal

---

### 6. `poc/tracking.py` - MLflow Tracking

Logs runs for reproducibility and comparison.

**Logged data:**
- **Tags:** signal_name, snapshot_id, data_fingerprint, git_sha, author
- **Params:** lag, residualize, tc_model, weight
- **Metrics:** sharpe, ann_ret, max_dd, turnover (per config + baselines)
- **Artifacts:** tearsheet.html, summary.csv, daily.parquet

**Reproducibility via Fingerprint:**
```python
# When logging a run, include the data fingerprint
mlflow.set_tag('data_fingerprint', catalog.get('fingerprint', 'unknown'))

# Later, verify reproducibility
logged_fingerprint = run.data.tags.get('data_fingerprint')
current_fingerprint = catalog.get('fingerprint')
assert logged_fingerprint == current_fingerprint, "Data has changed!"
```

**Key functions:**
```python
log_run(suite_result, signal_name, catalog, tearsheet_path) -> run_id
get_run_history(experiment_name, max_results=100) -> List[dict]
```

---

### 7. `poc/tearsheet.py` - Tear Sheet Generator

Creates HTML report with traffic-light verdict and composite quality score.

**Sections:**
1. **Header** - Signal name, date range, snapshot
2. **Verdict Panel** - Green/Yellow/Red with reasons + letter grade (A-F)
3. **Quality Score Breakdown** - Weighted component scores
4. **Headline Metrics** - Sharpe, return, drawdown, turnover, coverage
5. **Suite Results** - All configs with key metrics
6. **Robustness Analysis** - Cap-tier and year-by-year breakdown (all metrics use 252-day annualization for consistency)
7. **Baseline Comparison** - Correlations table
8. **Signal Uniqueness** - Baseline correlations
9. **Risk Factor Exposures** - Correlation to size, value, momentum, volatility, etc.
10. **Signal Coverage** - Avg securities/day, unique securities, universe coverage %

**Verdict logic:**
| Color | Criteria |
|-------|----------|
| 🟢 Green | Sharpe ≥ 1.0, passes lag/resid tests, low baseline/factor correlation, consistent across cap/year |
| 🟡 Yellow | Sharpe ≥ 0.5, some concerns (high decay, turnover, correlation, inconsistency) |
| 🔴 Red | Sharpe < 0.5 or fails critical tests |

**Composite Quality Score (0-100):**

| Component | Weight | Formula | Scale |
|-----------|--------|---------|-------|
| Sharpe | 25% | `min(100, sharpe / 2.0 * 100)` | Sharpe 2.0 = 100 |
| Lag Stability | 15% | `(sharpe_lag2 / sharpe_lag0) * 100` | 100% = no decay |
| Resid Stability | 15% | `(sharpe_resid_ind / sharpe_resid_off) * 100` | 100% = survives |
| Baseline Uniqueness | 15% | `100 - (max_baseline_corr / 0.5 * 100)` | 0% corr = 100 |
| Cap Consistency | 10% | `(large_cap_sharpe / small_cap_sharpe) * 100` | 100% = balanced |
| Year Consistency | 10% | `(positive_years / total_years) * 100` | 100% = all positive |
| Factor Uniqueness | 10% | `100 - (max_factor_corr / 0.3 * 100)` | 0% corr = 100 |

**Grade thresholds:** A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, F < 35

**Key functions:**
```python
compute_verdict(suite_result) -> {'color': str, 'reasons': List[str]}
compute_composite_score(suite_result) -> {'total_score': float, 'breakdown': Dict, 'grade': str}
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

# 2. Create data snapshot (date-based ID)
python -c "from poc.catalog import create_snapshot; create_snapshot()"

# Or with content-based ID (for reproducibility/deduplication)
python -c "from poc.catalog import create_snapshot; create_snapshot(use_content_hash=True)"

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

The system uses a **multi-layer optimization strategy** where heavy computation is pushed as early as possible:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    create_snapshot() - ONE TIME (~60s)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Normalize data (dedup on security_id+date, convert to int32)     │    │
│  │  • Create master.parquet (pre-merged ret+risk, indexed)             │    │
│  │  • Create asof_resid.parquet, asof_cap.parquet (pre-sorted)         │    │
│  │  • Create factors.parquet (pre-indexed)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    load_catalog() - FAST (~2-5s)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Read pre-normalized parquet files (no dedup/conversion)          │    │
│  │  • Restore MultiIndex on master_data (cheap)                        │    │
│  │  • Create dates_indexed dict (tiny data)                            │    │
│  │  • Load pre-sorted asof tables                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    run_backtest() - FAST (seconds)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • _fast_join_master() uses get_indexer + np.take (no merge)        │    │
│  │  • merge_asof uses pre-sorted asof tables (no sort)                 │    │
│  │  • No is_unique checks, no _factorize_keys overhead                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Snapshot Precomputation

All heavy data preparation is done **once** at snapshot creation time:

```bash
# Create optimized snapshot (takes ~60s for large data, but only done once)
python -c "from poc.catalog import create_snapshot; create_snapshot('data', 'snapshots', 'my-snapshot')"
```

**Files created in snapshot:**

| File | Contents | Purpose |
|------|----------|---------|
| `ret.parquet` | Normalized returns | Deduplicated, security_id=int32 |
| `risk.parquet` | Normalized risk factors | Deduplicated, group cols=category |
| `trading_date.parquet` | Normalized dates | n=int32, date=datetime64 |
| `master.parquet` | Pre-merged ret+risk | Indexed by (security_id, date) |
| `factors.parquet` | Pre-indexed factors | For fast factor correlation |
| `asof_resid.parquet` | Pre-sorted for residualization | date→date_sig, sorted |
| `asof_cap.parquet` | Pre-sorted for cap lookup | date→date_sig, sorted |

**V2 manifest marker:**
```json
{
  "version": 2,
  "files": {
    "ret": {"normalized": true, ...},
    "risk": {"normalized": true, ...}
  }
}
```

### Data Normalization

Normalization eliminates expensive operations during backtest:

| Optimization | Before | After | Benefit |
|--------------|--------|-------|---------|
| Deduplicate on keys | Every merge checks uniqueness | Done once at snapshot | Eliminates `is_unique` checks |
| `security_id` → `int32` | `int64` (8 bytes) | `int32` (4 bytes) | 2x faster hashing in `_factorize_keys` |
| Group cols → `category` | String/int comparisons | Integer codes | Faster groupby/factorize |
| Dates → `datetime64[ns]` | Mixed types | Consistent | No type coercion |

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

#### 2. Vectorized NumPy Residualization

Replaces `groupby().apply()` with pure NumPy loop over sorted date chunks:

```python
# Original: ~8-10s (creates thousands of Pandas objects)
resid = temp.groupby("date").apply(_residualize_factors_within_industry, ...)

# Optimized: ~1-2s (pure NumPy with bincount/solve)
dates = temp["date"].values
y = temp[sigvar].to_numpy()
X = temp[factor_cols].to_numpy()
residuals = _vectorized_resid_all_numpy(dates, y, X, industry_codes)
```

The NumPy implementation:
- Uses `np.unique(dates, return_index=True)` to find day boundaries
- Uses `np.bincount` for fast industry group aggregations
- Uses `np.linalg.solve` for regression within each day

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

### Benchmark Results (3-Year Data, 377K signal rows)

| Configuration | Original | Optimized | Speedup |
|---------------|----------|-----------|---------|
| resid=off, overall | 19.3s | 4.9s | **3.9x** |
| resid=off, overall+cap | 29.4s | 4.7s | **6.2x** |
| resid=all, overall | 27.2s | 10.0s | **2.7x** |
| resid=all, overall+cap | 42.2s | 9.9s | **4.3x** |

**Key observations:**
- `overall+cap` cases show biggest gains because cached sorted asof tables avoid repeated 30M-row sorts
- `merge_asof` (fuzzy time-based join) cannot use pre-merged master data, which limits speedup for resid cases

### Optimization Techniques

1. **Master Data Pre-Merge**: Pre-indexed `(security_id, date)` lookups using `get_indexer` + `numpy.take`
2. **Cached Sorted Asof Tables**: Avoid repeated 30M-row sorts for `merge_asof` calls
3. **Fast Multi-Key Lookup**: Replace turnover prev merge with numpy indexer+take
4. **Vectorized Residualization**: Pure NumPy instead of statsmodels (97x faster for the math)

**Residualization math comparison:**
- Statsmodels OLS (groupby.apply): 4.85s
- Pure NumPy: 0.05s (~97x faster)

### Streamlit Data Caching

The app uses `@st.cache_resource` to keep catalog data in memory across reruns:

```python
@st.cache_resource(show_spinner="Loading data snapshot...")
def get_cached_catalog(snapshot_path: str):
    """Load catalog once and cache in memory across reruns."""
    return load_catalog(snapshot_path, use_master=True)
```

**How it works:**
- First run: Loads full catalog from disk (a few seconds for large snapshots)
- Subsequent runs: Uses cached data instantly
- Changing snapshot: Loads that snapshot once, then caches it
- Cache persists until "Clear Cache" button is clicked or app restarts

**Memory usage:**
| Snapshot | Rows | Approx RAM |
|----------|------|------------|
| 2018 only | ~600K | ~500 MB |
| 2018-2023 | ~4M | ~1.5 GB |
| Full history (2001-2025) | ~30M | ~3 GB |

**Controls:**
- Sidebar → "Data Cache" expander → "Clear Data Cache" button to force reload from disk

### Compare Tab Caching

The Compare tab loads MLflow run data and artifacts. Several caching layers ensure fast performance:

**1. Run History Caching:**
```python
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_run_history():
    """Get run history with 60-second cache to avoid repeated MLflow queries."""
    return get_run_history()
```
- Caches the list of past runs for 60 seconds
- Avoids repeated MLflow queries on tab switches

**2. MLflow Artifact Caching:**
```python
# In compare.py
_run_data_cache: Dict[str, Tuple[Dict, Optional[pd.DataFrame]]] = {}

def load_run_data(run_id: str, use_cache: bool = True):
    if use_cache and run_id in _run_data_cache:
        return _run_data_cache[run_id]
    # ... load from MLflow ...
    _run_data_cache[run_id] = result
    return result
```
- Caches downloaded artifacts (daily parquet files) in memory
- First comparison downloads from MLflow; repeat comparisons are instant

**3. Parallel Run Loading:**
```python
def compare_runs(run_a_id: str, run_b_id: str) -> CompareResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(load_run_data, run_a_id)
        future_b = executor.submit(load_run_data, run_b_id)
        run_a, daily_a = future_a.result()
        run_b, daily_b = future_b.result()
```
- Loads both runs simultaneously, cutting uncached load time ~50%

**Controls:**
- Sidebar → "Data Cache" expander → "Clear Compare Cache" to clear MLflow artifact cache and run history
- History tab → "Refresh" button clears run history cache

The **bulk of the overall speedup comes from eliminating merge/sort operations** (`_factorize_keys`, sorting), not from faster residualization math.

### Post-Backtest Optimizations

After backtests complete, `run_suite()` computes additional analytics. These have been optimized:

#### 1. Baseline Signal Reuse

**Problem:** `_compute_correlations()` was calling `generate_all_baselines()` again, duplicating work.

**Fix:** Pass already-generated `baseline_signals` dict to avoid regeneration.

```python
# Before (slow - generates baselines twice)
baseline_signals = generate_all_baselines(...)  # First time
baselines = {name: run_backtest(df, ...) for name, df in baseline_signals.items()}
correlations = _compute_correlations(...)  # Called generate_all_baselines again!

# After (fast - reuse baseline_signals)
baseline_signals = generate_all_baselines(...)  # Only once
baselines = {...}
correlations = _compute_correlations(..., baseline_signals=baseline_signals)
```

#### 2. Fast IC Computation

**Problem:** `_compute_ic()` used slow `merge()` and `groupby().apply()` with scipy.

**Fix:** Use pre-indexed `master_data` with `get_indexer` + numpy, vectorized Spearman correlation.

```python
# Before (slow - merge + groupby.apply + scipy)
merged = signal.merge(ret, on=['security_id', 'date_sig'])
ic_by_date = merged.groupby('date_sig').apply(lambda g: spearmanr(g['signal'], g['ret'])[0])

# After (fast - get_indexer + numpy)
positions = master_data.index.get_indexer(lookup_idx)
merged['fwd_ret'] = np.take(master_data['ret'].values, positions[valid_mask])

# Vectorized Spearman (rank correlation via numpy)
sig_ranks = sig_day.argsort().argsort()
ret_ranks = ret_day.argsort().argsort()
corr = cov(sig_ranks, ret_ranks) / (std(sig_ranks) * std(ret_ranks))
```

#### 3. Fast Factor Exposures

Uses pre-indexed `factors` DataFrame from catalog:

```python
# Fast path with pre-indexed factors
positions = factors_df.index.get_indexer(lookup_idx)
factor_values = factors_df[factor].values[positions[valid_mask]]
corr = np.corrcoef(signal_values, factor_values)[0, 1]
```

### Profile Breakdown (resid=off, 6.2s total)

| Operation | Time | Notes |
|-----------|------|-------|
| `_factorize_keys` | 2.7s | Remaining merges (datefile, small DFs) |
| `is_unique` | 2.2s | Pandas internal uniqueness checks |
| `_take_nd_ndarray` | 0.5s | Array copying |
| groupby/transform | 0.5s | Weight calculations |

### Equivalence Guarantee

`BacktestFast` produces numerically identical results to the original `Backtest` class (differences at machine epsilon ~10⁻¹⁶). Run equivalence tests:

```bash
python tests/test_equivalence.py
```

### Profiling Tool

Use `profile_backtest.py` to identify bottlenecks:

```bash
# Basic profiling
python profile_backtest.py --top 40

# With residualization (slower path)
python profile_backtest.py --top 40 --resid

# Sort by cumulative time
python profile_backtest.py --top 40 --cumulative
```

**Key metrics to watch:**

| Function | Description | Target |
|----------|-------------|--------|
| `is_unique` | Pandas uniqueness checks during merge | < 1s (should be near 0 with precomputed) |
| `_factorize_keys` | Hash-based key creation for merges | < 1s (minimized by using master_data) |
| `get_join_indexers_non_unique` | Non-unique join path | Should not appear (keys should be unique) |

**Profiler output example:**
```
Top 10 functions by total time:

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       36    0.52    0.014    0.55    0.015 merge.py:2399(_factorize_keys)
      145    0.08    0.001    0.08    0.001 base.py:2236(is_monotonic_increasing)
      ...
```

### Creating Optimized Snapshots

To get full benefit of optimizations, recreate snapshots with v2 format:

```bash
# Delete old snapshot
rm -rf snapshots/full-history

# Create new v2 snapshot with all precomputation
python -c "from poc.catalog import create_snapshot; create_snapshot('data', 'snapshots', 'full-history')"
```

This runs the heavy normalization and pre-indexing once, making all subsequent `load_catalog()` calls fast.

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
