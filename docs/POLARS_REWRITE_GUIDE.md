# Polars Reimplementation Guide

Lessons learned from rewriting the backtest engine from Pandas to Polars, achieving **~6.4x speedup** (21.3s → 3.3s on 10 years of data) while maintaining exact numerical equivalence.

## What Worked

### 1. Convert Once at Boundaries (Kill Pandas↔Polars Ping-Pong)

**Implementation:** `BacktestFastV2._to_polars()` and `BacktestFastMinimal.__init__`

```python
# ✅ GOOD: Convert once at init
self.master_df = master if isinstance(master, pl.DataFrame) else master.collect()
self.date_df = datefile if isinstance(datefile, pl.DataFrame) else datefile.collect()

# ✅ GOOD: Convert back only at final output
def gen_result(self):
    out = self._engine.run(self._signal_pl)
    result = out["summary"].to_pandas()  # Only conversion point
```

**Why it worked:** Eliminated per-`byvar` conversions. Conversion overhead dropped to ~0.25% of runtime.

**What didn't work:** Converting inside loops or per-operation.

---

### 2. Cache Polars Versions of Big Tables

**Implementation:** `set_precomputed_indexes()` and cached attributes

```python
# ✅ GOOD: Cache commonly used subsets
self._master_schema = set(self.master_df.columns)
self._date_n = self.date_df.select(["date", "n"])
self._cap_df = self.master_df.select(["security_id", "date", "cap"])

# ✅ GOOD: Pre-sort for join_asof
if self.asof_df is not None:
    self._asof_sorted = self.asof_df.rename({"date": "date_sig"}).sort("date_sig")
```

**Why it worked:** Avoids repeated sorting/schema resolution. Pre-sorted data makes `join_asof` ~10x faster.

---

### 3. Use a Pre-Joined Master Table

**Implementation:** `BacktestFastV2._init_polars_data()`

```python
# ✅ GOOD: Build master once, join signals to it
if self.retfile is self.otherfile:
    self._master_pl = self._to_polars(self.retfile)  # Skip redundant join
else:
    ret_pl = self._to_polars(self.retfile)
    other_pl = self._to_polars(self.otherfile)
    self._master_pl = ret_pl.join(other_pl.select(other_cols), on=["security_id", "date"], how="left")
```

**Why it worked:** Reduced expensive joins from ~5 per byvar to ~2 total.

---

### 4. Universe-Only Filtering (Biggest Win: ~1.8x)

**Implementation:** `poc/catalog.py` (external to engine)

```python
# ✅ GOOD: Use semi-join for fast universe filtering
if universe_only and "universe_flag" in master.columns:
    universe_ids = master.filter(pl.col("universe_flag") == 1).select("security_id").unique()
    master = master.join(universe_ids, on="security_id", how="semi")
```

**Why it worked:** Master rows dropped ~44% (30M → 17M). This was the single biggest speedup.

**What didn't work:** Python set membership checks (slow).

---

### 5. Cache Schema Once (Avoid Repeated Introspection)

**Implementation:** `BacktestFastMinimal.__init__`

```python
# ✅ GOOD: Cache schema once
self._master_schema = set(self.master_df.columns)

# ✅ GOOD: Use cached schema for fast checks
def _join_master(self, lf, cols, how="inner"):
    available = [c for c in need if c in self._master_schema]  # O(1) lookup
```

**Why it worked:** Polars pays non-trivial overhead resolving schemas on LazyFrames repeatedly.

**What didn't work:** Calling `.collect_schema()` or `.columns` inside hot loops.

---

### 6. Strategic Eager Materialization

**Implementation:** `BacktestFastMinimal.run()`

```python
# ✅ GOOD: Collect at sensible boundaries
base_df = base.collect()  # Collect after preprocess

for byvar in cfg.byvars:
    lf = base_df.lazy()
    lf = self._add_positions(lf, byvar)
    lf = self._add_weights(lf, byvar)
    weighted_df = lf.collect()  # Collect once per byvar
    
    # Downstream ops use eager DataFrames
    turnover_daily, tc_daily, counts_daily = self._turnover_and_tc(weighted_df.lazy(), byvar)
```

**Why it worked:** Very long lazy plans add overhead. Collecting at the right place reduced DAG complexity.

**What didn't work:** 
- Keeping everything lazy until the very end (huge DAG overhead)
- Collecting too early (lost streaming benefits)

---

### 7. NumPy for Tight Loops (Residualization)

**Implementation:** `_resid_by_date_industry()` and `_resid_by_date_industry_factors()`

```python
# ✅ GOOD: NumPy for per-date iteration with bincount
for s, e in zip(starts, ends):
    yd = y[s:e]
    ind_d = ind[s:e]
    _, inv = np.unique(ind_d, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=yd)
    means = sums / cnt
    out[s:e] = yd - means[inv]
```

**Why it worked:** NumPy `bincount` is highly optimized. Per-date loops with small arrays are faster than Polars group_by with nested aggregations.

**Future optimization:** Add `@numba.njit` decorator for 3-10x additional speedup (implemented with fallback).

---

### 8. Turnover: Match Semantics First, Then Optimize

**Implementation:** `BacktestFastMinimal._turnover_and_tc()`

```python
# ✅ GOOD: Explicit entry/exit handling with left join + anti-join
cur = w_df.select(["security_id", byvar, "n", "weight"])
prev = cur.with_columns((pl.col("n") + 1).alias("n")).rename({"weight": "weight_prev"})

# Left join for current positions
diff = cur.join(prev, on=["security_id", byvar, "n"], how="left")

# Anti-join for exits (positions in prev not in cur)
exits = prev.join(cur.select(["security_id", byvar, "n"]).unique(), 
                  on=["security_id", byvar, "n"], how="anti")

diff = pl.concat([diff, exits], how="diagonal_relaxed")

# Zero out first day + divide by 4 (convention)
diff = diff.with_columns([
    pl.when(pl.col("n") == n_min).then(0.0).otherwise(pl.col("weight_diff")).alias("weight_diff")
])
turnover_daily = diff.group_by([byvar, "date"]).agg((pl.sum("weight_diff") / 4.0).alias("turnover"))
```

**Why it worked:** Exact equivalence with the original engine was critical for trust.

---

### 9. Residualization: Match Factor Join Date (Critical Bug Fix)

**Problem:** Residualization was producing different results despite using the "same" algorithm.

**Root cause (debugging journey):**

1. **Initial hypothesis:** Different OLS formula (two-step vs single regression)
   - Fixed by implementing single OLS with intercept + factors + industry dummies
   - Still failed!

2. **Second hypothesis:** Global vs per-date industry factorization
   - Code was pre-factorizing industry_id globally before passing to residualization
   - Fixed by passing original industry_id values
   - Still failed!

3. **Actual bug (found via row-level debugging):**
   - OLD engine: `pd.merge_asof(on="date_sig", tolerance="5d")` — factors from **signal date**
   - NEW engine: `join(on="date")` — factors from **trade date** (date_ret)
   - These are different dates! Signal date can be days before trade date.

**Debug commands that revealed the issue:**

```python
# Check what date OLD engine uses for factor join
# OLD: pd.merge_asof(on="date_sig", tolerance="5d")  ← SIGNAL DATE
# NEW: join(on="date")  ← TRADE DATE (wrong!)

# For signal row: security_id=75, date_sig=2020-01-03, date_ret=2020-01-06
# OLD gets factors from master @ 2020-01-03 (or earlier within 5d)
# NEW was getting factors from master @ 2020-01-06 (WRONG!)
```

**Fix:**

```python
# ✅ GOOD: Match OLD engine's merge_asof on date_sig
factor_df = (
    self.master_df
    .select(["security_id", "date", "industry_id", *cfg.resid_vars])
    .rename({"date": "date_sig"})
    .sort("date_sig")
)

lf2 = (
    lf.sort("date_sig")
    .join_asof(
        factor_df.lazy(),
        by="security_id",
        on="date_sig",
        strategy="backward",
        tolerance="5d",  # Match OLD engine exactly
    )
    .drop_nulls()
)
```

**Result:** Exact equivalence achieved. Speedup: **12.5x** (1.9s → 0.15s)

**Lesson:** When debugging equivalence failures, trace row-level data through both pipelines. The bug was not in the algorithm but in **which data** was being fed to it.

---

### 10. Conditional Filters: Match Exact Branching Logic

**Problem:** No-resid equivalence failed with real data (sharpe -0.09 vs -0.27) despite synthetic data passing.

**Root cause:**

```python
# OLD engine (backtest_engine.py)
if self.from_open:
    temp = temp[temp["ret"] > -0.95]  # Only filters when from_open=True
else:
    # No filter applied!
    pass

# NEW engine (backtest_engine_minimal_fast.py) - WRONG
lf = lf.filter(pl.col("ret") > -0.95)  # Always filters
```

**Debug approach:**
1. Compare preprocessed row counts: OLD had 4 more rows than NEW
2. Found the extra rows all had `ret = -0.95` exactly
3. Traced filter logic in both engines

**Fix:**

```python
# ✅ GOOD: Match OLD engine's conditional filter
if cfg.from_open:
    lf = lf.rename({"openret": "ret", "resopenret": "resret"})
    lf = lf.filter(pl.col("ret") > -0.95)  # Only when from_open
```

**Result:** Real data now matches exactly (0.00% diff on all metrics).

**Lesson:** When porting conditional logic, test with real data that exercises edge cases. Synthetic data often doesn't have extreme values like -95% returns.

---

## What Didn't Work

### ❌ Shift-Based Turnover (Missed Exits)

```python
# ❌ BAD: Shift misses securities that exit the portfolio
lf = lf.with_columns([
    pl.col("weight").shift(1).over(["security_id", byvar]).alias("weight_prev")
])
turnover = (pl.col("weight") - pl.col("weight_prev")).abs().sum()
```

**Why it failed:** When a security exits the portfolio, there's no row to shift from. The explicit join + anti-join approach correctly captures exits.

---

### ❌ Quantile Thresholds Instead of Rank (Numerical Differences)

```python
# ❌ BAD: Quantile thresholds produce different results than rank-based percentiles
q_low = pl.col(sigvar).quantile(a/100).over(["date", byvar])
q_high = pl.col(sigvar).quantile(b/100).over(["date", byvar])
position = pl.when(pl.col(sigvar) <= q_low).then(-1).when(pl.col(sigvar) >= q_high).then(1).otherwise(0)
```

**Why it failed:** Rank-based percentiles handle ties differently than quantile thresholds. Had to revert to rank-based approach:

```python
# ✅ GOOD: Rank-based matches original exactly
pct = (pl.col(sigvar).rank("average").over(["date", byvar]) * 100.0 / pl.len().over(["date", byvar])).ceil()
position = pl.when(pl.col("pct") <= a).then(-1).when(pl.col("pct") > b).then(1).otherwise(0)
```

---

### ❌ Aggressive Lazy Chains Without Intermediate Collects

```python
# ❌ BAD: Too much lazy → huge DAG overhead
lf = self.preprocess(signal)
for byvar in cfg.byvars:
    lf = self._add_positions(lf, byvar)  # Still lazy
    lf = self._add_weights(lf, byvar)    # Still lazy
    lf = self._turnover_and_tc(lf, byvar)  # Still lazy
    # ... many more lazy ops
    result = lf.collect()  # Massive DAG, slow to optimize
```

**Why it failed:** Polars' query optimizer has overhead proportional to DAG size. Very long plans can be slower than strategic intermediate collects.

---

### ❌ collect_schema() in Hot Paths

```python
# ❌ BAD: Schema resolution inside loops
def _turnover_and_tc(self, weights, byvar):
    schema = weights.collect_schema()  # Expensive!
    has_cap = "cap" in schema
```

**Why it failed:** `collect_schema()` on LazyFrames forces partial query planning. Cache schema once at init.

---

## Speedup Attribution: Where the Gains Come From

Benchmark: 10 years of real data (12M rows), single signal backtest.

### Overall Results

| Engine | Time | vs OLD |
|--------|------|--------|
| OLD (Pandas, backtest_engine.py) | 21.3s | baseline |
| NEW (Polars, no precompute) | 3.3s | **6.4x faster** |
| NEW (Polars, precomputed) | 3.3s | **6.5x faster** |

### Speedup Breakdown by Source

| Source | Speedup | Time Saved | % of Total Gain |
|--------|---------|------------|-----------------|
| **Polars rewrite** | 6.4x | 18.0s | **~95%** |
| Master table (single join vs multiple) | 2.5x on joins | 1.2s | ~6% |
| Precomputation (engine reuse) | 1.01x | 0.03s | ~0.2% |

**Key insight:** The Polars rewrite is the dominant factor (~95% of speedup). Master table and precomputation provide incremental benefits.

### Master Table: Scales With Data Size

The master table benefit increases with data size because join overhead compounds:

| Data Size | Multiple Joins | Single Join | Speedup | % of Runtime Saved |
|-----------|----------------|-------------|---------|-------------------|
| 1 year (1.2M rows) | 0.14s | 0.06s | 2.3x | ~12% |
| 5 years (6M rows) | 0.99s | 0.40s | 2.5x | ~37% |
| 10 years (12M rows) | 1.98s | 0.79s | 2.5x | ~36% |

For long historical backtests, the master table saves ~1-2s per run.

### Precomputation: When It Matters

Precomputation (reusing the engine instance across signals) provides modest per-signal savings:

| Scenario | No Precompute | Precomputed | Savings |
|----------|---------------|-------------|---------|
| 1 signal | 3.34s | 3.30s | 0.04s |
| 5 signals | 17.4s | 16.9s | 0.5s |
| 10 signals | ~35s | ~33s | ~2s |

**When precomputation helps most:**
- API/server mode: engine initialized once at startup
- Batch research: testing many signals against same universe
- Interactive apps: engine persists across UI interactions

**When it doesn't matter:**
- Single ad-hoc backtests (most time is in computation, not setup)

### Data Persistence: Avoid Reloading

The biggest persistence benefit comes from **loading data once** with date filtering and using `run_fast()`:

| Method | Time | Notes |
|--------|------|-------|
| Cold start (load + first run) | 33s | One-time cost |
| Warm run with `run_fast()` | **1.8s** | Direct engine, minimal overhead |
| Speedup | **18x** | After first run |

*Benchmark: Real signal (reversal_signal_analyst.csv), 6.4M rows, 2012-2021*

```python
from api.service import BacktestService

# Load data once with date filtering (~33s cold start)
service = BacktestService.get(
    start_date='2012-01-01',
    end_date='2021-12-31',
)

# Fast path: use run_fast() with pre-aligned signal (~1.8s each)
for signal in signals:
    result = service.run_fast(
        signal,                    # Must have date_ret column
        sigvar='signal',
        byvar_list=['overall'],
    )
    summary, daily, turnover, tc = result
    print(f"Sharpe: {summary.iloc[0]['sharpe_ret']:.2f}")

# Standard path: use run() if signal needs alignment (~5s each)
result = service.run(raw_signal, lag=0, resid='off')
```

**When to use each:**
- `run_fast()`: Signal already has `date_ret` column, batch testing many signals (18x faster)
- `run()`: Raw signal needs date alignment, validation needed (~5s overhead)

### Per-Technique Speedups

| Technique | Speedup | Notes |
|-----------|---------|-------|
| Polars execution engine | ~6x | Columnar + SIMD + lazy optimization |
| NumPy residualization | ~12.5x | NumPy OLS vs statsmodels per-date |
| Universe-only filtering | ~1.8x | Smaller working set |
| Pre-joined master table | ~1.3x | Fewer join operations |
| Cached schemas/subsets | ~1.2x | Avoid repeated introspection |
| Strategic eager collects | ~1.1x | Reduce DAG overhead |

---

## Checklist for Future Polars Migrations

- [ ] Convert Pandas↔Polars only at boundaries
- [ ] Cache column schemas and commonly-used subsets at init
- [ ] Pre-sort any tables used in `join_asof`
- [ ] Build a pre-joined "master" table to reduce join count
- [ ] Filter data early (universe, date range) to shrink working set
- [ ] Collect at sensible boundaries (after preprocess, after weights)
- [ ] Use NumPy/Numba for per-group loops with small arrays
- [ ] Match join dates exactly (date_sig vs date_ret can differ!)
- [ ] Test numerical equivalence before optimizing further
- [ ] Debug row-level data when equivalence fails (the bug is often in the data, not the algorithm)
- [ ] Profile to find actual bottlenecks (usually joins, not conversions)
- [ ] Reuse engine instance for multi-signal testing (precompute once, run many)
