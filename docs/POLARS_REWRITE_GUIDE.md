# Polars Reimplementation Guide

Lessons learned from rewriting the backtest engine from Pandas to Polars, achieving **~1.8x speedup** while maintaining exact numerical equivalence.

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

## Performance Summary

| Technique | Speedup | Notes |
|-----------|---------|-------|
| Universe-only filtering | ~1.8x | Single biggest win |
| Pre-joined master table | ~1.3x | Fewer join operations |
| Cached schemas/subsets | ~1.2x | Avoid repeated introspection |
| Strategic eager collects | ~1.1x | Reduce DAG overhead |
| NumPy residualization | ~1.5x (when used) | Tight loops beat Polars group_by |
| Boundary-only conversion | ~1.05x | Negligible once done right |

**Key insight:** Once conversions were fixed, **~97% of runtime was dominated by joins against the master table**, not conversion overhead. The wins came from:
1. **Shrinking master** (universe-only, date-range filtering)
2. **Joining fewer times / smarter** (pre-joined master, cached subsets)

---

## Checklist for Future Polars Migrations

- [ ] Convert Pandas↔Polars only at boundaries
- [ ] Cache column schemas and commonly-used subsets at init
- [ ] Pre-sort any tables used in `join_asof`
- [ ] Build a pre-joined "master" table to reduce join count
- [ ] Filter data early (universe, date range) to shrink working set
- [ ] Collect at sensible boundaries (after preprocess, after weights)
- [ ] Use NumPy/Numba for per-group loops with small arrays
- [ ] Test numerical equivalence before optimizing further
- [ ] Profile to find actual bottlenecks (usually joins, not conversions)
