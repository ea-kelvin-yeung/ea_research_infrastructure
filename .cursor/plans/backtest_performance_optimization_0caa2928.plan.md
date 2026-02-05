---
name: Backtest Performance Optimization
overview: Create an optimized backtest engine that produces identical results to the original but runs significantly faster, with a 1-year test signal for validation.
todos:
  - id: test-signal
    content: Create 1-year test signal generator (tests/fixtures/test_signal_1year.parquet)
    status: completed
  - id: copy-engine
    content: Copy backtest_engine.py to backtest_engine_fast.py, rename class to BacktestFast
    status: completed
  - id: opt-residualize
    content: Implement fast residualization using numpy lstsq instead of statsmodels OLS
    status: completed
  - id: opt-turnover
    content: Implement fast turnover calculation using concat+groupby instead of outer merge
    status: completed
  - id: opt-ranking
    content: Replace lambda transforms with vectorized groupby methods for ranking/weights
    status: completed
  - id: opt-premerge
    content: Pre-merge byvar data once before gen_result() loop
    status: cancelled
  - id: opt-dtypes
    content: Add dtype optimization (int32, category) and groupby flags (sort=False, observed=True)
    status: completed
  - id: validation-tests
    content: Create equivalence tests comparing original vs optimized engine results
    status: completed
  - id: benchmark
    content: Create benchmark script to measure and report speedup
    status: completed
isProject: false
---

# Backtest Performance Optimization Plan

## Strategy

Keep the original `backtest_engine.py` unchanged. Create a new `backtest_engine_fast.py` with optimizations. Use a 1-year test signal to validate that both produce identical results.

---

## Phase 1: Test Infrastructure

### 1.1 Create 1-Year Test Signal

Create `tests/test_signal_1year.py` that generates a deterministic test signal:

```python
# Generate signal for 2018 (full year, ~250 trading days)
# Use fixed seed for reproducibility
np.random.seed(42)
# Cover ~500 securities x 250 days = 125,000 rows
```

Save to `tests/fixtures/test_signal_1year.parquet`

### 1.2 Create Validation Framework

Create `tests/test_engine_equivalence.py`:

```python
def test_results_identical():
    # Run original engine
    result_orig = Backtest(...).gen_result()
    
    # Run optimized engine
    result_fast = BacktestFast(...).gen_result()
    
    # Compare with tolerance for floating point
    pd.testing.assert_frame_equal(result_orig[0], result_fast[0], rtol=1e-10)
```

Test configurations:

- `resid=False` (baseline)
- `resid=True, resid_style='industry'`
- `resid=True, resid_style='all'`
- Different `weight` options (equal, value, volume)
- Different `byvar_list` options

---

## Phase 2: Optimized Engine

### 2.1 Copy Original Engine

```bash
cp backtest_engine.py backtest_engine_fast.py
```

Rename class to `BacktestFast`.

### 2.2 Optimization 1: Fast Residualization (Biggest Win)

**Location**: `pre_process()` method, lines 482-526

**Current (slow)**:

```python
temp.groupby("date").apply(model)  # OLS per date via statsmodels
```

**Optimized**:

```python
def fast_residualize(df, signal_col, factor_cols, industry_col=None):
    """Vectorized residualization using numpy lstsq."""
    residuals = []
    for date, group in df.groupby('date'):
        y = group[signal_col].values
        
        if industry_col:
            # Demean within industry (absorbs fixed effects)
            y_dm = y - group.groupby(industry_col)[signal_col].transform('mean').values
            X = group[factor_cols].values
            X_dm = X - group.groupby(industry_col)[factor_cols].transform('mean').values
        else:
            y_dm = y - y.mean()
            X_dm = group[factor_cols].values - group[factor_cols].mean().values
        
        # Fast least squares
        beta, _, _, _ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
        resid = y_dm - X_dm @ beta
        residuals.append(pd.Series(resid, index=group.index))
    
    return pd.concat(residuals)
```

**Expected speedup**: 5x-50x for residualization, 3x-10x end-to-end when `resid=True`

### 2.3 Optimization 2: Fast Turnover Calculation

**Location**: `backtest()` method, lines 973-1095

**Current (slow)**:

```python
turnover.merge(turnover2, how="outer", ...)  # Outer merge is expensive
```

**Optimized**:

```python
def fast_turnover(weights_df, byvar):
    """Concat + groupby sum trick instead of outer merge."""
    a = weights_df[['security_id', byvar, 'n', 'weight']].copy()
    b = a.copy()
    b['n'] = b['n'] + 1
    b['weight'] = -b['weight']  # Subtract yesterday
    
    z = pd.concat([a, b], ignore_index=True)
    diff = z.groupby(['security_id', byvar, 'n'], sort=False)['weight'].sum().abs()
    return diff.groupby([byvar, 'n'], sort=False).sum()
```

**Expected speedup**: 2x-5x for turnover, 1.2x-3x end-to-end

### 2.4 Optimization 3: Vectorized Ranking/Weights

**Location**: `gen_fractile()` lines 607-669, `portfolio_ls()` lines 785-860

**Current (slow)**:

```python
.transform(lambda x: x.rank(method="average"))
.transform(lambda x: 1 / x.count())
```

**Optimized**:

```python
# Direct rank without lambda
.rank(method='average')

# Count without lambda  
.transform('size')

# Equal weight
1.0 / df.groupby(['date', 'fractile'], sort=False)['security_id'].transform('size')
```

**Expected speedup**: 1.5x-4x for portfolio formation

### 2.5 Optimization 4: Pre-merge Data Once

**Location**: `gen_result()` method, lines 1508-1538

**Current (slow)**:

```python
for byvar in self.byvar_list:
    temp = pd.merge_asof(...)  # Repeated merges
```

**Optimized**:

```python
# Pre-merge all byvar columns once before loop
base = self.pre_process()
byvar_cols = ['cap', 'year', ...]  # All possible byvars
merged = base.merge(self.otherfile[['security_id', 'date'] + byvar_cols], ...)

for byvar in self.byvar_list:
    temp = merged  # No re-merge needed
```

### 2.6 Optimization 5: Dtype Optimization

Add at initialization:

```python
# Convert to efficient dtypes
self.retfile['security_id'] = self.retfile['security_id'].astype('int32')
self.otherfile['cap'] = self.otherfile['cap'].astype('category')
self.otherfile['industry_id'] = self.otherfile['industry_id'].astype('category')
```

Use `sort=False, observed=True` in all groupby operations.

---

## Phase 3: Validation and Benchmarking

### 3.1 Create Benchmark Script

Create `tests/benchmark_engines.py`:

```python
import time

# Benchmark configurations
configs = [
    {'resid': False},
    {'resid': True, 'resid_style': 'industry'},
    {'resid': True, 'resid_style': 'all'},
]

for cfg in configs:
    # Time original
    t0 = time.time()
    Backtest(**cfg).gen_result()
    orig_time = time.time() - t0
    
    # Time optimized
    t0 = time.time()
    BacktestFast(**cfg).gen_result()
    fast_time = time.time() - t0
    
    print(f"{cfg}: {orig_time:.1f}s -> {fast_time:.1f}s ({orig_time/fast_time:.1f}x)")
```

### 3.2 Validation Tests

Run equivalence tests for all combinations:

- 3 resid styles x 3 weight types x 3 byvar configs = 27 test cases
- Assert results match within `rtol=1e-10`

---

## Files to Create/Modify


| File                                       | Action | Description                        |
| ------------------------------------------ | ------ | ---------------------------------- |
| `backtest_engine_fast.py`                  | Create | Optimized engine (copy + optimize) |
| `tests/fixtures/test_signal_1year.parquet` | Create | 1-year test signal                 |
| `tests/test_engine_equivalence.py`         | Create | Validation tests                   |
| `tests/benchmark_engines.py`               | Create | Performance benchmarks             |
| `poc/wrapper.py`                           | Modify | Add option to use fast engine      |


---

## Expected Results


| Optimization         | Speedup                   |
| -------------------- | ------------------------- |
| Fast residualization | 5x-50x (resid block)      |
| Fast turnover        | 2x-5x (turnover block)    |
| Vectorized ranking   | 1.5x-4x (portfolio block) |
| Pre-merge data       | 1.1x-2x (overall)         |
| Dtype optimization   | 1.1x-1.5x (overall)       |


**Combined end-to-end**: 3x-10x faster when `resid=True`, 2x-5x when `resid=False`

---

## Risks and Mitigations

1. **Numerical precision differences**: Use `rtol=1e-10` tolerance in comparisons
2. **Edge cases in residualization**: Test with small groups, missing data
3. **Memory usage**: Pre-merging increases memory; monitor peak usage
4. **Categorical dtype issues**: Some operations may fail; keep fallback to object dtype

