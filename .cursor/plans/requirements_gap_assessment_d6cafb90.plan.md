---
name: Requirements Gap Assessment
overview: Assessment of which original requirements have been addressed in the current implementation, and what gaps remain.
todos:
  - id: cap-breakdown
    content: Add cap-tier breakdown (large/mid/small) to tearsheet - data exists in backtest output
    status: completed
  - id: year-breakdown
    content: Populate year-by-year performance table (already stubbed in template)
    status: completed
  - id: factor-correlation
    content: Add correlation of signal to risk factors (size, value, momentum, volatility)
    status: completed
  - id: coverage-metric
    content: Add signal coverage metric (avg securities per day, universe %)
    status: completed
  - id: composite-score
    content: Design and implement weighted composite quality score
    status: completed
isProject: false
---

# Requirements Gap Assessment

## Summary

The current POC addresses **most** of the core requirements, with some gaps remaining around global signal support and advanced quality metrics.

---

## Requirements Status

### 1. Unified Backtesting Engine

**Status: PARTIALLY ADDRESSED**

- Implemented: `BacktestFast` wrapper with 3-6x performance optimization
- Implemented: Standardized signal contract (security_id, date_sig, date_avail, signal)
- Gap: No global market support - current implementation is US-focused
- Gap: AlphaClub integration not addressed

### 2. Consistent Feature Reporting

**Status: ADDRESSED**

- Implemented: Suite runner with grid-based testing ([suite.py](poc/suite.py))
- Implemented: Standardized summary table (Sharpe, annual return, max drawdown, turnover)
- Implemented: HTML tearsheet generation ([tearsheet.py](poc/tearsheet.py))
- Implemented: Streamlit UI for consistent viewing ([app.py](poc/app.py))

### 3. Library of Standard Signals

**Status: PARTIALLY ADDRESSED**

- Implemented: `reversal_5d` (5-day short-term reversal)
- Implemented: `value` (value factor from risk file)
- Removed: `momentum_12_1` (removed per user request)
- Gap: Missing estimate revisions, quality, low-vol, size, and other common factors
- Location: [baselines.py](poc/baselines.py)

### 4. Data Management (No Prod Server Access)

**Status: ADDRESSED**

- Implemented: Snapshot system with versioned data ([catalog.py](poc/catalog.py))
- Implemented: Local parquet/pickle storage
- Implemented: Master data pre-merge for performance
- Implemented: Disk caching for baselines

### 5. One-Button Suite Execution

**Status: ADDRESSED**

- Implemented: Grid search over lags (0, 1, 2, 3, 5) and residualization (off, industry, all)
- Implemented: Single "Run Suite" button in Streamlit UI
- Implemented: Parallel execution via joblib
- Implemented: Automatic baseline comparison

### 6. Run Tracking and Reproducibility

**Status: ADDRESSED**

- Implemented: MLflow integration ([tracking.py](poc/tracking.py))
- Implemented: Git SHA tracking
- Implemented: Snapshot versioning
- Implemented: Metrics, artifacts, and correlations logging
- Implemented: Run history viewing in UI

### 7. Automated Signal Quality Heuristics

**Status: PARTIALLY ADDRESSED**

Current verdict logic in [tearsheet.py](poc/tearsheet.py) checks:

- Sharpe ratio thresholds (>=1.0 green, >=0.5 yellow, <0.5 red)
- Lag sensitivity (does Sharpe decay from lag0 to lag2?)
- Residualization sensitivity (does signal survive industry neutralization?)
- Baseline correlation (is correlation > 0.5 to reversal/value?)
- High turnover (> 200% flagged)
- Max drawdown (> 30% flagged)

**Gaps - Not Yet Implemented:**


| Metric                     | Description                                      | Status                                   |
| -------------------------- | ------------------------------------------------ | ---------------------------------------- |
| **Capacity**               | Is it driven by small caps? Latency sensitive?   | NOT IMPLEMENTED                          |
| **Coverage**               | How many securities does it cover?               | NOT IMPLEMENTED                          |
| **Consistency**            | Performance across time/cap/sector               | NOT IMPLEMENTED                          |
| **Risk Factor Exposure**   | Correlation to size, value, momentum, volatility | NOT IMPLEMENTED                          |
| **Year-by-Year Breakdown** | Performance consistency across years             | STUBBED (table exists but not populated) |
| **Sector Breakdown**       | Performance by sector                            | NOT IMPLEMENTED                          |
| **Automated Score**        | Composite score combining all metrics            | NOT IMPLEMENTED                          |


---

## Gap Analysis: What's Missing

### High Priority Gaps

1. **Capacity Metrics**
  - Market cap breakdown of signal (% large/mid/small)
  - ADV (average daily volume) analysis
  - Latency sensitivity (lag decay rate)
2. **Coverage Metrics**
  - Average securities per day with signal
  - Universe coverage percentage
3. **Consistency Metrics**
  - Year-over-year Sharpe stability
  - Sector-level performance
  - Cap-tier performance (already in backtest, needs surfacing)
4. **Risk Factor Exposure**
  - Correlation to common risk factors (size, value, momentum, volatility)
  - These exist in the risk file but aren't correlated to the signal

### Lower Priority Gaps

1. **Global Market Support** - Requires architecture changes
2. **AlphaClub Integration** - Out of scope for POC
3. **Automated Composite Score** - Would require defining weights for each metric

---

## Verdict System Coverage

The current "85% codifiable heuristics" mentioned in requirements maps to:


| Heuristic                   | Implemented?       |
| --------------------------- | ------------------ |
| Sharpe ratio threshold      | YES                |
| Not high turnover           | YES (flags > 200%) |
| Survives lag tests          | YES                |
| Survives residualization    | YES                |
| Not correlated to baselines | YES                |
| Not driven by small caps    | NO                 |
| Consistent across time      | NO                 |
| Consistent across sectors   | NO                 |
| Good capacity/liquidity     | NO                 |


**Current coverage: ~~5/9 heuristics (~~55%)**

---

## Recommendations

To reach the "85% codifiable" target, add:

1. **Cap-tier breakdown** - Data exists in backtest output, surface to tearsheet
2. **Year breakdown** - Already stubbed in template, just needs data
3. **Risk factor correlations** - Correlate signal to size/value/momentum/vol factors
4. **Coverage metric** - Count unique securities with signal per day
5. **Composite score** - Weighted sum of all metrics with configurable weights

