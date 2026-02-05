# Backtest Engine Documentation

## Overview

The `backtest_engine.py` module provides a comprehensive backtesting framework for evaluating trading signals. It supports long-short and long-only strategies with flexible signal types, portfolio weighting schemes, transaction cost models, and risk factor analysis.

---

## Table of Contents

1. [Core Components](#1-core-components)
2. [Data Files & Schema](#2-data-files--schema)
3. [Backtest Class](#3-backtest-class)
4. [Date Alignment Functions](#4-date-alignment-functions)
5. [Signal Processing Pipeline](#5-signal-processing-pipeline)
6. [Usage Examples](#6-usage-examples)
7. [Output Structure](#7-output-structure)

---

## 1. Core Components

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT FILES                               │
├─────────────────────────────────────────────────────────────────┤
│  infile (signal)  │  retfile (returns)  │  otherfile (factors)  │
│  datefile (calendar)  │  window_file (earnings, optional)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATE ALIGNMENT                                │
│  gen_date_trading() / gen_date_calender()                       │
│  Maps: date_sig → date_avail → date_ret / date_openret          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTEST CLASS                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ pre_process  │→ │ portfolio_ls │→ │ gen_weight   │          │
│  │ (merge data) │  │ (rank/sort)  │  │ (weighting)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    backtest()                             │  │
│  │  • Turnover calculation                                   │  │
│  │  • Transaction cost estimation                            │  │
│  │  • Portfolio return aggregation                           │  │
│  │  • Risk factor exposure                                   │  │
│  │  • Sharpe ratio, drawdown, statistics                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUTS                                    │
│  result (summary stats)  │  daily_stats  │  fractile analysis   │
│  Optional: ff_result (Fama-French)  │  beta estimates           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Files & Schema

The backtest engine expects four primary input DataFrames. These are typically loaded from parquet/pickle files in `backtest/data/`.

### 2.1 `ret.parquet` (retfile)

**Purpose:** Daily stock returns and price data.

| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Unique stock identifier |
| `date` | datetime | Trading date |
| `ret` | float | Daily close-to-close return |
| `resret` | float | Residual return (market-adjusted) |
| `openret` | float | *Optional.* Open-to-open return |
| `resopenret` | float | *Optional.* Residual open return |
| `vol` | float | *Optional.* Realized volatility (for power-law TC) |
| `close_adj` | float | *Optional.* Adjusted close price |
| `adv` | float | Average daily volume ($ or shares) |

**Required columns:** `security_id`, `date`, `ret`, `resret`

### 2.2 `risk.parquet` (otherfile)

**Purpose:** Risk factors, market cap, and classification data.

| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Unique stock identifier |
| `date` | datetime | As-of date for factor values |
| `industry_id` | int | GICS industry code |
| `sector_id` | int | GICS sector code |
| `size` | float | Size factor exposure (log market cap) |
| `value` | float | Value factor exposure (B/P ratio) |
| `growth` | float | Growth factor exposure |
| `leverage` | float | Leverage factor exposure |
| `volatility` | float | Volatility factor exposure |
| `momentum` | float | Momentum factor exposure |
| `yield` | float | Dividend yield factor exposure |
| `mcap` | float | Market capitalization ($) |
| `adv` | float | Average daily volume |
| `cap` | int | Cap category: 1=Large, 2=Medium, 3=Small |

**Required columns:** All columns listed above are required for full functionality.

### 2.3 `trading_date.pkl` (datefile)

**Purpose:** Trading calendar with date indexing and sample definitions.

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Trading date |
| `n` | int | Sequential trading day index (1, 2, 3, ...) |
| `insample` | int | *Optional.* In-sample flag (1=in, 0=out) |
| `insample2` | int | *Optional.* Alternative in-sample definition |

**Usage:** Used to align signal dates with actual trading dates and handle weekends/holidays.

### 2.4 `descriptor.parquet` (universe data)

**Purpose:** Daily universe membership for filtering eligible stocks.

| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Unique stock identifier |
| `as_of_date` / `date` | datetime | Date of universe membership |
| `universe_flag` | int | 1=in universe, 0=out of universe |

**Usage:** Filter focal stocks and peers to tradeable universe (liquidity, listing status, etc.).

### 2.5 Signal File (infile)

**Purpose:** The trading signal to backtest. Schema depends on `input_type`.

**For `input_type='value'`:**
| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Stock identifier |
| `date_sig` | datetime | Signal generation date |
| `date_avail` | datetime | Signal availability date |
| `date_ret` | datetime | Return measurement date |
| `{sigvar}` | float | The signal value (e.g., momentum score) |

**For `input_type='weight'`:**
| Column | Type | Description |
|--------|------|-------------|
| `security_id` | int | Stock identifier |
| `date` | datetime | Trading date |
| `{sigvar}` | float | Portfolio weight (-1 to +1) |

---

## 3. Backtest Class

### 3.1 Initialization Parameters

```python
Backtest(
    infile,           # Signal DataFrame
    retfile,          # Returns DataFrame  
    otherfile,        # Risk factors DataFrame
    datefile,         # Trading calendar DataFrame
    sigvar,           # Column name of signal variable
    
    # Strategy Configuration
    method='long_short',      # 'long_short' or 'long_only'
    long_index='sp500',       # Benchmark for long-only
    input_type='value',       # 'value', 'fractile', 'position', 'weight'
    fractile=[10, 90],        # Percentile thresholds [short, long]
    
    # Portfolio Construction
    weight='equal',           # 'equal', 'value', 'volume'
    upper_pct=0.95,          # Weight cap percentile
    mincos=10,               # Minimum stocks per side
    
    # Transaction Costs
    tc_model='naive',        # 'naive' or 'power_law'
    tc_level={'big': 2, 'median': 5, 'small': 10},  # bps by cap
    tc_value=[0.35, 0.4],    # Power-law parameters [beta, alpha]
    gmv=10,                  # Gross market value ($M) for power-law
    
    # Analysis Options
    byvar_list=['overall', 'year', 'cap'],  # Grouping variables
    resid=False,             # Residualize signal vs factors
    beta=False,              # Compute market beta
    ff_result=False,         # Fama-French regression
    
    # Filters
    insample='all',          # 'all', 'i1', 'i2'
    byvix=False,             # Split by VIX regime
    earnings_window=False,   # Split by earnings window
)
```

### 3.2 Key Methods

| Method | Description |
|--------|-------------|
| `pre_process()` | Merge signal with returns, apply residualization |
| `portfolio_ls(sig_file, byvar)` | Rank stocks, assign long/short positions |
| `gen_weight_ls(port, byvar)` | Calculate portfolio weights |
| `backtest(sigfile, byvar)` | Full backtest: turnover, returns, stats |
| `gen_fractile(port, n_fractile)` | Analyze return monotonicity by fractile |
| `cal_corr(infile, byvar)` | Calculate IC (information coefficient) |
| `ff_grab()` | Download Fama-French factors |
| `gen_result()` | Main entry point - runs complete backtest |

### 3.3 Signal Types (`input_type`)

| Type | Description | Example |
|------|-------------|---------|
| `'value'` | Raw signal values → ranked into percentiles | Momentum score |
| `'fractile'` | Pre-computed fractile assignments | Decile 1-10 |
| `'position'` | Pre-computed positions (-1, 0, +1) | Short/Neutral/Long |
| `'weight'` | Pre-computed portfolio weights | -0.02 to +0.02 |

---

## 4. Date Alignment Functions

### 4.1 `gen_date_trading()`

Aligns signals to trading days using the trading calendar.

```python
gen_date_trading(
    infile,           # Signal DataFrame
    datefile,         # Trading calendar
    varlist,          # Variables to carry forward
    avail_time,       # Hour signal becomes available (int or column name)
    date_signal=None, # Column with signal date
    date_available=None,  # Column with availability date
    buffer=0          # Extra days buffer
)
```

**Date Logic:**
- `date_sig`: When signal was generated
- `date_avail`: When signal becomes available (next trading day)
- `date_openret`: Trading date for open execution (if avail before 8am)
- `date_ret`: Trading date for close execution (if avail before 3pm)

**Example:**
```
Signal generated: Friday 4pm → date_sig = Friday
Available: Monday 8am → date_avail = Monday
Trade at close: Monday → date_ret = Monday
```

### 4.2 `gen_date_calender()`

Similar to `gen_date_trading()` but uses calendar days (less conservative).

---

## 5. Signal Processing Pipeline

### 5.1 Pre-Processing (`pre_process()`)

```
1. Merge signal file with returns
   infile[security_id, date_sig, date_ret, signal]
   + retfile[security_id, date, ret, resret]
   → Combined DataFrame

2. Optional: Residualize signal
   - 'all': Regress on factors + industry dummies
   - 'industry': Regress on industry dummies only
   - 'factor': Regress on factors only

3. Add grouping variables
   - 'overall' = 1 (all stocks)
   - 'year' = date.year
```

### 5.2 Portfolio Construction (`portfolio_ls()`)

```
1. Calculate percentile rank within group
   rank = groupby([byvar, date])[signal].rank()
   percentile = rank * 100 / group_size

2. Assign positions
   if percentile <= fractile[0]:  position = -1  (short)
   if percentile > fractile[1]:   position = +1  (long)
   else:                          position = 0   (neutral)

3. Optional: Double sort
   - First sort by control variable (e.g., size)
   - Then sort by signal within each control group
```

### 5.3 Weight Calculation (`gen_weight_ls()`)

| Weighting | Formula |
|-----------|---------|
| Equal | `weight = 1 / n_stocks_in_group` |
| Value | `weight = mcap / sum(mcap)` (capped at upper_pct) |
| Volume | `weight = adv / sum(adv)` (capped at upper_pct) |

### 5.4 Backtest Calculation (`backtest()`)

```
1. Turnover Calculation
   turnover = |weight_today - weight_yesterday|
   
2. Transaction Cost
   - Naive: tc = turnover × tc_level[cap] / 10000
   - Power-law: tc = β × σ × (GMV × |Δw| / ADV)^α / price

3. Portfolio Returns
   ret_gross = Σ(weight × ret)
   ret_net = ret_gross - tc

4. Statistics
   - Annualized return: mean(ret) × 252
   - Sharpe ratio: ann_return / (std × √252)
   - Max drawdown: min(cumret - cummax(cumret))
   
5. Factor Exposure
   exposure = Σ(weight × factor)
```

---

## 6. Usage Examples

### 6.1 Basic Long-Short Backtest

```python
from backtest_engine import Backtest, gen_date_trading
import pandas as pd

# Load data
ret = pd.read_parquet('backtest/data/ret.parquet')
risk = pd.read_parquet('backtest/data/risk.parquet')
tradingday = pd.read_pickle('backtest/data/trading_date.pkl')

# Prepare signal (momentum example)
signal = ret.groupby('security_id').apply(
    lambda x: x.set_index('date')['ret'].rolling(20).sum()
).reset_index()
signal.columns = ['security_id', 'date', 'momentum']

# Align dates
signal_aligned = gen_date_trading(
    infile=signal,
    datefile=tradingday,
    varlist=['momentum'],
    avail_time=16,  # Signal available after 4pm
    date_signal='date'
)

# Run backtest
bt = Backtest(
    infile=signal_aligned,
    retfile=ret,
    otherfile=risk,
    datefile=tradingday,
    sigvar='momentum',
    method='long_short',
    fractile=[10, 90],
    weight='equal',
    byvar_list=['overall', 'year']
)

result, daily_stats, fractile = bt.gen_result()
print(result)
```

### 6.2 Pre-Computed Weights (Peer Strategy)

```python
# Load peer-weighted signal
weights_df = pd.read_parquet('peer_weights.parquet')
# Columns: security_id, date, weight

bt = Backtest(
    infile=weights_df,
    retfile=ret,
    otherfile=risk,
    datefile=tradingday,
    sigvar='weight',
    input_type='weight',  # Use pre-computed weights
    method='long_short',
    weight_adj=True,      # Normalize weights to sum to ±1
)

result, daily_stats, _ = bt.gen_result()
```

### 6.3 With Fama-French Analysis

```python
bt = Backtest(
    infile=signal_aligned,
    retfile=ret,
    otherfile=risk,
    datefile=tradingday,
    sigvar='momentum',
    beta=True,
    benchmark='sp500',
    ff_result=True,
    ff_model='ff5',  # 5-factor model
)

result, daily_stats, fractile, ff_result = bt.gen_result()
print(ff_result)  # Alpha, factor loadings, p-values
```

---

## 7. Output Structure

### 7.1 `result` DataFrame

Summary statistics by group (overall, year, cap).

| Column | Description |
|--------|-------------|
| `group` | Grouping variable value |
| `numcos_l` | Avg number of long positions |
| `numcos_s` | Avg number of short positions |
| `num_date` | Number of trading days |
| `ret_ann` | Annualized gross return |
| `ret_std` | Annualized volatility |
| `sharpe_ret` | Gross Sharpe ratio |
| `ret_net_ann` | Annualized net return (after TC) |
| `sharpe_retnet` | Net Sharpe ratio |
| `maxdraw` | Maximum drawdown |
| `turnover` | Average daily turnover |
| `size`, `value`, ... | Average factor exposures |

### 7.2 `daily_stats` DataFrame

Daily time series data.

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `ret` | Daily gross return |
| `ret_net` | Daily net return |
| `resret` | Daily residual return |
| `cumret` | Cumulative gross return |
| `cumretnet` | Cumulative net return |
| `drawdown` | Current drawdown from peak |
| `turnover` | Daily turnover |
| `size`, `value`, ... | Daily factor exposures |

### 7.3 `fractile` DataFrame

Return and factor profile by signal fractile.

| Column | Description |
|--------|-------------|
| `fractile` | Signal decile (1-10) |
| `ret` | Annualized return for this fractile |
| `resret` | Annualized residual return |
| `numcos` | Average number of stocks |
| `size`, `value`, ... | Average factor exposures |

---

## 8. Transaction Cost Models

### 8.1 Naive Model (`tc_model='naive'`)

Simple fixed cost by market cap tier:

```python
tc_level = {'big': 2, 'median': 5, 'small': 10}  # basis points

tc = turnover × tc_level[cap] / 10000
```

### 8.2 Power-Law Model (`tc_model='power_law'`)

Market impact model based on trade size relative to liquidity:

```python
tc_beta, tc_alpha = 0.35, 0.4

# Impact = β × σ × (participation_rate)^α
tc = tc_beta × volatility × (GMV × |Δweight| / ADV)^tc_alpha / price
```

Where:
- `GMV`: Gross market value of portfolio ($M)
- `volatility`: Stock's realized volatility
- `ADV`: Average daily dollar volume
- `Δweight`: Change in portfolio weight

---

## 9. Integration with Minimal Backtest Framework

The `backtest/minimal/` folder contains a simplified backtest framework that uses the same data files but with a more streamlined API for peer momentum strategies.

### Data Flow

```
backtest/data/
├── ret.parquet         → Daily returns (ret, resret)
├── risk.parquet        → Risk factors (size, value, momentum, ...)
├── descriptor.parquet  → Universe flags (universe_flag)
└── trading_date.pkl    → Trading calendar

        │
        ▼
┌───────────────────────────────────────┐
│         backtest/minimal/             │
│  ├── universe.py  (load universe)     │
│  ├── signals.py   (compute signals)   │
│  ├── peer_scores.py (peer weights)    │
│  └── run.py (orchestration)           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│      backtest_engine.Backtest         │
│  (portfolio construction & stats)     │
└───────────────────────────────────────┘
```

---

## 10. Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty portfolio | `mincos` filter too strict | Reduce `mincos` or check data coverage |
| NaN returns | Missing data in retfile | Ensure date alignment is correct |
| Very high turnover | Signal too noisy | Smooth signal or increase holding period |
| Unrealistic Sharpe | Look-ahead bias | Verify `date_ret` is AFTER `date_avail` |
| Memory error | Large dataset | Process year-by-year |

---

## Appendix: File Locations

```
ea_industry_classification/
├── backtest/
│   ├── backtest_engine.py     # This module
│   ├── data/                  # Data files (gitignored)
│   │   ├── ret.parquet
│   │   ├── risk.parquet
│   │   ├── descriptor.parquet
│   │   └── trading_date.pkl
│   ├── minimal/               # Simplified backtest framework
│   │   ├── tests/             # Organized test files
│   │   ├── oracle/            # Oracle modules
│   │   ├── granger/           # Granger causality
│   │   ├── peer/              # Peer model training
│   │   └── ...
│   └── output/                # Backtest results
├── data/
│   ├── returns/               # Return data queries
│   └── factors/               # Factor data queries
└── docs/
    └── BACKTEST_ENGINE.md     # This documentation
```
