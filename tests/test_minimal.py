# benchmark_backtest.py
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# -----------------------------
# Synthetic data generation
# -----------------------------
def make_synth_data(
    n_securities=3000,
    n_days=1260,          # ~5 years of trading days
    start="2015-01-01",
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Trading calendar (business days)
    dates = pd.bdate_range(start=start, periods=n_days)
    datefile = pd.DataFrame({"date": dates})
    datefile["n"] = np.arange(len(dates), dtype=np.int32)
    # Optional insample flags
    datefile["insample"] = (datefile["date"].dt.year <= datefile["date"].dt.year.median()).astype(np.int8)
    datefile["insample2"] = 1 - datefile["insample"]

    # Security universe
    sec = np.arange(1, n_securities + 1, dtype=np.int32)

    # Create dense daily panel (security_id x date)
    # This will be big: n_securities * n_days rows
    security_id = np.repeat(sec, n_days)
    date = np.tile(dates.values, n_securities)

    n = security_id.size

    # “Realistic-ish” distributions
    industry_id = rng.integers(0, 80, size=n, dtype=np.int16)
    sector_id = rng.integers(0, 12, size=n, dtype=np.int16)
    cap = rng.integers(1, 4, size=n, dtype=np.int8)  # 1 big, 2 med, 3 small

    # Returns: fat-ish tails but clipped
    ret = rng.normal(0, 0.012, size=n).astype(np.float32)
    ret = np.clip(ret, -0.25, 0.25)
    resret = (ret - rng.normal(0, 0.002, size=n)).astype(np.float32)

    openret = (ret + rng.normal(0, 0.003, size=n)).astype(np.float32)
    resopenret = (openret - rng.normal(0, 0.002, size=n)).astype(np.float32)

    # Liquidity / size
    mcap = np.exp(rng.normal(10, 1.0, size=n)).astype(np.float32)   # lognormal
    adv = (np.exp(rng.normal(12, 1.0, size=n)) / 1000).astype(np.float32)

    # Risk factors
    size = rng.normal(0, 1, size=n).astype(np.float32)
    value = rng.normal(0, 1, size=n).astype(np.float32)
    growth = rng.normal(0, 1, size=n).astype(np.float32)
    leverage = rng.normal(0, 1, size=n).astype(np.float32)
    volatility = np.abs(rng.normal(0, 1, size=n)).astype(np.float32)
    momentum = rng.normal(0, 1, size=n).astype(np.float32)
    yield_ = rng.normal(0, 1, size=n).astype(np.float32)

    # Power-law TC extras (optional)
    vol = np.abs(rng.normal(0.02, 0.01, size=n)).astype(np.float32)
    close_adj = np.exp(rng.normal(3, 0.2, size=n)).astype(np.float32)

    master = pd.DataFrame({
        "security_id": security_id,
        "date": pd.to_datetime(date),
        "ret": ret,
        "resret": resret,
        "openret": openret,
        "resopenret": resopenret,
        "industry_id": industry_id,
        "sector_id": sector_id,
        "cap": cap,
        "mcap": mcap,
        "adv": adv,
        "size": size,
        "value": value,
        "growth": growth,
        "leverage": leverage,
        "volatility": volatility,
        "momentum": momentum,
        "yield": yield_,
        "vol": vol,
        "close_adj": close_adj,
    })

    # -----------------------------
    # Signal file
    # -----------------------------
    # Make one signal per (security_id, date_sig) at the same frequency as master.
    # date_sig = date, date_ret = next trading day, date_openret = same day
    # (Enough for both engines; adjust if you want “availability” lags.)
    sig = master[["security_id", "date"]].copy()
    sig.rename(columns={"date": "date_sig"}, inplace=True)

    # Map date_sig -> date_ret (next trading day) using datefile
    # For last day, date_ret will be NaT and gets dropped by preprocess anyway.
    idx = pd.Series(np.arange(n_days), index=dates)
    n_sig = idx.loc[pd.to_datetime(sig["date_sig"]).values].to_numpy()
    n_ret = n_sig + 1
    date_ret = np.empty_like(sig["date_sig"].values, dtype="datetime64[ns]")
    date_ret[:] = np.datetime64("NaT")
    ok = n_ret < n_days
    date_ret[ok] = dates.values[n_ret[ok]]
    sig["date_ret"] = pd.to_datetime(date_ret)
    sig["date_openret"] = sig["date_sig"]

    # Create signal values (correlated a bit with a factor + noise)
    # so that sorting produces non-degenerate portfolios.
    # Align to master row order: sig row order equals master row order because we used master[["security_id","date"]].
    sig["my_signal"] = (0.2 * master["momentum"].to_numpy() + rng.normal(0, 1, size=n)).astype(np.float32)

    # Drop rows where date_ret is missing (last trading day)
    sig = sig.dropna(subset=["date_ret"]).reset_index(drop=True)

    return master, datefile, sig


# -----------------------------
# Timing helper
# -----------------------------
def time_it(fn, repeats=3, warmup=1, name=""):
    # warmup
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    print(f"{name:>18}:  mean={times.mean():.3f}s  min={times.min():.3f}s  max={times.max():.3f}s")
    return times


def main():
    N_SECURITIES = 3000
    N_DAYS = 1260
    master, datefile, sig = make_synth_data(N_SECURITIES, N_DAYS, seed=0)

    from backtest_engine_fast import BacktestFast
    from backtest_engine_minimal_fast import BacktestFastV2

    # Common backtest parameters
    bt_params = dict(
        infile=sig,
        retfile=master,
        otherfile=master,
        datefile=datefile,
        sigvar="my_signal",
        method="long_short",
        from_open=False,
        input_type="value",
        weight="equal",
        tc_model="naive",
        byvar_list=["overall"],
        resid=False,
        sort_method="single",
        verbose=False,
        master_data=None,
        ff_result=False,
        beta=False,
    )

    # -----------------------------
    # A) OLD ENGINE (BacktestFast)
    # -----------------------------
    def run_old():
        bt = BacktestFast(
            **bt_params,
            byvix=False,
            earnings_window=False,
        )
        return bt.gen_result()

    # -----------------------------
    # B) NEW ENGINE (BacktestFastV2) - Same interface!
    # -----------------------------
    def run_new():
        bt = BacktestFastV2(**bt_params)
        return bt.gen_result()

    # -----------------------------
    # Equivalence Test
    # -----------------------------
    print(f"Rows in master: {len(master):,} | rows in signal: {len(sig):,}")
    print("\n" + "="*60)
    print("EQUIVALENCE TEST")
    print("="*60)
    
    old_result = run_old()
    new_result = run_new()
    
    # Both engines now return tuple: (combo, daily_stats, turnover_raw)
    old_stats = old_result[0]
    new_stats = new_result[0]
    
    # Compare key metrics
    metrics_to_compare = [
        ("sharpe_ret", "sharpe_ret", 0.01),      # 1% tolerance
        ("ret_ann", "ret_ann", 0.01),
        ("ret_std", "ret_std", 0.01),
        ("turnover", "turnover", 0.01),
        ("numcos_l", "numcos_l", 0.01),
        ("numcos_s", "numcos_s", 0.01),
    ]
    
    print(f"\n{'Metric':<15} {'OLD':>12} {'NEW':>12} {'Diff':>12} {'Match':>8}")
    print("-" * 60)
    
    all_match = True
    for old_col, new_col, rel_tol in metrics_to_compare:
        if old_col in old_stats.columns and new_col in new_stats.columns:
            old_val = old_stats[old_col].iloc[0]
            new_val = new_stats[new_col].iloc[0]
            diff = abs(old_val - new_val)
            tol = max(abs(old_val) * rel_tol, 1e-6)
            match = diff < tol
            all_match = all_match and match
            status = "OK" if match else "FAIL"
            print(f"{old_col:<15} {old_val:>12.6f} {new_val:>12.6f} {diff:>12.6f} {status:>8}")
        else:
            print(f"{old_col:<15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'SKIP':>8}")
    
    print("-" * 60)
    if all_match:
        print("EQUIVALENCE: PASSED - All metrics match")
    else:
        print("EQUIVALENCE: FAILED - Some metrics differ")
    
    # -----------------------------
    # Benchmark
    # -----------------------------
    print("\n" + "="*60)
    print("SPEED BENCHMARK")
    print("="*60)
    time_it(lambda: run_old(), repeats=3, warmup=1, name="OLD BacktestFast")
    time_it(lambda: run_new(), repeats=5, warmup=2, name="NEW Minimal")


if __name__ == "__main__":
    main()
