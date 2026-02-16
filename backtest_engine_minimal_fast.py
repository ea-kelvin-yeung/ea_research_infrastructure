# backtest_min.py
# Minimal, blazing-fast backtest engine (Polars-first, NumPy only where it wins)
# Dependencies: polars>=0.20, numpy
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import polars as pl


# -----------------------------------------------------------------------------
# Date helpers (kept minimal; use if you need to derive date_sig/date_avail/date_ret)
# -----------------------------------------------------------------------------
def gen_date_trading(
    infile: Union[pl.DataFrame, pl.LazyFrame],
    datefile: Union[pl.DataFrame, pl.LazyFrame],
    varlist: Sequence[str],
    avail_hour: Union[int, str],
    date_signal: Optional[str] = None,
    date_available: Optional[str] = None,
    buffer: int = 0,
) -> pl.DataFrame:
    """
    Minimal trading-day date generator:
    - datefile must have: date (Date), n (Int)
    - infile must have: security_id, and either date_signal or date_available, plus varlist
    - avail_hour: int hour OR name of a datetime column from infile
    Returns: security_id, date_sig, date_avail, date_openret, date_ret, + varlist
    """
    lf = infile.lazy() if isinstance(infile, pl.DataFrame) else infile
    dlf = datefile.lazy() if isinstance(datefile, pl.DataFrame) else datefile

    if isinstance(avail_hour, str):
        lf = lf.with_columns(pl.col(avail_hour).dt.hour().alias("avail_hour"))
    else:
        lf = lf.with_columns(pl.lit(int(avail_hour)).alias("avail_hour"))

    if date_available is None:
        if date_signal is None:
            raise ValueError("Provide date_signal if date_available is not provided.")
        # date_avail = next trading day after date_signal
        lf = (
            lf.join(dlf.select("date", "n").rename({"date": date_signal}), on=date_signal, how="left")
              .with_columns((pl.col("n") + 1).alias("n"))
              .join(dlf.select("date", "n").rename({"date": "date_avail"}), on="n", how="left")
              .drop("n")
        )
    else:
        lf = lf.with_columns(pl.col(date_available).alias("date_avail"))

    if date_signal is None:
        # date_sig = previous trading day before date_avail
        lf = (
            lf.join(dlf.select("date", "n").rename({"date": "date_avail"}), on="date_avail", how="left")
              .with_columns((pl.col("n") - 1).alias("n"))
              .join(dlf.select("date", "n").rename({"date": "date_sig"}), on="n", how="left")
              .drop("n")
        )
    else:
        lf = lf.with_columns(pl.col(date_signal).alias("date_sig"))

    # openret / ret execution dates
    lf = (
        lf.join(dlf.select("date", "n").rename({"date": "date_avail"}), on="date_avail", how="left")
          .with_columns([
              (pl.when(pl.col("avail_hour") <= 8).then(pl.col("n")).otherwise(pl.col("n") + 1) + buffer).alias("n_openret"),
              (pl.when(pl.col("avail_hour") <= 15).then(pl.col("n") + 1).otherwise(pl.col("n") + 2) + buffer).alias("n_ret"),
          ])
          .join(dlf.select("date", "n").rename({"date": "date_openret", "n": "n_openret"}), on="n_openret", how="left")
          .join(dlf.select("date", "n").rename({"date": "date_ret", "n": "n_ret"}), on="n_ret", how="left")
          .select(["security_id", "date_sig", "date_avail", "date_openret", "date_ret", *varlist])
    )
    return lf.collect()


# -----------------------------------------------------------------------------
# Fast residualization (NumPy; avoids per-day Pandas group objects)
# -----------------------------------------------------------------------------
def _resid_by_date_industry(dates: np.ndarray, y: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """
    Residualize y using OLS with industry dummies, within each date.
    
    This matches statsmodels: sigvar ~ C(industry_id)
    which is equivalent to demeaning by industry mean.
    
    dates must be sorted.
    ind should be integer codes (0..K-1); NaN/None should already be dropped.
    """
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    uniq_dates, starts = np.unique(dates, return_index=True)
    ends = np.append(starts[1:], n)

    for s, e in zip(starts, ends):
        yd = y[s:e]
        ind_d = ind[s:e]
        if yd.size == 0:
            continue
        # Demean by industry (equivalent to OLS with industry dummies)
        _, inv = np.unique(ind_d, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        cnt[cnt == 0] = 1.0
        sums = np.bincount(inv, weights=yd).astype(np.float64)
        means = sums / cnt
        out[s:e] = yd - means[inv]
    return out


def _resid_by_date_industry_factors(
    dates: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    ind: np.ndarray,
    ridge: float = 1e-10,
) -> np.ndarray:
    """
    Residualize y on industry dummies + factors, within each date.
    
    This matches statsmodels: sigvar ~ factor1 + factor2 + ... + C(industry_id)
    
    Design matrix: [intercept, factors, industry_dummies (N-1)]
    Uses single OLS regression per date.
    
    dates must be sorted.
    """
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    uniq_dates, starts = np.unique(dates, return_index=True)
    ends = np.append(starts[1:], n)

    for s, e in zip(starts, ends):
        yd = y[s:e]
        Xd = X[s:e]
        ind_d = ind[s:e]
        m = yd.size
        if m == 0:
            continue

        # Remap industry codes to 0..k-1
        uniq_ind, inv = np.unique(ind_d, return_inverse=True)
        k = len(uniq_ind)
        
        # Build design matrix: [intercept, factors, industry_dummies]
        # Industry dummies: k-1 columns (drop first for identifiability, like statsmodels)
        n_factors = Xd.shape[1]
        n_dummies = k - 1  # Drop first industry as reference
        
        # Design matrix: intercept + factors + industry dummies
        design = np.empty((m, 1 + n_factors + n_dummies), dtype=np.float64)
        design[:, 0] = 1.0  # Intercept
        design[:, 1:1+n_factors] = Xd  # Factors
        
        # Industry dummies (one-hot, drop first category)
        for j in range(n_dummies):
            design[:, 1 + n_factors + j] = (inv == (j + 1)).astype(np.float64)
        
        # OLS: (X'X + ridge*I) β = X'y
        XT = design.T
        XTX = XT @ design
        XTX.flat[:: XTX.shape[0] + 1] += ridge  # Ridge for stability
        
        try:
            beta = np.linalg.solve(XTX, XT @ yd)
            out[s:e] = yd - design @ beta
        except np.linalg.LinAlgError:
            out[s:e] = np.nan
    
    return out


# -----------------------------------------------------------------------------
# Core backtester (fast + minimal)
# -----------------------------------------------------------------------------
Method = Literal["long_short", "long_only"]
InputType = Literal["value", "fractile", "position", "weight"]
SortMethod = Literal["single", "double"]
WeightMethod = Literal["equal", "value", "volume"]
TCModel = Literal["naive", "power_law"]
ResidStyle = Literal["none", "industry", "all"]  # industry-only or industry+factors


@dataclass(slots=True)
class BacktestConfig:
    sigvar: str
    method: Method = "long_short"
    input_type: InputType = "value"

    from_open: bool = False  # trade at open (use openret/resopenret), else close (ret/resret)
    fractile: Tuple[float, float] = (10.0, 90.0)  # percentile thresholds (short<=a, long>b)
    mincos: int = 10

    sort_method: SortMethod = "single"
    double_var: Optional[str] = None
    double_frac: int = 3

    weight: WeightMethod = "equal"
    upper_pct: float = 95.0  # cap weights by quantile (value/volume)

    resid: bool = False
    resid_style: ResidStyle = "all"
    resid_vars: Tuple[str, ...] = ("size", "value", "growth", "leverage", "volatility", "momentum", "yield")

    tc_model: TCModel = "naive"
    tc_level: Dict[str, float] = None  # bps dict for naive: {"big":2,"median":5,"small":10}
    tc_value: Tuple[float, float] = (0.35, 0.4)  # (beta, alpha) for power-law
    gmv_musd: float = 10.0

    insample: Literal["all", "i1", "i2"] = "all"
    byvars: Tuple[str, ...] = ("overall", "year", "cap")  # group dimensions


class BacktestFastMinimal:
    """
    Extremely fast minimal implementation:
    - Everything in Polars (LazyFrame) end-to-end
    - NumPy only for residualization (optional, and still fast)
    - No yfinance / statsmodels / pandas required
    """

    def __init__(
        self,
        master: Union[pl.DataFrame, pl.LazyFrame],
        datefile: Union[pl.DataFrame, pl.LazyFrame],
        cfg: BacktestConfig,
        *,
        asof_vars: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
    ):
        """
        master (required): daily panel with at least:
          security_id, date,
          ret,resret and/or openret,resopenret,
          cap, mcap, adv,
          industry_id,
          + factor columns (size,value,...) if you want residualization/exposures.
        datefile (required): date, n, and optional insample / insample2.
        asof_vars (optional): if you need merge_asof on date_sig (security_id, date, ...columns).
                              If None, byvars/factors are read from master at trade-date.
        """
        self.cfg = cfg
        # Use eager DataFrames for faster schema access
        self.master_df = master if isinstance(master, pl.DataFrame) else master.collect()
        self.date_df = datefile if isinstance(datefile, pl.DataFrame) else datefile.collect()
        self.asof_df = asof_vars if asof_vars is None or isinstance(asof_vars, pl.DataFrame) else asof_vars.collect()

        if cfg.tc_level is None:
            cfg.tc_level = {"big": 2.0, "median": 5.0, "small": 10.0}

        # Cache schemas for fast lookup
        self._master_schema = set(self.master_df.columns)
        self._date_schema = set(self.date_df.columns)
        
        # Pre-select commonly used subsets for faster joins
        self._date_n = self.date_df.select(["date", "n"])
        self._cap_df = self.master_df.select(["security_id", "date", "cap"]) if "cap" in self._master_schema else None

        # cache sorted for asof join
        self._asof_sorted: Optional[pl.DataFrame] = None
        if self.asof_df is not None:
            self._asof_sorted = self.asof_df.rename({"date": "date_sig"}).sort("date_sig")

        # master joins are always on (security_id,date)
        self._master_keys = ["security_id", "date"]

    # --------------------------
    # Internal fast joins
    # --------------------------
    def _join_master(self, lf: pl.LazyFrame, cols: Sequence[str], how: str = "inner") -> pl.LazyFrame:
        need = ["security_id", "date", *[c for c in cols if c not in ("security_id", "date")]]
        available = [c for c in need if c in self._master_schema]
        rhs = self.master_df.select(available).lazy()
        return lf.join(rhs, on=["security_id", "date"], how=how)

    def _apply_insample_filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.cfg.insample == "all":
            return lf
        col = "insample" if self.cfg.insample == "i1" else "insample2"
        return (
            lf.join(self.date_df.select(["date", col]).lazy(), on="date", how="left")
              .filter(pl.col(col) == 1)
              .drop(col)
        )

    # --------------------------
    # Preprocess: join returns + (optional) residualize
    # --------------------------
    def preprocess(self, signal: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
        """
        signal must have:
          security_id, date_sig, and either date_ret or date_openret, and cfg.sigvar.
        """
        cfg = self.cfg
        lf = signal.lazy() if isinstance(signal, pl.DataFrame) else signal

        trade_date_col = "date_openret" if cfg.from_open else "date_ret"
        ret_cols = ["openret", "resopenret"] if cfg.from_open else ["ret", "resret"]

        lf = (
            lf.select(["security_id", "date_sig", trade_date_col, cfg.sigvar])
              .rename({trade_date_col: "date"})
              .unique(subset=["security_id", "date"], keep="last")
              .drop_nulls()
        )

        lf = self._join_master(lf, ret_cols, how="inner")

        if cfg.from_open:
            lf = lf.rename({"openret": "ret", "resopenret": "resret"})
        lf = lf.filter(pl.col("ret") > -0.95)

        # Add standard grouping vars
        lf = lf.with_columns([
            pl.lit("overall").alias("overall"),
            pl.col("date").dt.year().alias("year"),
        ])

        # If you need cap (or any byvar) from asof on date_sig, do it here
        # (otherwise we will just read cap etc. from master at trade-date later)
        if self._asof_sorted is not None:
            # Join-asof (very fast in Polars, requires both sides sorted by on-key)
            left = lf.sort("date_sig")
            rhs = self._asof_sorted
            lf = (
                left.join_asof(
                    rhs,
                    by="security_id",
                    on="date_sig",
                    strategy="backward",
                    tolerance="20d",
                )
                .drop_nulls()
            )

        # Residualize signal (NumPy fast path) if requested
        if cfg.resid:
            if cfg.input_type != "value":
                raise ValueError("Residualization only applies when input_type='value'.")

            # OLD engine uses merge_asof on date_sig for factors with 5-day tolerance
            # We need to replicate this behavior exactly
            factor_cols = ["industry_id", *cfg.resid_vars]
            
            # Get factors from master via join_asof on date_sig (like OLD engine)
            # Master columns needed: security_id, date, industry_id, + factor vars
            master_factor_cols = ["security_id", "date", "industry_id"] + list(cfg.resid_vars)
            # Filter to only cols that exist in master
            available = set(self.master_df.columns)
            master_factor_cols = [c for c in master_factor_cols if c in available]
            
            factor_df = (
                self.master_df
                .select(master_factor_cols)
                .rename({"date": "date_sig"})
                .sort("date_sig")
            )
            
            # Join factors via merge_asof on date_sig with 5-day tolerance (like OLD engine)
            lf2 = (
                lf.sort("date_sig")
                .join_asof(
                    factor_df.lazy(),
                    by="security_id",
                    on="date_sig",
                    strategy="backward",
                    tolerance="5d",
                )
                .drop_nulls()
            )

            # Collect only necessary columns to NumPy, residualize, then reattach in Polars
            # Include "overall" and "year" for grouping
            df = (
                lf2.select(["security_id", "date_sig", "date", "ret", "resret", cfg.sigvar, 
                           "overall", "year", "industry_id", *cfg.resid_vars])
                   .sort(["date", "security_id"])
                   .collect()
            )
            dates = df["date"].to_numpy()
            y = df[cfg.sigvar].to_numpy().astype(np.float64, copy=False)

            # Pass original industry_id values - residualization functions 
            # factorize per-date internally
            ind = df["industry_id"].to_numpy()

            if cfg.resid_style == "industry":
                y_res = _resid_by_date_industry(dates, y, ind)
            elif cfg.resid_style == "all":
                X = np.column_stack([df[c].to_numpy().astype(np.float64, copy=False) for c in cfg.resid_vars])
                y_res = _resid_by_date_industry_factors(dates, y, X, ind)
            else:
                raise ValueError(f"Unknown resid_style: {cfg.resid_style}")

            out = df.with_columns(pl.Series(cfg.sigvar, y_res))
            return out.lazy()

        return lf

    # --------------------------
    # Ranking & positions
    # --------------------------
    def _add_positions(self, lf: pl.LazyFrame, byvar: str) -> pl.LazyFrame:
        cfg = self.cfg

        if cfg.input_type == "position":
            return lf.with_columns(pl.col(cfg.sigvar).cast(pl.Int8).alias("position"))

        if cfg.input_type == "weight":
            # sign(weight) determines position if not provided
            return lf.with_columns(
                pl.when(pl.col(cfg.sigvar) >= 0).then(1).otherwise(-1).alias("position")
            )

        if cfg.input_type == "fractile":
            # assume sigvar already encodes {-1,0,1} or min/max
            # minimal: min -> short, max -> long, else flat
            return lf.with_columns([
                pl.when(pl.col(cfg.sigvar) == pl.col(cfg.sigvar).min().over(["date", byvar])).then(-1)
                  .when(pl.col(cfg.sigvar) == pl.col(cfg.sigvar).max().over(["date", byvar])).then(1)
                  .otherwise(0)
                  .alias("position")
            ])

        # cfg.input_type == "value"
        a, b = cfg.fractile

        if cfg.sort_method == "double":
            if not cfg.double_var:
                raise ValueError("double_var must be set when sort_method='double'.")

            # double_var source: asof_vars if present else master at trade-date
            lf = self._join_master(lf, [cfg.double_var], how="inner") if self._asof_sorted is None else lf

            # First: rank double_var into 1..double_frac within (date, byvar)
            lf = lf.with_columns([
                (
                    (pl.col(cfg.double_var).rank("average").over(["date", byvar]) * cfg.double_frac
                     / pl.len().over(["date", byvar]))
                    .ceil()
                    .cast(pl.Int16)
                    .alias("fractile_double")
                )
            ])

            # Second: rank signal within (date, byvar, fractile_double) into percentiles
            lf = lf.with_columns([
                (
                    (pl.col(cfg.sigvar).rank("average").over(["date", byvar, "fractile_double"]) * 100.0
                     / pl.len().over(["date", byvar, "fractile_double"]))
                    .ceil()
                    .alias("pct")
                )
            ])
        else:
            lf = lf.with_columns([
                (
                    (pl.col(cfg.sigvar).rank("average").over(["date", byvar]) * 100.0
                     / pl.len().over(["date", byvar]))
                    .ceil()
                    .alias("pct")
                )
            ])

        lf = lf.with_columns([
            pl.when(pl.col("pct") <= a).then(-1)
              .when(pl.col("pct") > b).then(1)
              .otherwise(0)
              .alias("position")
        ]).drop(["pct", *([cfg.double_var, "fractile_double"] if cfg.sort_method == "double" else [])])

        return lf

    # --------------------------
    # Weights
    # --------------------------
    def _add_weights(self, lf: pl.LazyFrame, byvar: str) -> pl.LazyFrame:
        cfg = self.cfg
        group = ["date", byvar, "position"]

        if cfg.input_type == "weight":
            # use sigvar as weight; normalize within side if desired
            lf = lf.with_columns(pl.col(cfg.sigvar).cast(pl.Float64).alias("weight"))
            # normalize to sum(|w|)=1 per side (fast + stable)
            lf = lf.with_columns([
                (pl.col("weight") / pl.col("weight").abs().sum().over(group) * pl.col("position")).alias("weight")
            ])
            return lf

        # Equal weights: no need to join mcap/adv
        if cfg.weight == "equal":
            lf = lf.with_columns([
                (pl.col("position").cast(pl.Float64) / pl.len().over(group)).alias("weight")
            ])
            return lf

        # need mcap/adv for value/volume weights
        need = ["mcap", "adv"]
        lf = self._join_master(lf, need, how="inner")

        if cfg.weight == "value":
            x = "mcap"
        else:  # volume
            x = "adv"

        hi = cfg.upper_pct / 100.0
        lo = 1.0 - hi

        lf = lf.with_columns([
            pl.col(x).quantile(hi).over(group).alias("_hi"),
            pl.col(x).quantile(lo).over(group).alias("_lo"),
        ]).with_columns([
            pl.when(pl.col(x) > pl.col("_hi")).then(pl.col("_hi"))
              .when(pl.col(x) < pl.col("_lo")).then(pl.col("_lo"))
              .otherwise(pl.col(x))
              .alias("_x")
        ]).with_columns([
            (pl.col("_x") / pl.col("_x").sum().over(group) * pl.col("position").cast(pl.Float64)).alias("weight")
        ]).drop(["_hi", "_lo", "_x"])

        return lf

    # --------------------------
    # Turnover + transaction costs (captures entries/exits via outer join on shifted n)
    # --------------------------
    def _turnover_and_tc(self, weights: pl.LazyFrame, byvar: str) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        cfg = self.cfg

        w = (
            weights.select(["security_id", "date", byvar, "weight"])
                   .join(self._date_n.lazy(), on="date", how="left")
        )

        # counts (from current holdings only)
        counts = (
            w.with_columns([
                (pl.col("weight") > 0).cast(pl.Int32).alias("_l"),
                (pl.col("weight") < 0).cast(pl.Int32).alias("_s"),
            ])
            .group_by([byvar, "date"])
            .agg([
                pl.sum("_l").alias("numcos_l"),
                pl.sum("_s").alias("numcos_s"),
            ])
            .with_columns(pl.min_horizontal("numcos_l", "numcos_s").alias("numcos"))
        )

        # filter trading days by mincos
        if cfg.method == "long_short":
            ok_days = counts.filter(pl.col("numcos") >= cfg.mincos).select([byvar, "date"])
        else:
            ok_days = counts.filter(pl.col("numcos_l") >= cfg.mincos).select([byvar, "date"])

        w = w.join(ok_days, on=[byvar, "date"], how="inner")

        # Fast turnover: use left join + separate exits handling
        # But collect counts first to avoid repeated lazy computation
        counts_df = counts.collect()
        w_df = w.collect()
        
        cur = w_df.select(["security_id", byvar, "n", "weight"])
        prev = cur.with_columns((pl.col("n") + 1).alias("n")).rename({"weight": "weight_prev"})

        # Left join for current positions
        diff = cur.join(prev, on=["security_id", byvar, "n"], how="left").with_columns([
            pl.col("weight_prev").fill_null(0.0),
            (pl.col("weight") - pl.col("weight_prev").fill_null(0.0)).abs().alias("weight_diff"),
        ])
        
        # Anti-join for exits (positions in prev not in cur)
        exits = (
            prev.join(cur.select(["security_id", byvar, "n"]).unique(), on=["security_id", byvar, "n"], how="anti")
               .with_columns([
                   pl.lit(0.0).alias("weight"),
                   pl.col("weight_prev").abs().alias("weight_diff"),
               ])
        )
        
        # Concat and add date
        diff = pl.concat([diff, exits], how="diagonal_relaxed")
        diff = diff.join(self._date_n, on="n", how="left").drop_nulls(subset=["date"])
        
        # Zero out first day
        n_min = diff["n"].min()
        diff = diff.with_columns([
            pl.when(pl.col("n") == n_min).then(0.0).otherwise(pl.col("weight_diff")).alias("weight_diff")
        ])

        # turnover per day
        turnover_daily = diff.group_by([byvar, "date"]).agg(pl.sum("weight_diff").alias("turnover"))

        # transaction costs - join required columns from master (use eager joins)
        if cfg.tc_model == "naive":
            cap_df = self._cap_df
            diff = diff.join(cap_df, on=["security_id", "date"], how="left")
            tc_big = cfg.tc_level["big"] / 10000.0
            tc_med = cfg.tc_level["median"] / 10000.0
            tc_small = cfg.tc_level["small"] / 10000.0
            tc = (
                diff.with_columns([
                    pl.when(pl.col("cap") == 1).then(tc_big)
                      .when(pl.col("cap") == 2).then(tc_med)
                      .otherwise(tc_small)
                      .alias("tc_rate")
                ])
                .with_columns((pl.col("tc_rate") * pl.col("weight_diff")).alias("tc"))
                .group_by([byvar, "date"])
                .agg(pl.sum("tc").alias("tc"))
            )
        else:
            tc_cols_df = self.master_df.select(["security_id", "date", "vol", "adv", "close_adj"])
            diff = diff.join(tc_cols_df, on=["security_id", "date"], how="left")
            beta, alpha = cfg.tc_value
            tc = (
                diff.with_columns([
                    (beta * pl.col("vol") *
                     ((cfg.gmv_musd * 1_000_000.0 * pl.col("weight_diff") / pl.col("adv")) ** alpha)
                     / pl.col("close_adj")).alias("tc_rate")
                ])
                .with_columns((pl.col("tc_rate") * pl.col("weight_diff")).alias("tc"))
                .group_by([byvar, "date"])
                .agg(pl.sum("tc").alias("tc"))
            )

        return turnover_daily.lazy(), tc.lazy(), counts_df.lazy()

    # --------------------------
    # Main run per byvar
    # --------------------------
    def run(self, signal: Union[pl.DataFrame, pl.LazyFrame]) -> Dict[str, Any]:
        cfg = self.cfg

        base = self.preprocess(signal)
        
        # Collect to DataFrame for fast schema access
        base_df = base.collect()
        base_cols = set(base_df.columns)

        # Ensure needed byvars exist
        missing_byvars = [b for b in cfg.byvars if b not in ("overall", "year") and b not in base_cols]
        if missing_byvars:
            base_df = self._join_master(base_df.lazy(), missing_byvars, how="left").collect()
            base_cols = set(base_df.columns)

        results: list[pl.DataFrame] = []
        daily_overall: Optional[pl.DataFrame] = None
        turnover_overall: Optional[pl.DataFrame] = None

        for byvar in cfg.byvars:
            lf = base_df.lazy()
            if byvar not in ("overall", "year"):
                lf = lf.filter(pl.col(byvar).is_not_null())

            # positions + keep only long/short rows
            lf = self._add_positions(lf, byvar).filter(pl.col("position").is_in([-1, 1]))

            # weights
            lf = self._add_weights(lf, byvar)
            if cfg.method == "long_only":
                lf = lf.with_columns(pl.when(pl.col("weight") < 0).then(0.0).otherwise(pl.col("weight")).alias("weight"))
            
            # Collect weighted data for faster downstream operations
            weighted_df = lf.collect()

            # turnover + tc
            turnover_daily, tc_daily, counts_daily = self._turnover_and_tc(weighted_df.lazy(), byvar)

            # daily portfolio return - ret/resret already in base from preprocess
            lf2 = weighted_df.lazy()

            # Determine available factor columns
            weighted_cols = set(weighted_df.columns)
            available_factors = [f for f in cfg.resid_vars if f in weighted_cols]
            
            port_daily = (
                lf2.group_by([byvar, "date"])
                   .agg([
                       (pl.col("weight") * pl.col("ret")).sum().alias("ret"),
                       (pl.col("weight") * pl.col("resret")).sum().alias("resret"),
                       *[
                           (pl.col("weight") * pl.col(f)).sum().alias(f)
                           for f in available_factors
                       ],
                   ])
                   .join(tc_daily, on=[byvar, "date"], how="left")
                   .with_columns(pl.col("tc").fill_null(0.0))
                   .with_columns([
                       (pl.col("ret") - pl.col("tc")).alias("ret_net"),
                       (pl.col("resret") - pl.col("tc")).alias("resret_net"),
                   ])
            )

            port_daily = self._apply_insample_filter(port_daily.lazy()).collect()

            # stats - compute directly on eager DataFrame for speed
            sqrt252 = float(np.sqrt(252.0))
            port_sorted = port_daily.sort([byvar, "date"])
            
            # Compute drawdown efficiently
            port_sorted = port_sorted.with_columns([
                pl.col("ret").cum_sum().over(byvar).alias("cumret"),
            ]).with_columns([
                (pl.col("cumret") - pl.col("cumret").cum_max().over(byvar)).alias("drawdown")
            ])
            
            stats = (
                port_sorted.group_by(byvar)
                .agg([
                    pl.len().alias("num_date"),
                    (pl.mean("ret") * 252).alias("ret_ann"),
                    (pl.std("ret") * sqrt252).alias("ret_std"),
                    (pl.mean("resret") * 252).alias("resret_ann"),
                    (pl.std("resret") * sqrt252).alias("resret_std"),
                    (pl.mean("ret_net") * 252).alias("ret_net_ann"),
                    (pl.std("ret_net") * sqrt252).alias("ret_net_std"),
                    pl.min("drawdown").alias("maxdraw"),
                    (pl.col("ret") > 0).cast(pl.Float64).mean().alias("retPctPos"),
                    (pl.col("resret") > 0).cast(pl.Float64).mean().alias("resretPctPos"),
                    (pl.col("ret_net") > 0).cast(pl.Float64).mean().alias("retnetPctPos"),
                ])
                .with_columns([
                    (pl.col("ret_ann") / pl.col("ret_std")).alias("sharpe_ret"),
                    (pl.col("resret_ann") / pl.col("resret_std")).alias("sharpe_resret"),
                    (pl.col("ret_net_ann") / pl.col("ret_net_std")).alias("sharpe_retnet"),
                ])
            )

            # turnover + counts summary (divide by 4 to match OLD engine convention)
            turnover_sum = (
                self._apply_insample_filter(
                    turnover_daily.join(counts_daily, on=[byvar, "date"], how="left").lazy()
                )
                .group_by(byvar)
                .agg([
                    (pl.mean("turnover") / 4.0).alias("turnover"),
                    pl.mean("numcos_l").alias("numcos_l"),
                    pl.mean("numcos_s").alias("numcos_s"),
                ])
                .collect()
            )

            # exposure summary (means of factor exposures if present)
            port_cols = set(port_daily.columns)
            exp_cols = [f for f in cfg.resid_vars if f in port_cols]
            exposure = port_daily.group_by(byvar).agg([pl.mean(c).alias(c) for c in exp_cols])

            combo = (
                stats.join(turnover_sum, on=byvar, how="left")
                     .join(exposure, on=byvar, how="left")
                     .rename({byvar: "group"})
            )

            if byvar == "overall":
                daily_overall = port_daily.select(["date", "ret", "ret_net", "resret", "resret_net"])
                turnover_overall = turnover_daily.collect()

            results.append(combo)

        return {
            "summary": pl.concat(results, how="diagonal_relaxed"),
            "daily_overall": daily_overall,
            "turnover_overall": turnover_overall,
        }


__all__ = [
    "gen_date_trading",
    "BacktestConfig",
    "BacktestFastMinimal",
]


# -----------------------------------------------------------------------------
# Backward compatibility: lazy import wrapper
# -----------------------------------------------------------------------------
def __getattr__(name):
    """Lazy import BacktestFastV2 for backward compatibility."""
    if name == "BacktestFastV2":
        from backtest_wrapper import BacktestFastV2
        return BacktestFastV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
