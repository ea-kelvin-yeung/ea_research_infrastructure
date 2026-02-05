# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:34 2020

@author: yunan
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import yfinance as yf
import pandas_datareader as pdr
from functools import reduce

pd.options.mode.chained_assignment = None


# %%

def _fast_residualize(group, y_col, x_cols):
    y = group[y_col].to_numpy(dtype="float64", copy=False)
    if x_cols:
        x_mat = group[x_cols].to_numpy(dtype="float64", copy=False)
        x_mat = np.column_stack([np.ones(len(y), dtype="float64"), x_mat])
        coef, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        resid = y - x_mat @ coef
    else:
        resid = y - y.mean()
    return pd.Series(resid, index=group.index)


def _fast_residualize_industry(group, y_col, industry_col):
    y = group[y_col].to_numpy(dtype="float64", copy=False)
    codes, _ = pd.factorize(group[industry_col], sort=False)
    if (codes < 0).any():
        mask = codes >= 0
        resid = np.full(len(y), np.nan, dtype="float64")
        codes = codes[mask]
        y_mask = y[mask]
        counts = np.bincount(codes)
        y_mean = np.bincount(codes, weights=y_mask) / counts
        resid[mask] = y_mask - y_mean[codes]
    else:
        counts = np.bincount(codes)
        y_mean = np.bincount(codes, weights=y) / counts
        resid = y - y_mean[codes]
    return pd.Series(resid, index=group.index)


def _vectorized_resid_all_numpy(dates, y, X, industry_codes):
    """
    Pure NumPy implementation of 'factors within industry' residualization.
    
    This is 20-50x faster than groupby().apply() because it avoids creating
    thousands of Pandas objects (one per day).
    
    Args:
        dates: 1D array of dates (must be sorted!)
        y: 1D array of target values (signal)
        X: 2D array of factors (n_samples, n_factors)
        industry_codes: 1D array of integer industry codes
        
    Returns:
        1D array of residuals
    """
    n_samples = len(y)
    residuals = np.full(n_samples, np.nan)
    
    # Identify the start/end indices for each date
    # (assumes dates are sorted, which we do in the calling function)
    unique_dates, start_indices = np.unique(dates, return_index=True)
    end_indices = np.append(start_indices[1:], n_samples)
    
    # Loop over dates using raw array slicing (extremely fast)
    for start, end in zip(start_indices, end_indices):
        # 1. Slice the day's data
        y_day = y[start:end]
        X_day = X[start:end]
        ind_day = industry_codes[start:end]
        
        # Skip empty days
        if len(y_day) == 0:
            continue
            
        # 2. Local Factorize / Map Industries to 0..K for bincount
        uniq_inds, inverse_inds = np.unique(ind_day, return_inverse=True)
        n_inds = len(uniq_inds)
        
        # 3. Calculate weights (counts) per industry
        counts = np.bincount(inverse_inds, minlength=n_inds).astype(float)
        # Avoid division by zero
        counts[counts == 0] = 1 
        
        # 4. Demean Y by Industry
        y_sums = np.bincount(inverse_inds, weights=y_day, minlength=n_inds)
        y_means = y_sums / counts
        y_demean = y_day - y_means[inverse_inds]
        
        # 5. Demean X by Industry (Vectorized per column)
        X_demean = np.empty_like(X_day)
        for i in range(X_day.shape[1]):
            x_col = X_day[:, i]
            x_sums = np.bincount(inverse_inds, weights=x_col, minlength=n_inds)
            x_means = x_sums / counts
            X_demean[:, i] = x_col - x_means[inverse_inds]
            
        # 6. Linear Regression (Normal Equation: (X'X)^-1 X'Y)
        XT = X_demean.T
        try:
            # Add small epsilon to diagonal for numerical stability
            XTX = XT @ X_demean
            XTX += np.eye(XTX.shape[0]) * 1e-10
            beta = np.linalg.solve(XTX, XT @ y_demean)
            resid = y_demean - X_demean @ beta
            residuals[start:end] = resid
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (rare in backtests)
            residuals[start:end] = np.nan

    return residuals


def _vectorized_resid_industry_numpy(dates, y, industry_codes):
    """
    Pure NumPy implementation of industry-only demeaning (no factor regression).
    
    This is the fastest possible residualization - just subtracts industry means.
    
    Args:
        dates: 1D array of dates (must be sorted!)
        y: 1D array of target values (signal)
        industry_codes: 1D array of integer industry codes
        
    Returns:
        1D array of residuals (y - industry_mean)
    """
    n_samples = len(y)
    residuals = np.full(n_samples, np.nan)
    
    unique_dates, start_indices = np.unique(dates, return_index=True)
    end_indices = np.append(start_indices[1:], n_samples)
    
    for start, end in zip(start_indices, end_indices):
        y_day = y[start:end]
        ind_day = industry_codes[start:end]
        
        if len(y_day) == 0:
            continue
            
        # Local factorize
        uniq_inds, inverse_inds = np.unique(ind_day, return_inverse=True)
        n_inds = len(uniq_inds)
        
        # Calculate industry means
        counts = np.bincount(inverse_inds, minlength=n_inds).astype(float)
        counts[counts == 0] = 1
        y_sums = np.bincount(inverse_inds, weights=y_day, minlength=n_inds)
        y_means = y_sums / counts
        
        # Demean
        residuals[start:end] = y_day - y_means[inverse_inds]

    return residuals


def _fast_residualize_factors_within_industry(group, y_col, x_cols, industry_col):
    y = group[y_col].to_numpy(dtype="float64", copy=False)
    codes, _ = pd.factorize(group[industry_col], sort=False)
    if (codes < 0).any():
        mask = codes >= 0
        resid = np.full(len(y), np.nan, dtype="float64")
        group = group.loc[mask]
        y = y[mask]
        codes = codes[mask]
    else:
        resid = None

    counts = np.bincount(codes)
    y_mean = np.bincount(codes, weights=y) / counts
    y_demean = y - y_mean[codes]

    if not x_cols:
        out = y_demean
    else:
        x_mat = group[x_cols].to_numpy(dtype="float64", copy=False)
        x_demean = np.empty_like(x_mat, dtype="float64")
        for j in range(x_mat.shape[1]):
            x_sum = np.bincount(codes, weights=x_mat[:, j])
            x_mean = x_sum / counts
            x_demean[:, j] = x_mat[:, j] - x_mean[codes]
        coef, _, _, _ = np.linalg.lstsq(x_demean, y_demean, rcond=None)
        out = y_demean - x_demean @ coef

    if resid is None:
        return pd.Series(out, index=group.index)
    resid[mask] = out
    return pd.Series(resid, index=group.index)
"""

##############################################################################
########################     Backtesting     #################################
##############################################################################



Generate the date variables to use in the backtest engine, usually, we need to specify the available date and time
Four date variable:
   date_sig: the signal date or event date, usually infered rather than given.
             The signal date is used to merge other factor variable we use to double sort or residualize
   date_avail: the signal available date, must be given
   date_ret: the date for close return, T+1 after trading at close
   date_openret: the date for open return, T after trading at open

Two method: calendar day or trading day. Trading day is more conservative thereby prefered

"""


def gen_date_calender(
    infile,
    tradingdays_file,
    sigvar,
    avail_time,
    date_signal=None,
    date_available=None,
    buffer=0,
):

    temp = infile.copy()
    tradingday = tradingdays_file[["date", "n"]]
    # generate available date if not provided, set it one day after the signal as_of_date
    if date_signal:
        temp["date_sig"] = temp[date_signal]
    elif not date_signal:
        temp["date_sig"] = temp[date_available] - pd.Timedelta(days=1)

    if date_available:
        temp["date_avail"] = temp[date_available]
    elif not date_available:
        temp["date_avail"] = temp[date_signal] + pd.Timedelta(days=1)

    if isinstance(avail_time, str):
        temp["avail_hour"] = temp[avail_time].dt.hour
    else:
        temp["avail_hour"] = avail_time

    temp = temp.merge(
        tradingday.rename(columns={"date": "date_avail"}), how="left", on="date_avail"
    )
    temp["trading"] = np.where(temp["n"].isnull(), 0, 1)
    temp = temp.sort_values(by=["security_id", "date_avail"])
    temp["n"] = temp.groupby("security_id")["n"].ffill()
    temp["n_openret"] = (
        np.where(
            (temp["trading"] == 1) & (temp["avail_hour"] <= 8), temp["n"], temp["n"] + 1
        )
        + buffer
    )
    temp["n_ret"] = (
        np.where(
            (temp["trading"] == 1) & (temp["avail_hour"] <= 15),
            temp["n"] + 1,
            temp["n"] + 2,
        )
        + buffer
    )
    temp = temp.merge(
        tradingday.rename(columns={"n": "n_openret", "date": "date_openret"}),
        on="n_openret",
    )
    temp = temp.merge(
        tradingday.rename(columns={"n": "n_ret", "date": "date_ret"}), on="n_ret"
    )

    temp = temp[
        ["security_id", sigvar, "date_sig", "date_avail", "date_openret", "date_ret"]
    ]

    return temp


def gen_date_trading(
    infile,
    datefile,
    varlist,
    avail_time,
    date_signal=None,
    date_available=None,
    buffer=0,
):

    temp = infile.copy()
    tradingday = datefile[["date", "n"]]

    if isinstance(avail_time, str):
        temp["avail_hour"] = temp[avail_time].dt.hour
    else:
        temp["avail_hour"] = avail_time

    if date_available:
        temp["date_avail"] = temp[date_available]
    elif not date_available:
        temp = temp.merge(
            tradingday.rename(columns={"date": date_signal}), on=date_signal
        )
        temp["n"] = temp["n"] + 1
        temp = temp.merge(
            tradingday.rename(columns={"date": "date_avail"}), on="n"
        ).drop(columns=["n"])

    # generate available date if not provided, set it one day after the signal as_of_date
    if date_signal:
        temp["date_sig"] = temp[date_signal]
    elif not date_signal:
        temp = temp.merge(
            tradingday.rename(columns={"date": date_available}), on=date_available
        )
        temp["n"] = temp["n"] - 1
        temp = temp.merge(tradingday.rename(columns={"date": "date_sig"}), on="n").drop(
            columns=["n"]
        )
    temp = temp[["security_id", "date_avail", "date_sig", "avail_hour"] + varlist]

    temp = temp.merge(
        tradingday.rename(columns={"date": "date_avail"}), how="left", on="date_avail"
    )
    temp["n_openret"] = (
        np.where(temp["avail_hour"] <= 8, temp["n"], temp["n"] + 1) + buffer
    )
    temp["n_ret"] = (
        np.where(temp["avail_hour"] <= 15, temp["n"] + 1, temp["n"] + 2) + buffer
    )

    temp = temp.merge(
        tradingday.rename(columns={"n": "n_openret", "date": "date_openret"}),
        on="n_openret",
    )
    temp = temp.merge(
        tradingday.rename(columns={"n": "n_ret", "date": "date_ret"}), on="n_ret"
    )

    temp = temp[
        ["security_id", "date_sig", "date_avail", "date_openret", "date_ret"] + varlist
    ]

    return temp


"""
#############     Parameter List       #################
        
    ##########  Must specify  ########
    
    infile:         the signal file, which already has the four date variables. Must be daily frequency
    
    retfile:        the file which contains return, factor and all other backtest variables. Must be daily frequency
    
                    the retfile must contain the following variables: 
                        security_id, date, ret, resret, industry_id, sector_id, size, value,\
                        growth, leverage, volatility, momentum, yield, mcap, adv, cap
                        
                    optional variables are:
                        openret, resopenret, vol, close_adj
                    
    datefile:       the file which contains trading date info, plus insample definition
    
    sigvar:         the signal variable
    
    ##########  Optional parameters   ##########
    
    method:         the backtest method
                    Input is string of 'long_short' or 'long_only'
                    
    long_index:     the benchmarked index for long-only backtest
                    Input is string of 'sp500', 'russell1000', 'russell2000', or 'russell3000'
       
    byvar_list:     the list of group by which the result will be displayed. 
                    Input is an arbitrary list: default is ['overall','year','cap']
                    It is recommended to always include 'overall' in the byvar list
                
    from_open:      trade at market open or market close, 
                    Input is Boolean; default is False
                
    input_type:     the type of signal
                    Input is a string of 'value','fractile','position','weight'; default is 'value'
    
    mincos:         the minimum number of stocks on the long and short side for that day to be included in the result
                    Input is a integer; default is 10
    
    insample:       whether the backtest use insample or the whole sample
                    Input is a string of 'all', 'i1','i2'; default is 'all'
    
    holding_period: The holding period of the portfolio
                    Input is a integer, default is 1
    
    output:         the output style, a string of 'simple' or 'full'
    
    fractile:       the long-short percentile threshold
                    Input is a list of short, long percentile
    
    
    @###   Weighting method   ###
    
    weight:         whether the portfolio is equal-weight, value-weighted or volume-weighted
                    Input is a string of 'equal', 'value', 'volume'; default is 'equal'
    
    upper_pct:      The upper cap of weight, determined by the x percentile of value or volume
                    default is 0.95
                    
    @###   Transaction cost assumption   ###
    
    
    tc_model:       which transaction cost model is used
                    Input is a string of 'naive', 'power_law'
    
    tc_level:       the sub-parameter if tc_model is 'naive'
                    Input is a dictionary of basis point for big, median, small cap. default is 2, 5, 10
    
    tc_value:       the sub-parameter if tc_model is 'power_law'
                    Input is a list of two key parameter of the power-law model, default is 0.35, 0.4
    
    gmv:            the sub-parameter if tc_model is 'power_law'
    
    
    
    @###   Additional results   ###
    # produce results under different macro-volatility conditions or by earnings window
    
    byvix:          whether display results seperately for high and low vix environment
    
    earnings_window: whether display results seperately for earnings window
    
    window_file:    the sample which contains earning window specification, must be specified if earnings_window = True
    
    
    @###   Double sort   ###
    # under double sort, it's better to just have 'overall' and 'year' in the byvar_list
    
    sort_method:    single-sorting or double sorting
                    Input is a string of 'single', 'double'; default is 'single'
    
    double_file:    the sample that contain the double sort variable, usually just the retfile
                    Input is the variable name or 'None'; default is 'None' which will use the retfile
                 
    double_var:     the variable for double sorting, must be specified if sort_method = 'double'
    
    double_frac:    the number of fractiles for the double sort variable
                    Input is integer, default is 3
    
    
    @###   residualization   ###
    
    resid:          whether or not to perform residualization of the signal w.r.t. risk factors, 
                    Input is Boolean; default is False
            
    resid_style:    the specification of the residualization.
                    Input is string of 'industry','factor','all'; default is 'all'
            
    resid_varlist:  the resid variables. 
                    Input is a list of variable; default is  ['size','value', 'growth', 'leverage', 'volatility', 'momentum', 'yields']
    
    @###    generate beta   ###

    beta:           To generate beta for overall sample and each year
                    Input is Boolean, default is False

    benchmark:      The market benchmark used to calculate beta
                    Input is string of 'sp500', 'russell1000', 'russell3000' or  'ff', default is 'sp500'

    @###    generate Fama-French result   ###
    
    ff_result:      To generate fama-french regression result for the long-short return
                    Input is Boolean, default is False
                    
    ff_model:       The fama-french factors to be included
                    Input is string of 'ff3', 'ff5', 'ff7' or 'ff7_industry'




"""


class BacktestFast:
    """Optimized version of Backtest class with performance improvements."""
    
    def __init__(
        self,
        infile,
        retfile,
        otherfile,
        datefile,
        sigvar,
        factor_list=[
            "size",
            "value",
            "growth",
            "leverage",
            "volatility",
            "momentum",
            "yield",
        ],
        method="long_short",
        long_index="sp500",
        byvar_list=["overall", "year", "cap"],
        from_open=False,
        input_type="value",
        weight_adj=False,
        mincos=10,
        insample="all",
        output="simple",
        fractile=[10, 90],
        frac_stretch=False,
        weight="equal",
        upper_pct=0.95,
        tc_model="naive",
        tc_level={"big": 2, "median": 5, "small": 10},
        tc_value=[0.35, 0.4],
        gmv=10,
        byvix=False,
        earnings_window=False,
        window_file=None,
        sort_method="single",
        double_file=None,
        double_var=None,
        double_frac=3,
        resid=False,
        resid_style="all",
        resid_varlist=[
            "size",
            "value",
            "growth",
            "leverage",
            "volatility",
            "momentum",
            "yields",
        ],
        beta=False,
        benchmark="sp500",
        ff_result=False,
        ff_model="ff3",
        verbose=False,
        master_data=None,
    ):

        self.verbose = verbose
        self.infile = infile
        self.retfile = retfile
        self.otherfile = otherfile
        self.datefile = datefile
        self.master_data = master_data  # Pre-merged indexed DataFrame for fast joins
        self.sigvar = sigvar
        # byvar_list should not include mcap, adv and factor
        self.byvar_list = byvar_list
        self.insample = insample
        self.gmv = gmv
        self.method = method
        self.long_index = long_index

        self.from_open = from_open
        self.input_type = input_type
        self.weight_adj = weight_adj
        self.resid = resid
        self.resid_style = resid_style
        self.fractile = fractile
        self.frac_stretch = frac_stretch
        self.weight = weight
        self.upper_pct = upper_pct

        self.sort_method = sort_method
        self.double_file = double_file
        self.double_var = double_var
        self.double_frac = double_frac

        self.tc_model = tc_model
        self.tc_level = tc_level
        self.mincos = mincos
        self.factor_list = factor_list
        self.earnings_window = earnings_window
        self.window_file = window_file
        self.output = output
        self.tc_value = tc_value
        self.byvix = byvix
        self.resid_varlist = resid_varlist

        self.beta = beta
        self.benchmark = benchmark
        self.ff_result = ff_result
        self.ff_model = ff_model
        self._double_file_sorted = None
        
        # Pre-index datefile for fast lookups
        self._datefile_by_date = None
        self._datefile_by_n = None

    def _get_datefile_by_date(self):
        """Get datefile indexed by date for fast lookups."""
        if self._datefile_by_date is None:
            self._datefile_by_date = self.datefile.set_index('date')
        return self._datefile_by_date
    
    def _get_datefile_by_n(self):
        """Get datefile indexed by n for fast lookups."""
        if self._datefile_by_n is None:
            self._datefile_by_n = self.datefile.set_index('n')
        return self._datefile_by_n
    
    def _fast_date_lookup(self, df, date_col, cols_to_add):
        """
        Fast lookup from datefile using index-based reindex.
        
        Much faster than merge because it uses numpy array indexing
        instead of hash-based key matching.
        """
        datefile_indexed = self._get_datefile_by_date()
        
        # Use reindex to lookup values (avoids is_unique check)
        lookup_result = datefile_indexed.reindex(df[date_col])
        
        for col in cols_to_add:
            if col in lookup_result.columns:
                df[col] = lookup_result[col].values
        
        return df
    
    def _fast_n_lookup(self, df, n_col, cols_to_add, rename_map=None):
        """
        Fast lookup from datefile by n using index-based reindex.
        """
        datefile_indexed = self._get_datefile_by_n()
        
        # Use reindex to lookup values
        lookup_result = datefile_indexed.reindex(df[n_col])
        
        for col in cols_to_add:
            if col in lookup_result.columns:
                new_col = rename_map.get(col, col) if rename_map else col
                df[new_col] = lookup_result[col].values
        
        return df

    def _fast_join_master(self, df, cols, how='inner'):
        """
        Fast join using pre-indexed master_data with numpy get_indexer.
        
        Uses get_indexer + numpy take instead of merge/join/reindex to
        completely bypass pandas is_unique checks and hash-based key matching.
        
        Args:
            df: DataFrame with security_id and date columns
            cols: List of columns to retrieve from master_data
            how: Join type ('inner' or 'left')
            
        Returns:
            DataFrame with requested columns joined
        """
        if self.master_data is None:
            # Fall back to traditional merge
            return df.merge(
                self.retfile[['security_id', 'date'] + [c for c in cols if c in self.retfile.columns]],
                on=['security_id', 'date'],
                how=how
            )
        
        # Filter columns that exist in master
        available_cols = [c for c in cols if c in self.master_data.columns]
        if not available_cols:
            return df
        
        # Create MultiIndex from df's keys (vectorized)
        idx = pd.MultiIndex.from_arrays(
            [df['security_id'].values, df['date'].values],
            names=['security_id', 'date']
        )
        
        # Get indexer positions (numpy array of positions, -1 for missing)
        positions = self.master_data.index.get_indexer(idx)
        
        # Add columns using numpy take (much faster than reindex)
        result = df.copy()
        for col in available_cols:
            col_values = self.master_data[col].values
            # Use take with mode='clip' for safety, then set -1 positions to NaN
            taken = np.take(col_values, np.clip(positions, 0, len(col_values) - 1))
            taken = taken.astype(float)  # Ensure float for NaN support
            taken[positions == -1] = np.nan
            result[col] = taken
        
        # For inner join, drop rows where joined columns are NaN
        if how == 'inner':
            result = result.dropna(subset=available_cols)
        
        return result

    def _get_otherfile_cols(self, cols):
        """Direct column access without sorting (faster than sorting)."""
        return self.otherfile[cols]

    def _get_doublefile_sorted(self):
        if self._double_file_sorted is None:
            if self.double_file is None:
                if self.double_var is None:
                    raise RuntimeError("double_var must be set for double sorting")
                df = self.otherfile[["security_id", "date", self.double_var]]
            else:
                df = self.double_file
            self._double_file_sorted = df.sort_values("date")
        return self._double_file_sorted

    def pre_process(self):
        # merge with return data
        for x in self.byvar_list:
            if x in ["mcap", "adv"] + self.factor_list:
                raise RuntimeError(f"byvar {x} need to be renamed")

        if self.from_open:
            temp = (
                self.infile[["security_id", "date_sig", "date_openret", self.sigvar]]
                .rename(columns={"date_openret": "date"})
                .drop_duplicates(subset=["security_id", "date"], keep="last")
                .dropna()
            )

            # Always use retfile for initial join (not master, which is ret ∩ risk)
            # Master filtering happens later in gen_fractile/portfolio_ls
            temp = temp.merge(
                self.retfile[["security_id", "date", "openret", "resopenret"]].rename(
                    columns={"openret": "ret", "resopenret": "resret"}
                ),
                how="inner",
                on=["security_id", "date"],
            )
            temp = temp[temp["ret"] > -0.95]
        else:
            temp = (
                self.infile[["security_id", "date_sig", "date_ret", self.sigvar]]
                .rename(columns={"date_ret": "date"})
                .drop_duplicates(subset=["security_id", "date"], keep="last")
                .dropna()
            )

            # Always use retfile for initial join (not master, which is ret ∩ risk)
            temp = temp.merge(
                self.retfile[["security_id", "date", "ret", "resret"]],
                how="inner",
                on=["security_id", "date"],
            )

        if self.resid:
            if self.input_type != "value":
                raise RuntimeError(
                    "signal residualization only applies to numerical signal"
                )
            temp = pd.merge_asof(
                temp.sort_values(by=["date_sig"]),
                self._get_otherfile_cols(
                    [
                        "security_id",
                        "date",
                        "industry_id",
                        "sector_id",
                        "size",
                        "value",
                        "growth",
                        "leverage",
                        "volatility",
                        "momentum",
                        "yield",
                    ]
                ).rename(columns={"date": "date_sig", "yield": "yields"}).sort_values(by="date_sig"),
                by="security_id",
                on="date_sig",
                allow_exact_matches=True,
                direction="backward",
                tolerance=pd.Timedelta("5d"),
            ).dropna()
            temp = temp.sort_values(by=["date", "security_id"]).reset_index(drop=True)

            # Prepare numpy arrays for vectorized residualization (20-50x faster)
            dates = temp["date"].values
            y = temp[self.sigvar].to_numpy(dtype="float64")
            industry_codes, _ = pd.factorize(temp["industry_id"], sort=False)

            if self.resid_style == "industry":
                # Pure industry demeaning (fastest)
                resid_values = _vectorized_resid_industry_numpy(dates, y, industry_codes)
            elif self.resid_style == "factor":
                # Factor regression without industry grouping
                X = temp[list(self.resid_varlist)].to_numpy(dtype="float64")
                # For factor-only, we don't demean by industry, just regress
                # Use a simple all-same-industry code to skip industry demeaning
                dummy_codes = np.zeros(len(y), dtype=int)
                resid_values = _vectorized_resid_all_numpy(dates, y, X, dummy_codes)
            elif self.resid_style == "all":
                # Full: factors within industry (most comprehensive)
                X = temp[list(self.resid_varlist)].to_numpy(dtype="float64")
                resid_values = _vectorized_resid_all_numpy(dates, y, X, industry_codes)
            else:
                raise RuntimeError(f"Unknown resid_style: {self.resid_style}")
            
            temp[self.sigvar] = resid_values
            temp = temp[
                ["security_id", "date_sig", "date", "ret", "resret", self.sigvar]
            ]

        temp["overall"] = 1
        temp["year"] = temp["date"].dt.year
        return temp

    def cal_corr(self, infile, byvar):

        cor_file = infile.merge(
            self._get_otherfile_cols(["security_id", "date"] + self.factor_list),
            on=["security_id", "date"],
        )

        if self.insample == "i1":
            cor_file = self._fast_date_lookup(cor_file, 'date', ['insample'])
            cor_file = cor_file[cor_file["insample"] == 1]
        elif self.insample == "i2":
            cor_file = self._fast_date_lookup(cor_file, 'date', ['insample2'])
            cor_file = cor_file[cor_file["insample2"] == 1]

        cor = cor_file.groupby(["date", byvar], sort=False)[
            [self.sigvar, "ret", "resret"]
        ].corr(method="spearman")
        cor = cor[cor[self.sigvar] == 1]
        cor["date"] = [x[0] for x in cor.index]
        cor[byvar] = [x[1] for x in cor.index]
        cor["_name_"] = [x[2] for x in cor.index]
        cor = cor[["date", byvar, "ret", "resret"]].reset_index(drop=True)

        corn = cor_file[["date", byvar]].copy()
        corn["n"] = corn.groupby(["date", byvar], sort=False)["date"].transform("size")
        corn = corn.drop_duplicates().reset_index(drop=True)
        cor = pd.merge(cor, corn, on=["date", byvar])
        cor = cor[
            cor["n"]
            >= self.mincos * 100 / (min(self.fractile[0], 1 - self.fractile[1]))
        ]
        cor = cor.drop("n", axis=1, inplace=False)
        cor = cor.drop(columns=["date"])

        cor["retIC"] = cor.groupby(byvar, sort=False)["ret"].transform("mean")
        cor["resretIC"] = cor.groupby(byvar, sort=False)["resret"].transform("mean")

        cor = cor.drop(["ret", "resret"], axis=1, inplace=False)
        cor = (
            cor.drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={byvar: "group"})
        )
        if byvar == "overall":
            cor["group"] = "overall"

        return cor

    def gen_fractile(self, port, n_fractile):

        # to generate average return and factor exposure for all fractile -
        #     primarily check for shape of factor exposure and monotonicity of fractile return
        # No need for this if input_type is position or weight

        if self.input_type == "value":
            # double sorting
            if self.sort_method == "double":
                if self.double_file is None:
                    self.double_file = self.otherfile[
                        ["security_id", "date", self.double_var]
                    ]
                port = pd.merge_asof(
                    port.sort_values(by=["date_sig"]),
                    self._get_doublefile_sorted().rename(
                        columns={"date": "date_doublevar"}
                    ),
                    by="security_id",
                    left_on="date_sig",
                    right_on="date_doublevar",
                    allow_exact_matches=True,
                    direction="backward",
                )

                port = port.dropna()

                grp_date = port.groupby("date", sort=False)[self.double_var]
                rank = grp_date.rank(method="average")
                group_size = grp_date.transform("count")
                port["fractile_double"] = np.ceil(
                    rank * self.double_frac / group_size
                )

                grp_fd = port.groupby(
                    ["fractile_double", "date"], sort=False
                )[self.sigvar]
                group_size = grp_fd.transform("count")
                rank = grp_fd.rank(method="max" if self.frac_stretch else "average")
                percentile = np.ceil(rank * 100 / group_size)
                if self.frac_stretch:
                    pct_group = percentile.groupby(
                        [port["fractile_double"], port["date"]], sort=False
                    )
                    min_pct = pct_group.transform("min")
                    max_pct = pct_group.transform("max")
                    percentile = (percentile - min_pct + 1) * 100 / (
                        max_pct - min_pct + 1
                    )
                port["fractile"] = np.ceil(percentile * n_fractile / 100)

            # single sorting
            elif self.sort_method == "single":

                grp_date = port.groupby("date", sort=False)[self.sigvar]
                group_size = grp_date.transform("count")
                rank = grp_date.rank(method="max" if self.frac_stretch else "average")
                percentile = np.ceil(rank * 100 / group_size)
                if self.frac_stretch:
                    pct_group = percentile.groupby(port["date"], sort=False)
                    min_pct = pct_group.transform("min")
                    max_pct = pct_group.transform("max")
                    percentile = (percentile - min_pct + 1) * 100 / (
                        max_pct - min_pct + 1
                    )
                port["fractile"] = np.ceil(percentile * n_fractile / 100)

        elif self.input_type == "fractile":
            port["fractile"] = port[self.sigvar]

        port = port[["security_id", "date", "fractile", "ret", "resret"]]
        
        # Use fast join if master_data available
        needed_cols = ["adv", "mcap"] + self.factor_list
        if self.master_data is not None:
            port = self._fast_join_master(port, needed_cols, how='inner')
        else:
            port = port.merge(
                self._get_otherfile_cols(["security_id", "date"] + needed_cols),
                on=["security_id", "date"],
            )

        if self.weight == "equal":
            group_size = port.groupby(["date", "fractile"], sort=False)[
                "security_id"
            ].transform("count")
            port["weight"] = 1 / group_size
        elif self.weight == "value":
            mcap_grp = port.groupby(["date", "fractile"], sort=False)["mcap"]
            port["mcap_h"] = mcap_grp.transform(
                "quantile", q=self.upper_pct / 100
            )
            port["mcap_l"] = mcap_grp.transform(
                "quantile", q=1 - self.upper_pct / 100
            )
            port["mcap"] = np.select(
                [port["mcap"] > port["mcap_h"], port["mcap"] < port["mcap_l"]],
                [port["mcap_h"], port["mcap_l"]],
                default=port["mcap"],
            )
            mcap_sum = port.groupby(["date", "fractile"], sort=False)[
                "mcap"
            ].transform("sum")
            port["weight"] = port["mcap"] / mcap_sum
        elif self.weight == "volume":
            adv_grp = port.groupby(["date", "fractile"], sort=False)["adv"]
            port["adv_h"] = adv_grp.transform("quantile", q=self.upper_pct / 100)
            port["adv_l"] = adv_grp.transform("quantile", q=1 - self.upper_pct / 100)
            port["adv"] = np.select(
                [port["adv"] > port["adv_h"], port["adv"] < port["adv_l"]],
                [port["adv_h"], port["adv_l"]],
                default=port["adv"],
            )
            adv_sum = port.groupby(["date", "fractile"], sort=False)["adv"].transform(
                "sum"
            )
            port["weight"] = port["adv"] / adv_sum
        port2 = port[
            ["security_id", "date", "ret", "resret", "fractile", "weight"]
            + self.factor_list
        ]

        numcos = (
            port2.groupby(["date", "fractile"], sort=False)["security_id"]
            .count()
            .reset_index()
        )
        numcos = numcos[numcos["fractile"].isin([1, n_fractile])]
        numcos["numcos"] = numcos.groupby("date", sort=False)["security_id"].transform(
            "min"
        )
        numcos = numcos[["date", "numcos"]].drop_duplicates()
        port2 = port2.merge(numcos, on="date")
        port2 = port2[port2["numcos"] >= self.mincos]

        weight = port2["weight"].to_numpy(dtype="float64", copy=False)
        scale_cols = [
            "ret",
            "resret",
            "size",
            "value",
            "growth",
            "leverage",
            "volatility",
            "momentum",
            "yield",
        ]
        port2[scale_cols] = port2[scale_cols].to_numpy(copy=False) * weight[:, None]

        check = port2.groupby(["date", "fractile"], sort=False).sum().reset_index()
        check2 = (
            port2.groupby(["date", "fractile"], sort=False)["security_id"]
            .count()
            .reset_index()
            .rename(columns={"security_id": "numcos"})
            .groupby("fractile", sort=False)
            .mean()
            .reset_index()
        )
        check3 = check.groupby(["fractile"], sort=False)[["ret", "resret"]].mean() * 252
        check = check.groupby("fractile", sort=False)[
            ["size", "value", "growth", "leverage", "volatility", "momentum", "yield"]
        ].mean()
        check = check.merge(check2, on="fractile")
        check = check.merge(check3.reset_index(), on="fractile")

        return check

    def portfolio_ls(self, sig_file, byvar):

        temp = sig_file.copy()
        if self.input_type == "value":
            # double sorting
            if self.sort_method == "double":
                if self.double_file is None:
                    self.double_file = self.otherfile[
                        ["security_id", "date", self.double_var]
                    ]
                temp = pd.merge_asof(
                    temp.sort_values(by=["date_sig"]),
                    self._get_doublefile_sorted().rename(
                        columns={"date": "date_doublevar"}
                    ),
                    by="security_id",
                    left_on="date_sig",
                    right_on="date_doublevar",
                    allow_exact_matches=True,
                    direction="backward",
                )

                temp = temp.dropna()

                grp_bd = temp.groupby([byvar, "date"], sort=False)[self.double_var]
                rank = grp_bd.rank(method="average")
                group_size = grp_bd.transform("count")
                temp["fractile_double"] = np.ceil(
                    rank * self.double_frac / group_size
                )

                grp_fd = temp.groupby(
                    [byvar, "fractile_double", "date"], sort=False
                )[self.sigvar]
                group_size = grp_fd.transform("count")
                rank = grp_fd.rank(method="max" if self.frac_stretch else "average")
                percentile = np.ceil(rank * 100 / group_size)
                if self.frac_stretch:
                    pct_group = percentile.groupby(
                        [temp[byvar], temp["fractile_double"], temp["date"]],
                        sort=False,
                    )
                    min_pct = pct_group.transform("min")
                    max_pct = pct_group.transform("max")
                    percentile = (percentile - min_pct + 1) * 100 / (
                        max_pct - min_pct + 1
                    )
                temp["percentile"] = percentile

                temp["position"] = np.select(
                    [
                        temp["percentile"] <= self.fractile[0],
                        temp["percentile"] > self.fractile[1],
                    ],
                    [-1, 1],
                    default=0,
                )
                temp = temp.drop(columns=["date_doublevar", self.double_var])

            # single sorting
            elif self.sort_method == "single":
                grp_bd = temp.groupby([byvar, "date"], sort=False)[self.sigvar]
                group_size = grp_bd.transform("count")
                rank = grp_bd.rank(method="max" if self.frac_stretch else "average")
                percentile = np.ceil(rank * 100 / group_size)
                if self.frac_stretch:
                    pct_group = percentile.groupby(
                        [temp[byvar], temp["date"]], sort=False
                    )
                    min_pct = pct_group.transform("min")
                    max_pct = pct_group.transform("max")
                    percentile = (percentile - min_pct + 1) * 100 / (
                        max_pct - min_pct + 1
                    )
                temp["percentile"] = percentile

                temp["position"] = np.select(
                    [
                        temp["percentile"] <= self.fractile[0],
                        temp["percentile"] > self.fractile[1],
                    ],
                    [-1, 1],
                    default=0,
                )

            temp = temp.drop(columns=["percentile"])

        # The byvar is meaningless if the input_type is not value
        elif self.input_type == "fractile":
            temp["position"] = np.select(
                [
                    temp[self.sigvar] == temp[self.sigvar].min(),
                    temp[self.sigvar] == temp[self.sigvar].max(),
                ],
                [-1, 1],
                default=0,
            )

        elif self.input_type == "position":
            temp["position"] = temp[self.sigvar]

        temp = temp[temp["position"].isin([-1, 1])]
        
        # Use fast join if master_data available
        if self.master_data is not None:
            temp = self._fast_join_master(temp, ["adv", "mcap"], how='inner')
        else:
            temp = temp.merge(
                self._get_otherfile_cols(["security_id", "date", "adv", "mcap"]),
                on=["security_id", "date"],
            )

        return temp

    def gen_weight_ls(self, port, byvar):

        if self.input_type != "weight":
            # generate weight
            if self.weight == "equal":
                group_size = port.groupby(
                    [byvar, "date", "position"], sort=False
                )["security_id"].transform("count")
                port["weight"] = (1 / group_size) * port["position"]
            elif self.weight == "value":
                mcap_grp = port.groupby([byvar, "date", "position"], sort=False)["mcap"]
                port["mcap_h"] = mcap_grp.transform(
                    "quantile", q=self.upper_pct / 100
                )
                port["mcap_l"] = mcap_grp.transform(
                    "quantile", q=1 - self.upper_pct / 100
                )
                port["mcap"] = np.select(
                    [port["mcap"] > port["mcap_h"], port["mcap"] < port["mcap_l"]],
                    [port["mcap_h"], port["mcap_l"]],
                    default=port["mcap"],
                )
                mcap_sum = port.groupby([byvar, "date", "position"], sort=False)[
                    "mcap"
                ].transform("sum")
                port["weight"] = (port["mcap"] / mcap_sum) * port["position"]
            elif self.weight == "volume":
                adv_grp = port.groupby([byvar, "date", "position"], sort=False)["adv"]
                port["adv_h"] = adv_grp.transform(
                    "quantile", q=self.upper_pct / 100
                )
                port["adv_l"] = adv_grp.transform(
                    "quantile", q=1 - self.upper_pct / 100
                )
                port["adv"] = np.select(
                    [port["adv"] > port["adv_h"], port["adv"] < port["adv_l"]],
                    [port["adv_h"], port["adv_l"]],
                    default=port["adv"],
                )
                adv_sum = port.groupby([byvar, "date", "position"], sort=False)[
                    "adv"
                ].transform("sum")
                port["weight"] = (port["adv"] / adv_sum) * port["position"]

        elif self.input_type == "weight":
            port["weight"] = port[self.sigvar]
            if "position" not in port.columns.to_list():
                port["position"] = np.where(port["weight"] > 0, 1, -1)
            if self.weight_adj == True:
                weight_sum = port.groupby([byvar, "date", "position"], sort=False)[
                    "weight"
                ].transform("sum")
                port["weight"] = (port["weight"] / weight_sum) * port["position"]
            port["weight"]
        port2 = port[
            ["security_id", "date", byvar, "ret", "resret", "position", "weight"]
        ]

        return port2

    def backtest(self, sigfile, byvar):

        if self.input_type != "weight":
            port = self.portfolio_ls(sigfile, byvar)
            if self.verbose:
                print("finish generating portfolio")
            weight_file = self.gen_weight_ls(port, byvar)

        elif self.input_type == "weight":
            if self.verbose:
                print("finish generating portfolio")
            weight_file = self.gen_weight_ls(sigfile, byvar)

        if self.verbose:
            print("finish calculating weight")

        # cor = self.cal_corr(sigfile, byvar)

        if self.method == "long_only":
            weight_file.loc[weight_file["weight"] < 0, "weight"] = 0

        turnover = weight_file[["security_id", "date", byvar, "weight"]].copy()
        # Fast date lookup instead of merge
        turnover = self._fast_date_lookup(turnover, 'date', ['n'])
        # number of stocks with non-zero position
        turnover["numcos_l"] = np.where(turnover["weight"] > 0, 1, 0)
        turnover["numcos_s"] = np.where(turnover["weight"] < 0, 1, 0)
        turnover["numcos_l"] = turnover.groupby(
            [byvar, "date"], sort=False
        )["numcos_l"].transform("sum")
        turnover["numcos_s"] = turnover.groupby(
            [byvar, "date"], sort=False
        )["numcos_s"].transform("sum")
        turnover["numcos"] = turnover[["numcos_l", "numcos_s"]].min(axis=1)
        if self.method == "long_short":
            turnover = turnover[turnover["numcos"] >= self.mincos]
        elif self.method == "long_only":
            turnover = turnover[turnover["numcos_l"] >= self.mincos]

        # calculate turnover from previous trading day
        prev = turnover[["security_id", byvar, "n", "weight"]].copy()
        prev["n"] = prev["n"] + 1
        # variable 'date' is dropped, and merged back later on
        turnover = turnover.drop(columns=["date"]).merge(
            prev,
            how="left",
            on=["security_id", byvar, "n"],
            suffixes=("", "_prev"),
        )
        turnover["weight_prev"] = turnover["weight_prev"].fillna(0)
        turnover["weight_diff"] = (turnover["weight"] - turnover["weight_prev"]).abs()

        current_keys = turnover[["security_id", byvar, "n"]].drop_duplicates()
        exits = prev.merge(
            current_keys, on=["security_id", byvar, "n"], how="left", indicator=True
        )
        exits = exits[exits["_merge"] == "left_only"].drop(columns=["_merge"])
        exits = exits.rename(columns={"weight": "weight_prev"})
        exits["weight"] = 0.0
        exits["weight_diff"] = exits["weight_prev"].abs()
        exits["numcos_l"] = np.nan
        exits["numcos_s"] = np.nan
        exits["numcos"] = np.nan

        turnover = pd.concat([turnover, exits], ignore_index=True, sort=False)
        # Fast n lookup instead of merge
        turnover = self._fast_n_lookup(turnover, 'n', ['date'])
        turnover["n_min"] = turnover.groupby(byvar, sort=False)["n"].transform("min")
        turnover["weight_diff"] = np.where(
            turnover["n"] == turnover["n_min"], 0, turnover["weight_diff"]
        )

        if "cap" not in turnover.columns:
            # Use fast join if master_data available
            if self.master_data is not None:
                turnover = self._fast_join_master(turnover, ["cap"], how='inner')
            else:
                turnover = turnover.merge(
                    self._get_otherfile_cols(["security_id", "date", "cap"]),
                    on=["security_id", "date"],
                )

        if self.tc_model == "naive":
            turnover["tc"] = np.select(
                [turnover["cap"] == 1, turnover["cap"] == 2],
                [self.tc_level["big"] / 10000, self.tc_level["median"] / 10000],
                default=self.tc_level["small"] / 10000,
            )
        elif self.tc_model == "power_law":
            # Use fast join if master_data available
            if self.master_data is not None:
                turnover = self._fast_join_master(turnover, ["vol", "adv", "close_adj"], how='inner')
            else:
                turnover = turnover.merge(
                    self.retfile[["security_id", "date", "vol", "adv", "close_adj"]],
                    on=["security_id", "date"],
                )

            tc_beta, tc_alpha = self.tc_value
            turnover["tc"] = (
                tc_beta
                * turnover["vol"]
                * np.power(
                    self.gmv * 1000000 * turnover["weight_diff"] / turnover["adv"],
                    tc_alpha,
                )
            ) / (turnover["close_adj"])

        turnover["tc"] = turnover["tc"] * turnover["weight_diff"]
        tc = turnover.groupby([byvar, "date"], sort=False)["tc"].sum().reset_index()
        # calculate turnover and transaction cost for each day
        turnover["turnover"] = turnover.groupby([byvar, "n"], sort=False)[
            "weight_diff"
        ].transform("sum")
        turnover = (
            turnover[["date", byvar, "numcos_l", "numcos_s", "turnover", "numcos"]]
            .drop_duplicates()
            .dropna()
        )
        if self.method == "long_short":
            date_minmax = (
                turnover[turnover["numcos"] >= self.mincos]
                .groupby(byvar, sort=False)["date"]
                .agg(["min", "max"])
                .reset_index()
            )
        elif self.method == "long_only":
            date_minmax = (
                turnover[turnover["numcos_l"] >= self.mincos]
                .groupby(byvar, sort=False)["date"]
                .agg(["min", "max"])
                .reset_index()
            )

        panel = (
            self.datefile[["date"]]
            .assign(temp=1)
            .merge(date_minmax.assign(temp=1), on="temp")
        )
        panel = panel.loc[
            (panel["date"] >= panel["min"]) & (panel["date"] <= panel["max"]),
            [byvar, "date"],
        ]
        turnover2 = panel.merge(turnover, how="left", on=[byvar, "date"])
        turnover = turnover2.fillna(0).drop(columns=["numcos"])
        if byvar == "overall":
            turnover_raw = turnover[["date", "turnover"]]

        if self.insample == "i1":
            turnover = self._fast_date_lookup(turnover, 'date', ['insample'])
            turnover = turnover[turnover["insample"] == 1]
        elif self.insample == "i2":
            turnover = self._fast_date_lookup(turnover, 'date', ['insample2'])
            turnover = turnover[turnover["insample2"] == 1]

        turnover = (
            turnover.groupby(byvar, sort=False)
            .mean()
            .reset_index()
            .rename(columns={byvar: "group"})
        )
        if byvar == "overall":
            turnover["group"] = "overall"

        if self.verbose:
            print("cal turnover")

        # calculate portfolio metrics: return, drawdown, factor exposure, industry exposure
        port = weight_file[["security_id", byvar, "date", "ret", "resret", "weight"]]
        port["numcos_l"] = np.where(port["weight"] > 0, 1, 0)
        port["numcos_s"] = np.where(port["weight"] < 0, 1, 0)
        port["numcos_l"] = port.groupby([byvar, "date"], sort=False)[
            "numcos_l"
        ].transform("sum")
        port["numcos_s"] = port.groupby([byvar, "date"], sort=False)[
            "numcos_s"
        ].transform("sum")
        port["numcos"] = port[["numcos_l", "numcos_s"]].min(axis=1)

        # Use fast join if master_data available
        factor_cols = ["size", "value", "growth", "leverage", "volatility", "momentum", "yield"]
        if self.master_data is not None:
            port = self._fast_join_master(port, factor_cols, how='inner')
        else:
            port = port.merge(
                self._get_otherfile_cols(["security_id", "date"] + factor_cols),
                how="inner",
                on=["security_id", "date"],
            )

        # the minimum number of stock on each side should be greater than the minimum threshold

        factor_list = [
            "size",
            "value",
            "growth",
            "leverage",
            "volatility",
            "momentum",
            "yield",
        ]
        var_list = ["ret", "resret"] + factor_list

        if self.method == "long_short":
            for stat in var_list:
                port.loc[port["numcos"] < self.mincos, stat] = 0
        elif self.method == "long_only":
            for stat in var_list:
                port.loc[port["numcos_l"] < self.mincos, stat] = 0

        weight = port["weight"].to_numpy(dtype="float64", copy=False)
        port[var_list] = port[var_list].to_numpy(copy=False) * weight[:, None]

        port2 = (
            port.groupby([byvar, "date"], sort=False)[var_list]
            .sum()
            .reset_index()
        )
        port2 = port2.merge(tc, how="left", on=[byvar, "date"])
        port2["tc"] = port2["tc"].fillna(0)
        port2["ret_net"] = port2["ret"] - port2["tc"]
        port2["resret_net"] = port2["resret"] - port2["tc"]
        port2 = panel.merge(port2, how="left", on=[byvar, "date"])
        port2 = port2.fillna(0)

        if self.insample == "i1":
            port2 = self._fast_date_lookup(port2, 'date', ['insample'])
            port2 = port2[port2["insample"] == 1]
        elif self.insample == "i2":
            port2 = self._fast_date_lookup(port2, 'date', ['insample2'])
            port2 = port2[port2["insample2"] == 1]

        port2["cumret"] = port2.groupby(byvar, sort=False)["ret"].transform("cumsum")
        port2["cumretnet"] = port2.groupby(byvar, sort=False)["ret_net"].transform(
            "cumsum"
        )
        port2["drawdown"] = port2["cumret"] - port2.groupby(
            byvar, sort=False
        )["cumret"].transform("cummax")

        if self.method == "long_only":
            if self.long_index == "sp500":
                long = yf.Ticker("^GSPC")
            elif self.long_index == "russell3000":
                long = yf.Ticker("^RUA")
            elif self.long_index == "russell1000":
                long = yf.Ticker("^RUI")
            elif self.long_index == "russell2000":
                long = yf.Ticker("^RUT")
            else:
                long = yf.Ticker(self.long_index)

            long = long.history(period="max")
            long = long.reset_index()
            long.columns = long.columns.str.lower()
            long = long[["date", "close"]].sort_values(by="date")
            long["date"] = long["date"].dt.tz_localize(None)
            long[self.long_index] = long["close"].pct_change()
            long = long[["date", self.long_index]].dropna()
            port2 = port2.merge(long, how="left", on="date")

        if byvar == "overall":
            if self.method == "long_only":
                daily_stats = port2[
                    [
                        "date",
                        "ret",
                        "ret_net",
                        "resret",
                        "cumret",
                        "cumretnet",
                        "drawdown",
                        self.long_index,
                    ]
                    + factor_list
                ]
                daily_stats[f"cum_{self.long_index}"] = daily_stats[
                    self.long_index
                ].cumsum()
            else:
                daily_stats = port2[
                    [
                        "date",
                        "ret",
                        "ret_net",
                        "resret",
                        "cumret",
                        "cumretnet",
                        "drawdown",
                    ]
                    + factor_list
                ]

        elif byvar == "cap":
            daily_stats = port2[[byvar, "date", "cumret"]]
            daily_stats["cap"] = np.select(
                [daily_stats["cap"] == 1, daily_stats["cap"] == 2],
                ["LargeCap", "MediumCap"],
                default="SmallCap",
            )

            daily_stats = daily_stats.set_index(["date", byvar])
            daily_stats = daily_stats.unstack().add_prefix("cumret_")
            daily_stats.columns = daily_stats.columns.droplevel()
            daily_stats.reset_index(inplace=True)

        # calculate factor exposure
        exposure = (
            port2[[byvar] + factor_list]
            .groupby(byvar, sort=False)
            .mean()
            .reset_index()
            .rename(columns={byvar: "group"})
        )
        if byvar == "overall":
            exposure["group"] = "overall"

        if self.verbose:
            print("cal exposure")

        # calculate return, sharpe and maximum drawdown
        if self.method == "long_only":
            port2 = port2[
                [
                    byvar,
                    "ret",
                    "resret",
                    "ret_net",
                    "resret_net",
                    "drawdown",
                    self.long_index,
                ]
            ]
        else:
            port2 = port2[[byvar, "ret", "resret", "ret_net", "resret_net", "drawdown"]]

        if "year" in byvar or "yr" in byvar:

            port2["trade"] = np.where(port2["ret"] == 0, 0, 1)
            group = port2.groupby(byvar, sort=False)
            port2["num_date"] = group["trade"].transform("sum")
            ret_count = group["ret"].transform("count")
            ret_mean = group["ret"].transform("mean")
            ret_std = group["ret"].transform("std")
            port2["ret_ann"] = ret_mean * ret_count
            port2["ret_std"] = ret_std * np.sqrt(ret_count)
            port2["sharpe_ret"] = port2["ret_ann"] / port2["ret_std"]
            resret_mean = group["resret"].transform("mean")
            resret_std = group["resret"].transform("std")
            port2["resret_ann"] = resret_mean * ret_count
            port2["resret_std"] = resret_std * np.sqrt(ret_count)
            port2["sharpe_resret"] = port2["resret_ann"] / port2["resret_std"]
            ret_net_mean = group["ret_net"].transform("mean")
            ret_net_std = group["ret_net"].transform("std")
            port2["ret_net_ann"] = ret_net_mean * ret_count
            port2["ret_net_std"] = ret_net_std * np.sqrt(ret_count)
            port2["sharpe_retnet"] = port2["ret_net_ann"] / port2["ret_net_std"]
            port2["maxdraw"] = group["drawdown"].transform("min")
            ret_pct = (np.sign(port2["ret"]) + 1) / 2
            resret_pct = (np.sign(port2["resret"]) + 1) / 2
            retnet_pct = (np.sign(port2["ret_net"]) + 1) / 2
            port2["retPctPos"] = ret_pct.groupby(port2[byvar], sort=False).transform(
                "mean"
            )
            port2["resretPctPos"] = resret_pct.groupby(
                port2[byvar], sort=False
            ).transform("mean")
            port2["retnetPctPos"] = retnet_pct.groupby(
                port2[byvar], sort=False
            ).transform("mean")
            stats_list_f = [
                "ret_ann",
                "ret_std",
                "sharpe_ret",
                "retPctPos",
                "resret_ann",
                "resret_std",
                "sharpe_resret",
                "resretPctPos",
                "ret_net_ann",
                "ret_net_std",
                "sharpe_retnet",
                "retnetPctPos",
                "maxdraw",
            ]
            stats_list_s = [
                "ret_ann",
                "ret_std",
                "sharpe_ret",
                "ret_net_ann",
                "ret_net_std",
                "sharpe_retnet",
                "maxdraw",
            ]

            if self.method == "long_only":

                long_count = group[self.long_index].transform("count")
                long_mean = group[self.long_index].transform("mean")
                long_std = group[self.long_index].transform("std")
                port2[f"{self.long_index}_ann"] = long_mean * long_count
                port2[f"{self.long_index}_std"] = long_std * np.sqrt(long_count)
                port2[f"sharpe_{self.long_index}"] = (
                    port2[f"{self.long_index}_ann"] / port2[f"{self.long_index}_std"]
                )
                stats_list_f = stats_list_f + [
                    f"{self.long_index}_ann",
                    f"{self.long_index}_std",
                    f"sharpe_{self.long_index}",
                ]
                stats_list_s = stats_list_s + [
                    f"{self.long_index}_ann",
                    f"{self.long_index}_std",
                    f"sharpe_{self.long_index}",
                ]

        else:

            port2["trade"] = np.where(port2["ret"] == 0, 0, 1)
            group = port2.groupby(byvar, sort=False)
            port2["num_date"] = group["trade"].transform("sum")
            # port2 = port2[port2['trade']==1]
            ret_mean = group["ret"].transform("mean")
            ret_std = group["ret"].transform("std")
            port2["ret_ann"] = ret_mean * 252
            port2["ret_std"] = ret_std * math.sqrt(252)
            port2["sharpe_ret"] = port2["ret_ann"] / port2["ret_std"]
            resret_mean = group["resret"].transform("mean")
            resret_std = group["resret"].transform("std")
            port2["resret_ann"] = resret_mean * 252
            port2["resret_std"] = resret_std * math.sqrt(252)
            port2["sharpe_resret"] = port2["resret_ann"] / port2["resret_std"]
            ret_net_mean = group["ret_net"].transform("mean")
            ret_net_std = group["ret_net"].transform("std")
            port2["ret_net_ann"] = ret_net_mean * 252
            port2["ret_net_std"] = ret_net_std * math.sqrt(252)
            port2["sharpe_retnet"] = port2["ret_net_ann"] / port2["ret_net_std"]
            port2["maxdraw"] = group["drawdown"].transform("min")
            ret_pct = (np.sign(port2["ret"]) + 1) / 2
            resret_pct = (np.sign(port2["resret"]) + 1) / 2
            retnet_pct = (np.sign(port2["ret_net"]) + 1) / 2
            port2["retPctPos"] = ret_pct.groupby(port2[byvar], sort=False).transform(
                "mean"
            )
            port2["resretPctPos"] = resret_pct.groupby(
                port2[byvar], sort=False
            ).transform("mean")
            port2["retnetPctPos"] = retnet_pct.groupby(
                port2[byvar], sort=False
            ).transform("mean")
            stats_list_f = [
                "ret_ann",
                "ret_std",
                "sharpe_ret",
                "retPctPos",
                "resret_ann",
                "resret_std",
                "sharpe_resret",
                "resretPctPos",
                "ret_net_ann",
                "ret_net_std",
                "sharpe_retnet",
                "retnetPctPos",
                "maxdraw",
            ]
            stats_list_s = [
                "ret_ann",
                "ret_std",
                "sharpe_ret",
                "ret_net_ann",
                "ret_net_std",
                "sharpe_retnet",
                "maxdraw",
            ]

            if self.method == "long_only":

                long_mean = group[self.long_index].transform("mean")
                long_std = group[self.long_index].transform("std")
                port2[f"{self.long_index}_ann"] = long_mean * 252
                port2[f"{self.long_index}_std"] = long_std * math.sqrt(252)
                port2[f"sharpe_{self.long_index}"] = (
                    port2[f"{self.long_index}_ann"] / port2[f"{self.long_index}_std"]
                )
                stats_list_f = stats_list_f + [
                    f"{self.long_index}_ann",
                    f"{self.long_index}_std",
                    f"sharpe_{self.long_index}",
                ]
                stats_list_s = stats_list_s + [
                    f"{self.long_index}_ann",
                    f"{self.long_index}_std",
                    f"sharpe_{self.long_index}",
                ]

        annret = (
            port2.loc[:, [byvar, "num_date"] + stats_list_f]
            .drop_duplicates()
            .rename(columns={byvar: "group"})
        )
        if byvar == "overall":
            annret["group"] = "overall"

        if self.verbose:
            print("cal return")

        combo = annret.merge(turnover, on="group")
        combo = combo.merge(exposure, on="group")
        # combo = combo.merge(cor, on='group')
        if self.output == "full":
            combo = combo[
                ["group", "numcos_l", "numcos_s", "num_date"]
                + stats_list_f
                + ["turnover"]
                + factor_list
            ]
        elif self.output == "simple":
            combo = combo[
                ["group", "numcos_l", "numcos_s", "num_date"]
                + stats_list_s
                + ["turnover"]
                + factor_list
            ]

        if byvar == "cap":
            combo["group"] = np.select(
                [combo["group"] == 1, combo["group"] == 2],
                ["Large Cap", "Medium Cap"],
                default="Small Cap",
            )
        if byvar == "cap":
            return combo.sort_values(by="group"), daily_stats

        elif byvar == "overall":
            return combo.sort_values(by="group"), daily_stats, turnover_raw

        else:
            return combo.sort_values(by="group")

    def ff_grab(self):
        tables = [
            "F-F_Research_Data_5_Factors_2x3_daily",
            "F-F_Momentum_Factor_daily",
            "F-F_ST_Reversal_Factor_daily",
            "12_Industry_Portfolios_daily",
        ]
        factors = []
        for t in tables:
            fds = pdr.famafrench.FamaFrenchReader(t, start="01-01-1926")
            temp = fds.read()[0].reset_index()
            temp.columns = temp.columns.str.lower()
            temp.columns = temp.columns.str.strip()
            factors.append(temp)
        factors = reduce(
            lambda df_l, df_r: pd.merge(df_l, df_r, on="date", how="inner"), factors
        )
        return factors

    def gen_result(self):
        result = []
        temp_base = self.pre_process()
        if self.input_type in ["value", "fractile"]:
            fractile = self.gen_fractile(temp_base, np.ceil(100 / self.fractile[0]))
        else:
            fractile = None

        byvar_cols = []
        for byvar in self.byvar_list:
            if byvar not in ["overall", "year", "capyr"]:
                byvar_cols.append(byvar)
        if "capyr" in self.byvar_list and "cap" not in byvar_cols:
            byvar_cols.append("cap")
        byvar_cols = list(dict.fromkeys(byvar_cols))

        if byvar_cols:
            temp_with_byvars = pd.merge_asof(
                temp_base.sort_values(by=["date_sig"]),
                self._get_otherfile_cols(["security_id", "date"] + byvar_cols).rename(
                    columns={"date": "date_sig"}
                ).sort_values(by="date_sig"),
                by="security_id",
                on="date_sig",
                allow_exact_matches=True,
                direction="backward",
                tolerance=pd.Timedelta("20d"),
            )
        else:
            temp_with_byvars = temp_base

        if "capyr" in self.byvar_list and "cap" in temp_with_byvars.columns:
            cap = temp_with_byvars["cap"]
            cap_label = np.select(
                [cap == 1, cap == 2], ["LargeCap", "MediumCap"], default="SmallCap"
            )
            temp_with_byvars["capyr"] = np.where(
                cap.notna(), cap_label + "_" + temp_with_byvars["year"].astype("str"), np.nan
            )

        daily_stats, daily_stats2, turnover = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        temp_iter = temp_with_byvars
        for byvar in self.byvar_list:

            if self.verbose:
                print(f"Processing byvar: {byvar}")

            if byvar not in ["overall", "year"]:
                temp_iter = temp_iter[temp_iter[byvar].notna()]
            temp = temp_iter

            if byvar == "overall":
                combo, daily_stats, turnover = self.backtest(temp, byvar)

            elif byvar == "cap":
                combo, daily_stats2 = self.backtest(temp, byvar)
            else:
                combo = self.backtest(temp, byvar)
            result.append(combo)

        if self.earnings_window:
            temp2 = temp_iter.merge(self.window_file, on=["security_id", "date"])
            combo = self.backtest(temp2, "earning_window")
            result.append(combo)

        if self.byvix:
            try:
                vix = yf.Ticker("^VIX")
                vix = vix.history(period="max")
                vix = vix.reset_index()
                vix.columns = vix.columns.str.lower()
                vix = vix[["date", "close"]].rename(columns={"close": "vix"})
                vix = vix[
                    (vix["date"] >= daily_stats["date"].min())
                    & (vix["date"] <= daily_stats["date"].max())
                ]
                vix_median = np.round(vix["vix"].median(), 1)
                vix["vix"] = np.where(
                    vix["vix"] > vix_median,
                    f"vix > {vix_median}",
                    f"vix <= {vix_median}",
                )
                temp2 = temp_iter.merge(vix, on=["date"])
                combo = self.backtest(temp2, "vix")
                result.append(combo)
            except:
                if self.verbose:
                    print("Cannot get Vix data from yfinance")

        result = pd.concat(result)
        result["turnover"] = result["turnover"] / 4

        if self.beta:

            if self.benchmark == "ff":
                ff = self.ff_grab()
                ff.iloc[:, 1:] = ff.iloc[:, 1:] / 100
                sample = daily_stats[["date", "ret"]].merge(ff, on="date")
                sample = sample.rename(columns={"mkt-rf": "market_ret"})

            else:
                if self.benchmark == "sp500":
                    sample = yf.Ticker("^GSPC")
                elif self.benchmark == "russell3000":
                    sample = yf.Ticker("^RUA")
                elif self.benchmark == "russell1000":
                    sample = yf.Ticker("^RUI")

                sample = sample.history(period="max")
                sample = sample.reset_index()
                sample.columns = sample.columns.str.lower()
                sample = sample[["date", "close"]].sort_values(by="date")
                sample["market_ret"] = sample["close"].pct_change()
                sample = sample[["date", "market_ret"]].dropna()
                sample = daily_stats[["date", "ret"]].merge(sample, on="date")

            def get_beta(sample):

                xvar = sm.add_constant(sample["market_ret"])
                res = sm.OLS(sample["ret"], xvar).fit()
                res = pd.read_html(
                    res.summary().tables[1].as_html(), header=0, index_col=0
                )[0]
                # Extract the key result, which is the coefficient and p-value
                res = pd.DataFrame(
                    columns=["alpha", "beta", "alpha_pvalue", "beta_pvalue"],
                    data=[
                        [
                            res.loc["const", "coef"] * 252,
                            res.loc["market_ret", "coef"],
                            res.loc["const", "P>|t|"],
                            res.loc["market_ret", "P>|t|"],
                        ]
                    ],
                )

                return res

            sample["year"] = sample["date"].dt.year
            beta = get_beta(sample)
            beta.insert(0, "group", "overall")
            beta = pd.concat(
                [
                    beta,
                    sample.groupby("year")
                    .apply(get_beta)
                    .reset_index(level=0)
                    .rename(columns={"year": "group"}),
                ]
            )
            result = result.merge(beta, how="left", on="group")
        if self.ff_result:
            ff = self.ff_grab()
            ff.iloc[:, 1:] = ff.iloc[:, 1:] / 100
            sample = daily_stats[["date", "ret"]].merge(ff, on="date")
            sample = sample.drop(columns=["rf"]).rename(
                columns={"mkt-rf": "market_ret"}
            )
            sample["ret"] = sample["ret"] * 252

            if self.ff_model == "capm":
                xvar = sm.add_constant(sample.iloc[:, 2])
            if self.ff_model == "ff3":
                xvar = sm.add_constant(sample.iloc[:, 2:5])
            if self.ff_model == "ff5":
                xvar = sm.add_constant(sample.iloc[:, 2:7])
            if self.ff_model == "ff7":
                xvar = sm.add_constant(sample.iloc[:, 2:9])
            if self.ff_model == "ff7_industry":
                xvar = sm.add_constant(sample.iloc[:, 3:])
            xvar.iloc[:, 1:] = xvar.iloc[:, 1:] * 252

            res = sm.OLS(sample["ret"], xvar).fit()
            ff_result = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0]
            ff_result.loc["const", "coef"] = ff_result.loc["const", "coef"]
            ff_result.loc["const", "std err"] = np.round(
                ff_result.loc["const", "coef"] / ff_result.loc["const", "t"], 3
            )
            xvar.iloc[:, 0] = 0
            daily_stats["ff_alpha"] = (sample["ret"] - res.predict(xvar)) / 252
            daily_stats["cum_alpha"] = daily_stats["ff_alpha"].cumsum()

        if "cap" in self.byvar_list:
            daily_stats = pd.merge(daily_stats, daily_stats2, on="date")

        if "overall" in self.byvar_list:
            daily_stats = daily_stats.merge(turnover, on="date")

        if self.ff_result:
            return result, daily_stats, fractile, ff_result
        else:
            return result, daily_stats, fractile
