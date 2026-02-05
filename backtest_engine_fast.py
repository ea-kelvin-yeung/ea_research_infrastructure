# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:47:34 2020

@author: yunan
"""


import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import math
import yfinance as yf
import pandas_datareader as pdr
from functools import reduce

pd.options.mode.chained_assignment = None


# %%
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
    ):

        self.verbose = verbose
        self.infile = infile
        self.retfile = retfile
        self.otherfile = otherfile
        self.datefile = datefile
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
            else:

                temp = pd.merge_asof(
                    temp.sort_values(by=["date_sig"]),
                    self.otherfile[
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
                    ]
                    .rename(columns={"date": "date_sig", "yield": "yields"})
                    .sort_values(by="date_sig"),
                    by="security_id",
                    on="date_sig",
                    allow_exact_matches=True,
                    direction="backward",
                    tolerance=pd.Timedelta("5d"),
                ).dropna()
                temp = temp.sort_values(by=["date", "security_id"])
                temp = temp.reset_index(drop=True)

                resid_varlist = " + ".join(map(str, self.resid_varlist))

                if self.resid_style == "all":

                    def model(df):
                        return (
                            ols(
                                formula=f"{self.sigvar} ~ {resid_varlist} + C(industry_id)",
                                data=df,
                            )
                            .fit()
                            .resid
                        )

                elif self.resid_style == "industry":

                    def model(df):
                        return (
                            ols(formula=f"{self.sigvar} ~ C(industry_id)", data=df)
                            .fit()
                            .resid
                        )

                elif self.resid_style == "factor":

                    def model(df):
                        return (
                            ols(formula=f"{self.sigvar} ~ {resid_varlist}", data=df)
                            .fit()
                            .resid
                        )

                fit = temp.groupby("date").apply(model)
                fit = fit.reset_index().drop(columns=["date"]).set_index("level_1")
                temp = temp.join(fit.rename(columns={0: f"{self.sigvar}_resid"}))
                temp = temp[
                    [
                        "security_id",
                        "date_sig",
                        "date",
                        "ret",
                        "resret",
                        f"{self.sigvar}_resid",
                    ]
                ].rename(columns={f"{self.sigvar}_resid": self.sigvar})

        temp["overall"] = 1
        temp["year"] = temp["date"].dt.year
        return temp

    def cal_corr(self, infile, byvar):

        cor_file = infile.merge(
            self.otherfile[["security_id", "date"] + self.factor_list],
            on=["security_id", "date"],
        )

        if self.insample == "i1":
            cor_file = cor_file.merge(self.datefile[["date", "insample"]], on="date")
            cor_file = cor_file[cor_file["insample"] == 1]
        elif self.insample == "i2":
            cor_file = cor_file.merge(self.datefile[["date", "insample2"]], on="date")
            cor_file = cor_file[cor_file["insample2"] == 1]

        cor = cor_file.groupby(["date", byvar])[[self.sigvar, "ret", "resret"]].corr(
            method="spearman"
        )
        cor = cor[cor[self.sigvar] == 1]
        cor["date"] = [x[0] for x in cor.index]
        cor[byvar] = [x[1] for x in cor.index]
        cor["_name_"] = [x[2] for x in cor.index]
        cor = cor[["date", byvar, "ret", "resret"]].reset_index(drop=True)

        corn = cor_file[["date", byvar]].copy()
        corn["n"] = corn.groupby(["date", byvar])["date"].transform("size")
        corn = corn.drop_duplicates().reset_index(drop=True)
        cor = pd.merge(cor, corn, on=["date", byvar])
        cor = cor[
            cor["n"]
            >= self.mincos * 100 / (min(self.fractile[0], 1 - self.fractile[1]))
        ]
        cor = cor.drop("n", axis=1, inplace=False)
        cor = cor.drop(columns=["date"])

        cor["retIC"] = cor.groupby(byvar)["ret"].transform("mean")
        cor["resretIC"] = cor.groupby(byvar)["resret"].transform("mean")

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
                    self.double_file.rename(
                        columns={"date": "date_doublevar"}
                    ).sort_values(by="date_doublevar"),
                    by="security_id",
                    left_on="date_sig",
                    right_on="date_doublevar",
                    allow_exact_matches=True,
                    direction="backward",
                )

                port = port.dropna()

                port["rank"] = port.groupby(["date"])[self.double_var].transform(
                    lambda x: x.rank(method="average")
                )
                port["group_size"] = port.groupby(["date"])[self.double_var].transform(
                    "count"
                )
                port["fractile_double"] = np.ceil(
                    port.pop("rank") * self.double_frac / port.pop("group_size")
                )

                port["group_size"] = port.groupby(["fractile_double", "date"])[
                    self.sigvar
                ].transform("count")
                if self.frac_stretch:
                    port["rank"] = port.groupby(["fractile_double", "date"])[
                        self.sigvar
                    ].transform(lambda x: x.rank(method="max"))
                    port["percentile"] = np.ceil(
                        port.pop("rank") * 100 / port.pop("group_size")
                    )
                    port["percentile"] = port.groupby(["fractile_double", "date"])[
                        "percentile"
                    ].transform(
                        lambda x: (x - x.min() + 1) * 100 / (x.max() - x.min() + 1)
                    )
                    port["fractile"] = np.ceil(
                        port.pop("percentile") * n_fractile / 100
                    )
                else:
                    port["rank"] = port.groupby(["fractile_double", "date"])[
                        self.sigvar
                    ].transform(lambda x: x.rank(method="average"))
                    port["fractile"] = np.ceil(
                        port.pop("rank") * n_fractile / port.pop("group_size")
                    )

            # single sorting
            elif self.sort_method == "single":

                port["group_size"] = port.groupby("date")[self.sigvar].transform(
                    "count"
                )

                if self.frac_stretch:
                    port["rank"] = port.groupby("date")[self.sigvar].transform(
                        lambda x: x.rank(method="max")
                    )
                    port["percentile"] = np.ceil(
                        port.pop("rank") * 100 / port.pop("group_size")
                    )
                    port["percentile"] = port.groupby(["date"])["percentile"].transform(
                        lambda x: (x - x.min() + 1) * 100 / (x.max() - x.min() + 1)
                    )
                    port["fractile"] = np.ceil(
                        port.pop("percentile") * n_fractile / 100
                    )
                else:
                    port["rank"] = port.groupby("date")[self.sigvar].transform(
                        lambda x: x.rank(method="average")
                    )
                    port["fractile"] = np.ceil(
                        port.pop("rank") * n_fractile / port.pop("group_size")
                    )

        elif self.input_type == "fractile":
            port["fractile"] = port[self.sigvar]

        port = port[["security_id", "date", "fractile", "ret", "resret"]]
        port = port.merge(
            self.otherfile[["security_id", "date", "adv", "mcap"] + self.factor_list],
            on=["security_id", "date"],
        )

        if self.weight == "equal":
            port["weight"] = port.groupby(["date", "fractile"])[
                "security_id"
            ].transform(lambda x: 1 / x.count())
        elif self.weight == "value":
            port["mcap_h"] = port.groupby(["date", "fractile"])["mcap"].transform(
                lambda x: x.quantile(q=self.upper_pct / 100)
            )
            port["mcap_l"] = port.groupby(["date", "fractile"])["mcap"].transform(
                lambda x: x.quantile(q=1 - self.upper_pct / 100)
            )
            port["mcap"] = np.select(
                [port["mcap"] > port["mcap_h"], port["mcap"] < port["mcap_l"]],
                [port["mcap_h"], port["mcap_l"]],
                default=port["mcap"],
            )
            port["weight"] = port.groupby(["date", "fractile"])["mcap"].transform(
                lambda x: x / x.sum() * 1
            )
        elif self.weight == "volume":
            port["adv_h"] = port.groupby(["date", "fractile"])["adv"].transform(
                lambda x: x.quantile(q=self.upper_pct / 100)
            )
            port["adv_l"] = port.groupby(["date", "fractile"])["adv"].transform(
                lambda x: x.quantile(q=1 - self.upper_pct / 100)
            )
            port["adv"] = np.select(
                [port["adv"] > port["adv_h"], port["adv"] < port["adv_l"]],
                [port["adv_h"], port["adv_l"]],
                default=port["adv"],
            )
            port["weight"] = port.groupby(["date", "fractile"])["adv"].transform(
                lambda x: x / x.sum() * 1
            )
        port2 = port[
            ["security_id", "date", "ret", "resret", "fractile", "weight"]
            + self.factor_list
        ]

        numcos = (
            port2.groupby(["date", "fractile"])["security_id"].count().reset_index()
        )
        numcos = numcos[numcos["fractile"].isin([1, n_fractile])]
        numcos["numcos"] = numcos.groupby("date")["security_id"].transform("min")
        numcos = numcos[["date", "numcos"]].drop_duplicates()
        port2 = port2.merge(numcos, on="date")
        port2 = port2[port2["numcos"] >= self.mincos]

        for var in [
            "ret",
            "resret",
            "size",
            "value",
            "growth",
            "leverage",
            "volatility",
            "momentum",
            "yield",
        ]:

            port2[var] = port2[var] * port2["weight"]

        check = port2.groupby(["date", "fractile"]).sum().reset_index()
        check2 = (
            port2.groupby(["date", "fractile"])["security_id"]
            .count()
            .reset_index()
            .rename(columns={"security_id": "numcos"})
            .groupby("fractile")
            .mean()
            .reset_index()
        )
        check3 = check.groupby(["fractile"])[["ret", "resret"]].mean() * 252
        check = check.groupby("fractile")[
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
                    self.double_file.rename(
                        columns={"date": "date_doublevar"}
                    ).sort_values(by="date_doublevar"),
                    by="security_id",
                    left_on="date_sig",
                    right_on="date_doublevar",
                    allow_exact_matches=True,
                    direction="backward",
                )

                temp = temp.dropna()

                temp["rank"] = temp.groupby([byvar, "date"])[self.double_var].transform(
                    lambda x: x.rank(method="average")
                )
                temp["group_size"] = temp.groupby([byvar, "date"])[
                    self.double_var
                ].transform("count")
                temp["fractile_double"] = np.ceil(
                    temp.pop("rank") * self.double_frac / temp.pop("group_size")
                )

                temp["group_size"] = temp.groupby([byvar, "fractile_double", "date"])[
                    self.sigvar
                ].transform("count")
                if self.frac_stretch:
                    temp["rank"] = temp.groupby([byvar, "fractile_double", "date"])[
                        self.sigvar
                    ].transform(lambda x: x.rank(method="max"))
                    temp["percentile"] = np.ceil(
                        temp.pop("rank") * 100 / temp.pop("group_size")
                    )
                    temp["percentile"] = temp.groupby(
                        [byvar, "fractile_double", "date"]
                    )["percentile"].transform(
                        lambda x: (x - x.min() + 1) * 100 / (x.max() - x.min() + 1)
                    )
                else:
                    temp["rank"] = temp.groupby([byvar, "fractile_double", "date"])[
                        self.sigvar
                    ].transform(lambda x: x.rank(method="average"))
                    temp["percentile"] = np.ceil(
                        temp.pop("rank") * 100 / temp.pop("group_size")
                    )

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
                temp["group_size"] = temp.groupby([byvar, "date"])[
                    self.sigvar
                ].transform("count")
                if self.frac_stretch:
                    temp["rank"] = temp.groupby([byvar, "date"])[self.sigvar].transform(
                        lambda x: x.rank(method="max")
                    )
                    temp["percentile"] = np.ceil(
                        temp.pop("rank") * 100 / temp.pop("group_size")
                    )
                    temp["percentile"] = temp.groupby([byvar, "date"])[
                        "percentile"
                    ].transform(
                        lambda x: (x - x.min() + 1) * 100 / (x.max() - x.min() + 1)
                    )
                else:
                    temp["rank"] = temp.groupby([byvar, "date"])[self.sigvar].transform(
                        lambda x: x.rank(method="average")
                    )
                    temp["percentile"] = np.ceil(
                        temp.pop("rank") * 100 / temp.pop("group_size")
                    )

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
        temp = temp.merge(
            self.otherfile[["security_id", "date", "adv", "mcap"]],
            on=["security_id", "date"],
        )

        return temp

    def gen_weight_ls(self, port, byvar):

        if self.input_type != "weight":
            # generate weight
            if self.weight == "equal":
                port["weight"] = (
                    port.groupby([byvar, "date", "position"])["security_id"].transform(
                        lambda x: 1 / x.count()
                    )
                    * port["position"]
                )
            elif self.weight == "value":
                port["mcap_h"] = port.groupby([byvar, "date", "position"])[
                    "mcap"
                ].transform(lambda x: x.quantile(q=self.upper_pct / 100))
                port["mcap_l"] = port.groupby([byvar, "date", "position"])[
                    "mcap"
                ].transform(lambda x: x.quantile(q=1 - self.upper_pct / 100))
                port["mcap"] = np.select(
                    [port["mcap"] > port["mcap_h"], port["mcap"] < port["mcap_l"]],
                    [port["mcap_h"], port["mcap_l"]],
                    default=port["mcap"],
                )
                port["weight"] = (
                    port.groupby([byvar, "date", "position"])["mcap"].transform(
                        lambda x: x / x.sum() * 1
                    )
                    * port["position"]
                )
            elif self.weight == "volume":
                port["adv_h"] = port.groupby([byvar, "date", "position"])[
                    "adv"
                ].transform(lambda x: x.quantile(q=self.upper_pct / 100))
                port["adv_l"] = port.groupby([byvar, "date", "position"])[
                    "adv"
                ].transform(lambda x: x.quantile(q=1 - self.upper_pct / 100))
                port["adv"] = np.select(
                    [port["adv"] > port["adv_h"], port["adv"] < port["adv_l"]],
                    [port["adv_h"], port["adv_l"]],
                    default=port["adv"],
                )
                port["weight"] = (
                    port.groupby([byvar, "date", "position"])["adv"].transform(
                        lambda x: x / x.sum() * 1
                    )
                    * port["position"]
                )

        elif self.input_type == "weight":
            port["weight"] = port[self.sigvar]
            if "position" not in port.columns.to_list():
                port["position"] = np.where(port["weight"] > 0, 1, -1)
            if self.weight_adj == True:
                port["weight"] = (
                    port.groupby([byvar, "date", "position"])["weight"].transform(
                        lambda x: x / x.sum()
                    )
                    * port["position"]
                )
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
        turnover = turnover.merge(self.datefile[["date", "n"]], on="date")
        # number of stocks with non-zero position
        turnover["numcos_l"] = np.where(turnover["weight"] > 0, 1, 0)
        turnover["numcos_s"] = np.where(turnover["weight"] < 0, 1, 0)
        turnover["numcos_l"] = turnover.groupby([byvar, "date"])["numcos_l"].transform(
            "sum"
        )
        turnover["numcos_s"] = turnover.groupby([byvar, "date"])["numcos_s"].transform(
            "sum"
        )
        turnover["numcos"] = turnover[["numcos_l", "numcos_s"]].min(axis=1)
        if self.method == "long_short":
            turnover = turnover[turnover["numcos"] >= self.mincos]
        elif self.method == "long_only":
            turnover = turnover[turnover["numcos_l"] >= self.mincos]

        # calculate turnover from previous trading day
        turnover2 = turnover.copy()
        turnover2["n"] = turnover2["n"] + 1
        # variable 'date' is dropped, and merged back later on, because of the outer join
        turnover = turnover.drop(columns=["date"]).merge(
            turnover2[["security_id", byvar, "n", "weight"]],
            how="outer",
            on=["security_id", byvar, "n"],
        )
        turnover["weight_x"] = turnover["weight_x"].fillna(0)
        turnover["weight_y"] = turnover["weight_y"].fillna(0)
        turnover["weight_diff"] = np.absolute(
            turnover["weight_x"] - turnover["weight_y"]
        )
        turnover = turnover.merge(self.datefile[["date", "n"]], on="n")
        turnover["n_min"] = turnover.groupby(byvar)["n"].transform("min")
        turnover["weight_diff"] = np.where(
            turnover["n"] == turnover["n_min"], 0, turnover["weight_diff"]
        )

        if "cap" not in turnover.columns:
            turnover = turnover.merge(
                self.otherfile[["security_id", "date", "cap"]],
                on=["security_id", "date"],
            )

        if self.tc_model == "naive":
            turnover["tc"] = np.select(
                [turnover["cap"] == 1, turnover["cap"] == 2],
                [self.tc_level["big"] / 10000, self.tc_level["median"] / 10000],
                default=self.tc_level["small"] / 10000,
            )
        elif self.tc_model == "power_law":
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
        tc = turnover.groupby([byvar, "date"])["tc"].sum().reset_index()
        # calculate turnover and transaction cost for each day
        turnover["turnover"] = turnover.groupby([byvar, "n"])["weight_diff"].transform(
            "sum"
        )
        turnover = (
            turnover[["date", byvar, "numcos_l", "numcos_s", "turnover", "numcos"]]
            .drop_duplicates()
            .dropna()
        )
        if self.method == "long_short":
            date_minmax = (
                turnover[turnover["numcos"] >= self.mincos]
                .groupby(byvar)["date"]
                .agg(["min", "max"])
                .reset_index()
            )
        elif self.method == "long_only":
            date_minmax = (
                turnover[turnover["numcos_l"] >= self.mincos]
                .groupby(byvar)["date"]
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
            turnover = turnover.merge(self.datefile[["date", "insample"]], on="date")
            turnover = turnover[turnover["insample"] == 1]
        elif self.insample == "i2":
            turnover = turnover.merge(self.datefile[["date", "insample2"]], on="date")
            turnover = turnover[turnover["insample2"] == 1]

        turnover = (
            turnover.groupby(byvar)
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
        port["numcos_l"] = port.groupby([byvar, "date"])["numcos_l"].transform("sum")
        port["numcos_s"] = port.groupby([byvar, "date"])["numcos_s"].transform("sum")
        port["numcos"] = port[["numcos_l", "numcos_s"]].min(axis=1)

        port = port.merge(
            self.otherfile[
                [
                    "security_id",
                    "date",
                    "size",
                    "value",
                    "growth",
                    "leverage",
                    "volatility",
                    "momentum",
                    "yield",
                ]
            ],
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

        for stat in var_list:
            port[stat] = port[stat] * port["weight"]

        port2 = port.groupby([byvar, "date"])[var_list].sum().reset_index()
        port2 = port2.merge(tc, how="left", on=[byvar, "date"])
        port2["tc"] = port2["tc"].fillna(0)
        port2["ret_net"] = port2["ret"] - port2["tc"]
        port2["resret_net"] = port2["resret"] - port2["tc"]
        port2 = panel.merge(port2, how="left", on=[byvar, "date"])
        port2 = port2.fillna(0)

        if self.insample == "i1":
            port2 = port2.merge(self.datefile[["date", "insample"]], on="date")
            port2 = port2[port2["insample"] == 1]
        elif self.insample == "i2":
            port2 = port2.merge(self.datefile[["date", "insample2"]], on="date")
            port2 = port2[port2["insample2"] == 1]

        port2["cumret"] = port2.groupby(byvar)["ret"].transform("cumsum")
        port2["cumretnet"] = port2.groupby(byvar)["ret_net"].transform("cumsum")
        port2["drawdown"] = port2["cumret"] - port2.groupby(byvar)["cumret"].transform(
            "cummax"
        )

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
            .groupby(byvar)
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
            port2["num_date"] = port2.groupby(byvar)["trade"].transform("sum")
            port2["ret_ann"] = port2.groupby(byvar)["ret"].transform(
                lambda x: x.mean() * x.count()
            )
            port2["ret_std"] = port2.groupby(byvar)["ret"].transform(
                lambda x: x.std() * math.sqrt(x.count())
            )
            port2["sharpe_ret"] = port2["ret_ann"] / port2["ret_std"]
            port2["resret_ann"] = port2.groupby(byvar)["resret"].transform(
                lambda x: x.mean() * x.count()
            )
            port2["resret_std"] = port2.groupby(byvar)["resret"].transform(
                lambda x: x.std() * math.sqrt(x.count())
            )
            port2["sharpe_resret"] = port2["resret_ann"] / port2["resret_std"]
            port2["ret_net_ann"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: x.mean() * x.count()
            )
            port2["ret_net_std"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: x.std() * math.sqrt(x.count())
            )
            port2["sharpe_retnet"] = port2["ret_net_ann"] / port2["ret_net_std"]
            port2["maxdraw"] = port2.groupby(byvar)["drawdown"].transform("min")
            port2["retPctPos"] = port2.groupby(byvar)["ret"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
            port2["resretPctPos"] = port2.groupby(byvar)["resret"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
            port2["retnetPctPos"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
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

                port2[f"{self.long_index}_ann"] = port2.groupby(byvar)[
                    self.long_index
                ].transform(lambda x: x.mean() * x.count())
                port2[f"{self.long_index}_std"] = port2.groupby(byvar)[
                    self.long_index
                ].transform(lambda x: x.std() * math.sqrt(x.count()))
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
            port2["num_date"] = port2.groupby(byvar)["trade"].transform("sum")
            # port2 = port2[port2['trade']==1]
            port2["ret_ann"] = port2.groupby(byvar)["ret"].transform(
                lambda x: x.mean() * 252
            )
            port2["ret_std"] = port2.groupby(byvar)["ret"].transform(
                lambda x: x.std() * math.sqrt(252)
            )
            port2["sharpe_ret"] = port2["ret_ann"] / port2["ret_std"]
            port2["resret_ann"] = port2.groupby(byvar)["resret"].transform(
                lambda x: x.mean() * 252
            )
            port2["resret_std"] = port2.groupby(byvar)["resret"].transform(
                lambda x: x.std() * math.sqrt(252)
            )
            port2["sharpe_resret"] = port2["resret_ann"] / port2["resret_std"]
            port2["ret_net_ann"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: x.mean() * 252
            )
            port2["ret_net_std"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: x.std() * math.sqrt(252)
            )
            port2["sharpe_retnet"] = port2["ret_net_ann"] / port2["ret_net_std"]
            port2["maxdraw"] = port2.groupby(byvar)["drawdown"].transform("min")
            port2["retPctPos"] = port2.groupby(byvar)["ret"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
            port2["resretPctPos"] = port2.groupby(byvar)["resret"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
            port2["retnetPctPos"] = port2.groupby(byvar)["ret_net"].transform(
                lambda x: np.mean(np.sign(x) + 1) / 2
            )
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

                port2[f"{self.long_index}_ann"] = port2.groupby(byvar)[
                    self.long_index
                ].transform(lambda x: x.mean() * 252)
                port2[f"{self.long_index}_std"] = port2.groupby(byvar)[
                    self.long_index
                ].transform(lambda x: x.std() * math.sqrt(252))
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
        temp = self.pre_process()
        if self.input_type in ["value", "fractile"]:
            fractile = self.gen_fractile(temp, np.ceil(100 / self.fractile[0]))
        else:
            fractile = None
        daily_stats, daily_stats2, turnover = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        for byvar in self.byvar_list:

            if self.verbose:
                print(f"Processing byvar: {byvar}")

            if byvar not in ["overall", "year", "capyr"]:
                temp = pd.merge_asof(
                    temp.sort_values(by=["date_sig"]),
                    self.otherfile[["security_id", "date", byvar]]
                    .rename(columns={"date": "date_sig"})
                    .sort_values(by="date_sig"),
                    by="security_id",
                    on="date_sig",
                    allow_exact_matches=True,
                    direction="backward",
                    tolerance=pd.Timedelta("20d"),
                ).dropna()
            elif byvar == "capyr":
                temp = pd.merge_asof(
                    temp.sort_values(by=["date_sig"]),
                    self.otherfile[["security_id", "date", "cap"]]
                    .rename(columns={"date": "date_sig"})
                    .sort_values(by="date_sig"),
                    by="security_id",
                    on="date_sig",
                    allow_exact_matches=True,
                    direction="backward",
                    tolerance=pd.Timedelta("20d"),
                ).dropna()

                temp["cap"] = np.select(
                    [temp["cap"] == 1, temp["cap"] == 2],
                    ["LargeCap", "MediumCap"],
                    "SmallCap",
                )
                temp["capyr"] = temp["cap"] + "_" + temp["year"].astype("str")

            if byvar == "overall":
                combo, daily_stats, turnover = self.backtest(temp, byvar)

            elif byvar == "cap":
                combo, daily_stats2 = self.backtest(temp, byvar)
            else:
                combo = self.backtest(temp, byvar)
            result.append(combo)

        if self.earnings_window:
            temp2 = temp.merge(self.window_file, on=["security_id", "date"])
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
                temp2 = temp.merge(vix, on=["date"])
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
