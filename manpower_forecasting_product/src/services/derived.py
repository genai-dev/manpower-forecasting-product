from __future__ import annotations
import numpy as np
import pandas as pd


def derive_productivity_actual(manpower_actual: pd.DataFrame, nop_actual: pd.DataFrame):
    dims=["Channel","Zone","Vintage","Month"]
    n=nop_actual.groupby(dims,observed=True,as_index=False)["NOP"].sum()
    m=manpower_actual[dims+["Closing"]]
    out=n.merge(m,on=dims,how="left")
    out["Productivity"]=np.where(out["Closing"].gt(0),out["NOP"]/out["Closing"],np.nan)
    return out


def derive_productivity_forecast(manpower_fc: pd.DataFrame, nop_fc: pd.DataFrame):
    dims=["Channel","Zone","Vintage","Month"]
    n=nop_fc.groupby(dims,observed=True,as_index=False)["ForecastNOP"].sum()
    m=manpower_fc[dims+["ForecastClosing"]]
    out=n.merge(m,on=dims,how="left")
    out["ForecastProductivity"]=np.where(out["ForecastClosing"].gt(0),out["ForecastNOP"]/out["ForecastClosing"],np.nan)
    return out
