from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from src.services.derived import derive_productivity_actual, derive_productivity_forecast

BUS_DIMS=["Channel","Zone","Vintage","ProductMixSegment","SubIDCode"]
MP_DIMS=["Channel","Zone","Vintage"]


def _combined(actual, forecast, actual_col, forecast_col, dims):
    a=actual[dims+["Month",actual_col]].copy().rename(columns={actual_col:"Value"}); a["DataType"]="Actual"
    f=forecast[dims+["Month",forecast_col]].copy().rename(columns={forecast_col:"Value"}); f["DataType"]="Forecast"
    return pd.concat([a,f],ignore_index=True).sort_values(dims+["Month"])


def export_actual_forecast(path: Path, manpower_actual, hiring_actual, nop_actual, ats_actual, manpower_fc, hiring_fc, nop_fc, ats_fc, validation=None):
    # Detailed manpower metrics.
    close=_combined(manpower_actual,manpower_fc,"Closing","ForecastClosing",MP_DIMS)
    exit_df=_combined(manpower_actual,manpower_fc,"Exit","ForecastExit",MP_DIMS)
    opening=_combined(manpower_actual,manpower_fc,"Opening","ForecastOpening",MP_DIMS)
    attr=_combined(manpower_actual,manpower_fc,"AttritionRate","ForecastAttritionRate",MP_DIMS)

    # Hiring is intentionally Channel+Zone+Month.
    hdims=["Channel","Zone"]
    ha=hiring_actual[hdims+["Month","Hiring"]].copy().rename(columns={"Hiring":"Value"}); ha["DataType"]="Actual"
    hf=hiring_fc[hdims+["Month","ForecastHiring"]].copy().rename(columns={"ForecastHiring":"Value"}); hf["DataType"]="Forecast"
    hiring=pd.concat([ha,hf],ignore_index=True).sort_values(hdims+["Month"])

    nop=_combined(nop_actual,nop_fc,"NOP","ForecastNOP",BUS_DIMS)
    ats=_combined(ats_actual,ats_fc,"ATS","ForecastATS",BUS_DIMS)
    prod_a=derive_productivity_actual(manpower_actual,nop_actual); prod_f=derive_productivity_forecast(manpower_fc,nop_fc)
    productivity=_combined(prod_a,prod_f,"Productivity","ForecastProductivity",MP_DIMS)

    def wide(df,dims):
        t=df.copy(); t["MonthLabel"]=pd.to_datetime(t.Month).dt.strftime("%b-%y")
        # Actual/forecast do not overlap by design; agg first is safe.
        return t.pivot_table(index=dims,columns="MonthLabel",values="Value",aggfunc="first",observed=True).reset_index()

    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        wide(close,MP_DIMS).to_excel(writer,sheet_name="Closing Actual+Forecast",index=False)
        wide(exit_df,MP_DIMS).to_excel(writer,sheet_name="Exit Actual+Forecast",index=False)
        wide(opening,MP_DIMS).to_excel(writer,sheet_name="Opening Actual+Forecast",index=False)
        wide(attr,MP_DIMS).to_excel(writer,sheet_name="Attrition Actual+Forecast",index=False)
        wide(hiring,hdims).to_excel(writer,sheet_name="Hiring Actual+Forecast",index=False)
        wide(nop,BUS_DIMS).to_excel(writer,sheet_name="NOP Actual+Forecast",index=False)
        wide(productivity,MP_DIMS).to_excel(writer,sheet_name="Productivity A+F",index=False)
        wide(ats,BUS_DIMS).to_excel(writer,sheet_name="ATS Actual+Forecast",index=False)
        manpower_fc.to_excel(writer,sheet_name="Manpower Forecast Detail",index=False)
        nop_fc.to_excel(writer,sheet_name="NOP Forecast Detail",index=False)
        ats_fc.to_excel(writer,sheet_name="ATS Forecast Detail",index=False)
        if validation is not None: validation.to_excel(writer,sheet_name="Model Validation",index=False)
    return path
