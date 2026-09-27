from __future__ import annotations

import numpy as np
import pandas as pd


def apply_manpower_scenario(baseline_hiring: pd.DataFrame, overrides: list[dict]):
    """
    Scenario grain: Channel + Zone + Month.
    Supported overrides: Hiring, Exit, AttritionRate. Optional mode='pct' applies percentage change.
    Scenario values are temporary and must never be appended to training data.
    """
    b=baseline_hiring.copy().rename(columns={"Opening":"BaselineOpening","Closing":"BaselineClosing","Exit":"BaselineExit","ForecastHiring":"BaselineHiring"})
    b["Month"]=pd.to_datetime(b["Month"]).dt.to_period("M").dt.to_timestamp()
    b["BaselineAttritionRate"]=np.where(b["BaselineOpening"].gt(0),b["BaselineExit"]/b["BaselineOpening"],np.nan)
    ov={(str(x["Channel"]),str(x["Zone"]),pd.Timestamp(x["Month"]).to_period("M").to_timestamp()):x for x in overrides}
    rows=[]
    for (ch,z),g in b.groupby(["Channel","Zone"],observed=True):
        g=g.sort_values("Month"); prev_close=None
        for _,r in g.iterrows():
            opening=float(r["BaselineOpening"]) if prev_close is None or pd.isna(prev_close) else float(prev_close)
            hiring=float(r["BaselineHiring"]); exit_count=float(r["BaselineExit"]); attr=exit_count/opening if opening else np.nan
            x=ov.get((str(ch),str(z),pd.Timestamp(r["Month"])))
            if x:
                field=x["field"]; val=float(x["value"]); mode=x.get("mode","absolute")
                def change(old): return old*(1+val/100.0) if mode=="pct" else val
                if field=="Hiring": hiring=change(hiring)
                elif field=="Exit": exit_count=change(exit_count); attr=exit_count/opening if opening else np.nan
                elif field=="AttritionRate":
                    attr=change(attr); exit_count=round(max(0,attr*opening)) if opening else 0
                else: raise ValueError(f"Unsupported manpower scenario field: {field}")
            closing=max(0,opening+hiring-exit_count); prev_close=closing
            rows.append({"Channel":ch,"Zone":z,"Month":r["Month"],"BaselineOpening":r["BaselineOpening"],"BaselineHiring":r["BaselineHiring"],"BaselineExit":r["BaselineExit"],"BaselineClosing":r["BaselineClosing"],"ScenarioOpening":opening,"ScenarioHiring":hiring,"ScenarioExit":exit_count,"ScenarioAttritionRate":attr,"ScenarioClosing":closing,"ClosingDelta":closing-r["BaselineClosing"]})
    return pd.DataFrame(rows)


def apply_business_scenario(nop_forecast: pd.DataFrame, ats_forecast: pd.DataFrame, overrides: list[dict]):
    keys=["Channel","Zone","Vintage","ProductMixSegment","SubIDCode","Month"]
    b=nop_forecast.merge(ats_forecast,on=keys,how="outer")
    b["BaselineNOP"]=b["ForecastNOP"]; b["BaselineATS"]=b["ForecastATS"]; b["ScenarioNOP"]=b["BaselineNOP"]; b["ScenarioATS"]=b["BaselineATS"]
    for x in overrides:
        mask=pd.Series(True,index=b.index)
        for k in ["Channel","Zone","Vintage","ProductMixSegment","SubIDCode"]:
            if k in x and x[k] is not None: mask &= b[k].astype(str).eq(str(x[k]))
        if "Month" in x: mask &= pd.to_datetime(b["Month"]).dt.to_period("M").eq(pd.Timestamp(x["Month"]).to_period("M"))
        field=x["field"]; val=float(x["value"]); mode=x.get("mode","absolute")
        col="ScenarioNOP" if field=="NOP" else "ScenarioATS" if field=="ATS" else None
        if col is None: raise ValueError(f"Unsupported business scenario field: {field}")
        b.loc[mask,col]=b.loc[mask,col]*(1+val/100.0) if mode=="pct" else val
    # ATS is Rated NB per policy, therefore NOP*ATS recreates total Rated NB at this grain.
    b["BaselineRatedNB"]=b["BaselineNOP"]*b["BaselineATS"]
    b["ScenarioRatedNB"]=b["ScenarioNOP"]*b["ScenarioATS"]
    b["RatedNBDelta"]=b["ScenarioRatedNB"]-b["BaselineRatedNB"]
    return b
