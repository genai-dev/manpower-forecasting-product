from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.common import excel_file, find_col, normalize_channel, normalize_vintage, normalize_zone, vintage_from_numeric


def _sheet(xls, name):
    lookup={str(s).strip().casefold():s for s in xls.sheet_names}
    if name.casefold() not in lookup: return None
    return lookup[name.casefold()]


def read_standard_monthly_workbook(path: Path):
    """
    Future standard template:
      Manpower: Month, Employee ID, Channel, Zone, Vintage, FOS/Non FOS(optional)
      Exit: Exit Month, Employee ID, Channel, Zone, Vintage, FOS/Non FOS(optional)
      Business: Month, Policy Ref, Channel, Zone, Vintage or Vintage Numeric,
                Product Mix Segment, Sub ID Code, Rated NB in Crs
    Returns monthly aggregated canonical frames.
    """
    xls=excel_file(path)
    result={}
    mp_sheet=_sheet(xls,"Manpower")
    if mp_sheet:
        df=pd.read_excel(path,sheet_name=mp_sheet,engine="pyxlsb" if path.suffix.casefold()==".xlsb" else "openpyxl")
        month=find_col(df,["Month"]); emp=find_col(df,["Employee ID","Emp No","Employee No"]); ch=find_col(df,["Channel"]); z=find_col(df,["Zone"]); v=find_col(df,["Vintage","Vintage Slab"])
        fos=find_col(df,["FOS/Non FOS"],required=False)
        if fos: df=df[df[fos].astype(str).str.strip().str.casefold().eq("fos")]
        w=pd.DataFrame({"Channel":df[ch].map(lambda x:normalize_channel(x,True)),"Zone":df[z].map(normalize_zone),"Vintage":df[v].map(normalize_vintage),"Month":pd.to_datetime(df[month]).dt.to_period("M").dt.to_timestamp(),"EmployeeID":df[emp]})
        result["closing"]=w[w.EmployeeID.notna()].groupby(["Channel","Zone","Vintage","Month"],observed=True).size().reset_index(name="Closing")
    ex_sheet=_sheet(xls,"Exit")
    if ex_sheet:
        df=pd.read_excel(path,sheet_name=ex_sheet,engine="pyxlsb" if path.suffix.casefold()==".xlsb" else "openpyxl")
        month=find_col(df,["Exit Month","Month"]); emp=find_col(df,["Employee ID","Emp No","Employee No"]); ch=find_col(df,["Channel"]); z=find_col(df,["Zone"]); v=find_col(df,["Vintage","Vintage Slab"])
        fos=find_col(df,["FOS/Non FOS"],required=False)
        if fos: df=df[df[fos].astype(str).str.strip().str.casefold().eq("fos")]
        w=pd.DataFrame({"Channel":df[ch].map(lambda x:normalize_channel(x,True)),"Zone":df[z].map(normalize_zone),"Vintage":df[v].map(normalize_vintage),"Month":pd.to_datetime(df[month]).dt.to_period("M").dt.to_timestamp(),"EmployeeID":df[emp]})
        result["exit"]=w[w.EmployeeID.notna()].groupby(["Channel","Zone","Vintage","Month"],observed=True).size().reset_index(name="Exit")
    bs_sheet=_sheet(xls,"Business")
    if bs_sheet:
        df=pd.read_excel(path,sheet_name=bs_sheet,engine="pyxlsb" if path.suffix.casefold()==".xlsb" else "openpyxl")
        month=find_col(df,["Month"]); policy=find_col(df,["Policy Ref"]); ch=find_col(df,["Channel"]); z=find_col(df,["Zone"]); pm=find_col(df,["Product Mix Segment"]); sub=find_col(df,["Sub ID Code"],required=False); rated=find_col(df,["Rated NB in Crs"])
        slab=find_col(df,["Vintage Slab","Vintage"],required=False); numeric=find_col(df,["Vintage Numeric","Vintage Months","Vintage (Months)"],required=False)
        vintage=df[numeric].map(vintage_from_numeric) if numeric else df[slab].map(normalize_vintage)
        w=pd.DataFrame({"Channel":df[ch].map(lambda x:normalize_channel(x,True)),"Zone":df[z].map(normalize_zone),"Vintage":vintage,"ProductMixSegment":df[pm].astype(str).str.strip(),"Month":pd.to_datetime(df[month]).dt.to_period("M").dt.to_timestamp(),"SubIDCode":df[sub].astype(str).str.strip() if sub else "ALL","PolicyRef":df[policy],"RatedNB":pd.to_numeric(df[rated],errors="coerce")})
        grain=["Channel","Zone","Vintage","ProductMixSegment","Month","SubIDCode"]
        result["nop"]=w[w.PolicyRef.notna()].groupby(grain,observed=True).size().reset_index(name="NOP")
        ats=w[w.PolicyRef.notna()].groupby(grain,observed=True).agg(PolicyCount=("PolicyRef","count"),RatedNBSum=("RatedNB","sum")).reset_index(); ats["ATS"]=np.where(ats.PolicyCount.gt(0),ats.RatedNBSum/ats.PolicyCount,np.nan); result["ats"]=ats
    return result
