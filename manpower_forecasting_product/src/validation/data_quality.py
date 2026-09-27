from __future__ import annotations

import pandas as pd


def validate_monthly_panel(df: pd.DataFrame, dims: list[str], target_cols: list[str], expected_start=None, expected_end=None):
    issues=[]
    required=dims+["Month"]+target_cols
    missing=[c for c in required if c not in df.columns]
    if missing: issues.append({"severity":"ERROR","check":"required_columns","detail":f"Missing columns: {missing}"})
    if missing: return pd.DataFrame(issues)
    d=df.copy(); d["Month"]=pd.to_datetime(d["Month"], errors="coerce")
    if d["Month"].isna().any(): issues.append({"severity":"ERROR","check":"month_parse","detail":f"{int(d['Month'].isna().sum())} invalid Month values"})
    dup=int(d.duplicated(dims+["Month"]).sum())
    if dup: issues.append({"severity":"ERROR","check":"duplicate_grain","detail":f"{dup} duplicate rows at canonical grain"})
    for c in target_cols:
        neg=int((pd.to_numeric(d[c],errors="coerce")<0).sum())
        if neg: issues.append({"severity":"WARN","check":f"negative_{c}","detail":f"{neg} negative values"})
    if expected_start and d["Month"].min()>pd.Timestamp(expected_start): issues.append({"severity":"WARN","check":"history_start","detail":f"Starts {d['Month'].min().date()} not {expected_start}"})
    if expected_end and d["Month"].max()<pd.Timestamp(expected_end): issues.append({"severity":"WARN","check":"history_end","detail":f"Ends {d['Month'].max().date()} not {expected_end}"})
    if not issues: issues.append({"severity":"OK","check":"panel_validation","detail":"All core panel checks passed"})
    return pd.DataFrame(issues)


def manpower_reconciliation(base: pd.DataFrame, hiring: pd.DataFrame):
    monthly=base.groupby("Month",as_index=False).agg(Closing=("Closing","sum"),Exit=("Exit","sum"),Opening=("Opening",lambda x:x.sum(min_count=1)))
    h=hiring.groupby("Month",as_index=False)["Hiring"].sum(min_count=1)
    out=monthly.merge(h,on="Month",how="left")
    out["IdentityResidual"]=out["Closing"]-(out["Opening"]+out["Hiring"]-out["Exit"])
    return out


def eda_summary(df: pd.DataFrame, dims: list[str], targets: list[str]):
    d=df.copy(); d["Month"]=pd.to_datetime(d["Month"],errors="coerce")
    summary={
        "rows":int(len(d)),
        "min_month":str(d["Month"].min().date()) if d["Month"].notna().any() else None,
        "max_month":str(d["Month"].max().date()) if d["Month"].notna().any() else None,
        "missing_values":{c:int(d[c].isna().sum()) for c in [*dims,"Month",*targets] if c in d.columns},
        "unique_categories":{c:int(d[c].nunique(dropna=True)) for c in dims if c in d.columns},
        "category_values":{c:sorted(d[c].dropna().astype(str).unique().tolist())[:200] for c in dims if c in d.columns},
    }
    monthly=d.groupby("Month",as_index=False)[targets].sum(min_count=1) if targets else pd.DataFrame()
    return summary,monthly
