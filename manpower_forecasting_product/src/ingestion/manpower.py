from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.common import (
    excel_file, find_col, fiscal_month, is_excluded_channel, load_sheet_auto,
    normalize_channel, normalize_fy, normalize_vintage, normalize_zone,
)

EXPECTED = ["FY", "Channel Mapping", "Sub Channel", "Vintage Slab", "Vintage Bracket", "Month", "Exit Month", "Emp No", "Zone", "FOS/Non FOS"]


def _find_sheet(xls: pd.ExcelFile, aliases: list[str]) -> str:
    lookup = {str(s).strip().casefold(): s for s in xls.sheet_names}
    for a in aliases:
        if a.casefold() in lookup:
            return lookup[a.casefold()]
    for s in xls.sheet_names:
        ns = str(s).strip().casefold()
        if any(a.casefold() in ns or ns in a.casefold() for a in aliases):
            return s
    raise ValueError(f"Could not find sheet {aliases}. Available: {xls.sheet_names}")


def _count_emp(df: pd.DataFrame, emp_col, group_cols, value_name):
    work = df[df[emp_col].notna()].copy()
    return work.groupby(group_cols, observed=True, dropna=False).size().reset_index(name=value_name)


def _clean_aggregated(df: pd.DataFrame, value_name: str):
    out = df.copy()
    out["Channel"] = out["Channel"].map(lambda x: normalize_channel(x, use_mapping=False))
    out["Zone"] = out["Zone"].map(normalize_zone)
    out["Vintage"] = out["Vintage"].map(normalize_vintage)
    out = out[out["Channel"].notna() & out["Zone"].notna() & out["Vintage"].notna() & out["Month"].notna()].copy()
    out = out[~out["Channel"].map(is_excluded_channel)].copy()
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce").fillna(0)
    return out[["Channel", "Zone", "Vintage", "Month", value_name]].sort_values(["Channel", "Zone", "Vintage", "Month"]).reset_index(drop=True)


def load_fy24_fy25_closing(path: Path) -> pd.DataFrame:
    """Authoritative rule: Manpower_FOS; FY filter first; NO FOS filter."""
    xls = excel_file(path)
    sheet = _find_sheet(xls, ["Manpower_FOS", "Manpower FOS"])
    df = load_sheet_auto(path, sheet, EXPECTED)
    fy_col = find_col(df, ["FY", "Financial Year"])
    channel_col = find_col(df, ["Channel Mapping"])
    vintage_col = find_col(df, ["Vintage Slab", "Vintage Bracket"])
    month_col = find_col(df, ["Month"])
    zone_col = find_col(df, ["Zone/Region", "Zone Region", "Zone"])
    emp_col = find_col(df, ["Emp No", "Employee No", "Employee Number"])
    df["_FY"] = df[fy_col].map(normalize_fy)
    parts = []
    for fy in ["FY24", "FY25"]:
        w = df[df["_FY"].eq(fy)].copy()
        if w.empty:
            raise ValueError(f"No rows found for {fy} in {sheet}")
        w["Channel"] = w[channel_col]
        w["Zone"] = w[zone_col]
        w["Vintage"] = w[vintage_col]
        w["Month"] = [fiscal_month(v, fy) for v in w[month_col]]
        counted = _count_emp(w, emp_col, ["Channel", "Zone", "Vintage", "Month"], "Closing")
        parts.append(_clean_aggregated(counted, "Closing"))
    return pd.concat(parts, ignore_index=True)


def _load_exit_sheet(path: Path, sheet_aliases: list[str], fy: str, fos_required: bool) -> pd.DataFrame:
    xls = excel_file(path)
    sheet = _find_sheet(xls, sheet_aliases)
    df = load_sheet_auto(path, sheet, EXPECTED)
    channel_col = find_col(df, ["Channel Mapping"])
    vintage_col = find_col(df, ["Vintage Slab", "Vintage Bracket"])
    month_col = find_col(df, ["Exit Month", "Month"])
    zone_col = find_col(df, ["Zone", "Zone/Region", "Zone Region"])
    emp_col = find_col(df, ["Emp No", "Employee No", "Employee Number"])
    fos_col = find_col(df, ["FOS/Non FOS", "FOS Non FOS", "FOS/Non-FOS"], required=False)
    if fos_required:
        if fos_col is None:
            raise ValueError(f"{sheet}: FOS/Non FOS is required by business rule but not found")
        df = df[df[fos_col].astype(str).str.strip().str.casefold().eq("fos")].copy()
    elif fos_col is not None:
        # FY25: user says already FOS-only. Do not apply a second filter.
        pass
    df["Channel"] = df[channel_col]
    df["Zone"] = df[zone_col]
    df["Vintage"] = df[vintage_col]
    df["Month"] = [fiscal_month(v, fy) for v in df[month_col]]
    counted = _count_emp(df, emp_col, ["Channel", "Zone", "Vintage", "Month"], "Exit")
    return _clean_aggregated(counted, "Exit")


def load_fy24_fy25_exit(path: Path) -> pd.DataFrame:
    fy24 = _load_exit_sheet(path, ["Exit_fy_23_24", "Exit fy 23 24"], "FY24", fos_required=True)
    fy25 = _load_exit_sheet(path, ["Exit_fy_24_25", "Exit fy 24 25"], "FY25", fos_required=False)
    return pd.concat([fy24, fy25], ignore_index=True)


def _load_fy26_metric(path: Path, kind: str) -> pd.DataFrame:
    xls = excel_file(path)
    sheet = _find_sheet(xls, ["Manpower"] if kind == "Closing" else ["Exit"])
    df = load_sheet_auto(path, sheet, EXPECTED)
    sub_col = find_col(df, ["Sub Channel", "Subchannel", "Sub Channel Name"])
    vintage_col = find_col(df, ["Vintage Bracket", "Vintage Slab"])
    month_col = find_col(df, ["Month"] if kind == "Closing" else ["Exit Month", "Month"])
    zone_col = find_col(df, ["Zone", "Zone/Region", "Zone Region"])
    emp_col = find_col(df, ["Emp No", "Employee No", "Employee Number"])
    fos_col = find_col(df, ["FOS/Non FOS", "FOS Non FOS", "FOS/Non-FOS"])
    df = df[df[fos_col].astype(str).str.strip().str.casefold().eq("fos")].copy()
    df["Channel"] = df[sub_col].map(lambda x: normalize_channel(x, use_mapping=True))
    df["Zone"] = df[zone_col]
    df["Vintage"] = df[vintage_col]
    df["Month"] = [fiscal_month(v, "FY26") for v in df[month_col]]
    df = df[~df["Channel"].map(is_excluded_channel)].copy()
    counted = _count_emp(df, emp_col, ["Channel", "Zone", "Vintage", "Month"], kind)
    # Channel mapping already applied, so avoid remapping but normalize formatting.
    out = counted.copy()
    out["Channel"] = out["Channel"].map(lambda x: normalize_channel(x, use_mapping=False))
    out["Zone"] = out["Zone"].map(normalize_zone)
    out["Vintage"] = out["Vintage"].map(normalize_vintage)
    out = out[out["Zone"].notna() & out["Vintage"].notna() & out["Month"].notna()].copy()
    return out[["Channel", "Zone", "Vintage", "Month", kind]]


def build_manpower_history(fy24_25_path: Path, fy26_path: Path):
    closing = pd.concat([load_fy24_fy25_closing(fy24_25_path), _load_fy26_metric(fy26_path, "Closing")], ignore_index=True)
    exit_df = pd.concat([load_fy24_fy25_exit(fy24_25_path), _load_fy26_metric(fy26_path, "Exit")], ignore_index=True)
    keys = ["Channel", "Zone", "Vintage", "Month"]
    closing = closing.groupby(keys, observed=True, as_index=False)["Closing"].sum()
    exit_df = exit_df.groupby(keys, observed=True, as_index=False)["Exit"].sum()

    # Model panel: support comes from Closing; missing Exit for an observed manpower row = 0 exits.
    base = closing.merge(exit_df, on=keys, how="left")
    base["Exit"] = base["Exit"].fillna(0)

    # Opening is previous-month Closing at the same Channel/Zone/Vintage grain.
    prev = closing.copy()
    prev["Month"] = prev["Month"] + pd.offsets.MonthBegin(1)
    prev = prev.rename(columns={"Closing": "Opening"})
    base = base.merge(prev, on=keys, how="left")
    base["AttritionRate"] = np.where(base["Opening"].gt(0), base["Exit"] / base["Opening"], np.nan)

    # Hiring is reconciled at Channel + Zone + Month to avoid false vintage movements being treated as hiring.
    hz = base.groupby(["Channel", "Zone", "Month"], observed=True, as_index=False).agg(
        Opening=("Opening", lambda x: x.sum(min_count=1)),
        Closing=("Closing", "sum"),
        Exit=("Exit", "sum"),
    )
    hz["Hiring"] = np.where(hz["Opening"].notna(), hz["Closing"] - hz["Opening"] + hz["Exit"], np.nan)
    return base.sort_values(keys).reset_index(drop=True), hz.sort_values(["Channel", "Zone", "Month"]).reset_index(drop=True)
