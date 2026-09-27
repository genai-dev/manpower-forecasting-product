from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.common import (
    excel_file, find_col, fiscal_month, is_excluded_channel, load_sheet_auto,
    normalize_channel, normalize_vintage, normalize_zone, sheet_year_hint, vintage_from_numeric,
)

EXPECTED = ["Vertical", "Policy Ref", "Vintage Slab", "Vintage", "Product Mix Segment", "Month", "Sub ID Code", "Rated NB in Crs", "Zone"]


def _all_data_sheets(path: Path):
    return list(excel_file(path).sheet_names)


def _normalise_business_frame(df: pd.DataFrame, sheet_name: str, fy: str, need_ats: bool):
    vertical_col = find_col(df, ["Vertical", "Channel", "Channel Mapping"])
    policy_col = find_col(df, ["Policy Ref", "Policy Reference", "Policy No"])
    product_col = find_col(df, ["Product Mix Segment", "Product Mix", "Product Segment"])
    zone_col = find_col(df, ["Zone", "Zone/Region", "Zone Region"])
    subid_col = find_col(df, ["Sub ID Code", "Sub Id Code", "SubID Code", "Sub ID"], required=False)
    month_col = find_col(df, ["Month", "Accounting Month", "Policy Month"], required=False)

    w = df.copy()
    w["Channel"] = w[vertical_col].map(lambda x: normalize_channel(x, use_mapping=True))
    w = w[~w["Channel"].map(is_excluded_channel)].copy()
    w["Zone"] = w[zone_col].map(normalize_zone)
    w["ProductMixSegment"] = w[product_col].astype(str).str.strip()
    w["SubIDCode"] = w[subid_col].astype(str).str.strip() if subid_col else "ALL"
    w["PolicyRef"] = w[policy_col]

    year_hint = sheet_year_hint(sheet_name, fy)
    if month_col is None:
        raise ValueError(
            f"{path_label(df)} / {sheet_name}: Month column is required because each sheet spans multiple months. "
            "The sheet name can provide the year, but not the row-level month."
        )
    w["Month"] = [fiscal_month(v, fy, year_hint=year_hint) for v in w[month_col]]

    # NOP may already contain Vintage Slab. ATS explicitly requires numeric-Vintage -> VLOOKUP-equivalent slab creation.
    if need_ats:
        rated_col = find_col(df, ["Rated NB in Crs", "Rated NB", "Rated NB Crs"])
        numeric_vintage = find_col(df, ["Vintage", "Vintage Months", "Vintage (Months)", "Vintage Month"], required=False)
        slab_col = find_col(df, ["Vintage Slab", "Vintage Bracket"], required=False)
        if numeric_vintage is not None:
            w["Vintage"] = w[numeric_vintage].map(vintage_from_numeric)
        elif slab_col is not None:
            w["Vintage"] = w[slab_col].map(normalize_vintage)
        else:
            raise ValueError(f"{sheet_name}: ATS needs numeric Vintage or Vintage Slab")
        w["RatedNB"] = pd.to_numeric(w[rated_col], errors="coerce")
    else:
        slab_col = find_col(df, ["Vintage Slab", "Vintage Bracket"])
        w["Vintage"] = w[slab_col].map(normalize_vintage)

    w = w[w["Zone"].notna() & w["Vintage"].notna() & w["Month"].notna() & w["PolicyRef"].notna()].copy()
    return w


def path_label(df):
    return "Business workbook"


def consolidate_business_file(path: Path, fy: str):
    """Concatenate both half-year sheets into one FY-level raw table before aggregation."""
    nop_parts, ats_parts = [], []
    for sheet in _all_data_sheets(path):
        try:
            df = load_sheet_auto(path, sheet, EXPECTED)
            # Skip helper/lookup sheets that do not contain the business fields.
            find_col(df, ["Policy Ref", "Policy Reference", "Policy No"])
            find_col(df, ["Vertical", "Channel", "Channel Mapping"])
        except Exception:
            continue
        nop_parts.append(_normalise_business_frame(df, sheet, fy, need_ats=False))
        ats_parts.append(_normalise_business_frame(df, sheet, fy, need_ats=True))
    if not nop_parts:
        raise ValueError(f"No valid business data sheets found in {path.name}")
    return pd.concat(nop_parts, ignore_index=True), pd.concat(ats_parts, ignore_index=True)


def build_business_history(file_specs: list[dict]):
    nop_all, ats_all = [], []
    for spec in file_specs:
        path = Path(spec["path"])
        fy = spec["fy"]
        nop_raw, ats_raw = consolidate_business_file(path, fy)
        nop_all.append(nop_raw)
        ats_all.append(ats_raw)

    nop_raw = pd.concat(nop_all, ignore_index=True)
    ats_raw = pd.concat(ats_all, ignore_index=True)
    grain = ["Channel", "Zone", "Vintage", "ProductMixSegment", "Month", "SubIDCode"]

    nop = nop_raw.groupby(grain, observed=True, as_index=False).agg(NOP=("PolicyRef", "count"))

    ats = ats_raw.groupby(grain, observed=True, as_index=False).agg(
        PolicyCount=("PolicyRef", "count"),
        RatedNBSum=("RatedNB", "sum"),
    )
    # ATS = average rated NB per policy. Keeping both components makes the calculation auditable.
    ats["ATS"] = np.where(ats["PolicyCount"].gt(0), ats["RatedNBSum"] / ats["PolicyCount"], np.nan)
    return nop.sort_values(grain).reset_index(drop=True), ats.sort_values(grain).reset_index(drop=True)
