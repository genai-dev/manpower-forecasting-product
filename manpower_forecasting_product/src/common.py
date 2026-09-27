from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config.business_rules import (
    ATS_VINTAGE_BREAKPOINTS,
    CHANNEL_MAP,
    EXCLUDE_CHANNEL_CONTAINS,
    EXCLUDE_CHANNEL_EXACT,
    VALID_ZONES,
    ZONE_ALIASES,
)


def norm_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().casefold()
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s)


def find_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True):
    lookup = {norm_text(c): c for c in df.columns}
    for a in aliases:
        k = norm_text(a)
        if k in lookup:
            return lookup[k]
    for a in aliases:
        k = norm_text(a)
        for nk, original in lookup.items():
            if len(k) >= 5 and (k in nk or nk in k):
                return original
    if required:
        raise ValueError(f"Missing required column. Tried {list(aliases)}. Found {list(df.columns)}")
    return None


def normalize_zone(value):
    k = norm_text(value)
    return ZONE_ALIASES.get(k, np.nan)


def normalize_channel(value, use_mapping: bool = True):
    if pd.isna(value):
        return np.nan
    raw = re.sub(r"\s+", " ", str(value).strip())
    key = raw.casefold()
    mapped = CHANNEL_MAP.get(key, raw) if use_mapping else raw
    mapped = re.sub(r"\s+", " ", str(mapped).strip())
    return mapped


def is_excluded_channel(value) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip()
    if s in EXCLUDE_CHANNEL_EXACT:
        return True
    k = s.casefold()
    return any(term in k for term in EXCLUDE_CHANNEL_CONTAINS)


def normalize_vintage(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().casefold().replace("months", "m").replace("month", "m")
    s = re.sub(r"\s+", " ", s)
    aliases = {
        "0-3": "0-3 M", "0-3 m": "0-3 M", "0 - 3": "0-3 M", "0 - 3 m": "0-3 M",
        "4-6": "4-6 M", "4-6 m": "4-6 M", "4 - 6": "4-6 M", "4 - 6 m": "4-6 M",
        "7-12": "7-12 M", "7-12 m": "7-12 M", "7 - 12": "7-12 M", "7 - 12 m": "7-12 M",
        "12-24": "13-24 M", "12-24 m": "13-24 M", "13-24": "13-24 M", "13-24 m": "13-24 M",
        "25+": "24+ M", "24+": "24+ M", "above 24": "24+ M", "above 24 m": "24+ M", "above24": "24+ M",
    }
    return aliases.get(s, str(value).strip())


def vintage_from_numeric(value):
    """Python equivalent of =VLOOKUP(vintage, Sheet1!A1:B5, 2, TRUE)."""
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(x) or x < 0:
        return np.nan
    label = ATS_VINTAGE_BREAKPOINTS[0][1]
    for threshold, bucket in ATS_VINTAGE_BREAKPOINTS:
        if float(x) >= threshold:
            label = bucket
        else:
            break
    return label


def normalize_fy(value):
    k = norm_text(value).replace("'", "").replace(" ", "")
    if k in {"fy24", "fy2024", "23/24", "2023/24", "23/24.0", "2023-24", "23-24"}:
        return "FY24"
    if k in {"fy25", "fy2025", "24/25", "2024/25", "2024-25", "24-25"}:
        return "FY25"
    if k in {"fy26", "fy2026", "25/26", "2025/26", "2025-26", "25-26"}:
        return "FY26"
    m = re.search(r"(?:fy)?(24|25|26)$", k)
    return f"FY{m.group(1)}" if m else np.nan


def fiscal_month(value, fy: str, year_hint: int | None = None):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).to_period("M").to_timestamp()
    s = str(value).strip()
    # If year is explicitly present, use it.
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.notna(parsed) and re.search(r"\d{2,4}", s):
        return pd.Timestamp(parsed).to_period("M").to_timestamp()
    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    token = re.sub(r"[^a-z]", "", s.casefold())
    m = month_names.get(token)
    if m is None:
        # Supports forms such as 1.Apr / APR / Apr-23.
        found = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s.casefold())
        if found:
            m = month_names[found.group(1)]
        else:
            return pd.NaT
    if year_hint:
        return pd.Timestamp(year=int(year_hint), month=m, day=1)
    start_year = {"FY24": 2023, "FY25": 2024, "FY26": 2025, "FY27": 2026}.get(fy)
    if start_year is None:
        return pd.NaT
    y = start_year if m >= 4 else start_year + 1
    return pd.Timestamp(year=y, month=m, day=1)


def sheet_year_hint(sheet_name: str, fy: str) -> int | None:
    """Use sheet names to disambiguate month year (e.g. Apr-Dec'23, Jan-Mar'24)."""
    s = str(sheet_name)
    years = re.findall(r"(?:20)?(\d{2})", s)
    candidates = [2000 + int(y) for y in years if int(y) in {23, 24, 25, 26, 27}]
    if candidates:
        # Prefer the last explicit year in a sheet label.
        return candidates[-1]
    return None


def read_excel_sheet(path: Path, sheet_name: str, header=0) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    engine = "pyxlsb" if suffix == ".xlsb" else "openpyxl"
    return pd.read_excel(path, sheet_name=sheet_name, header=header, engine=engine)


def excel_file(path: Path) -> pd.ExcelFile:
    engine = "pyxlsb" if path.suffix.casefold() == ".xlsb" else "openpyxl"
    return pd.ExcelFile(path, engine=engine)


def detect_header(path: Path, sheet_name: str, expected_terms: Iterable[str], max_rows: int = 25) -> int:
    suffix = path.suffix.casefold()
    engine = "pyxlsb" if suffix == ".xlsb" else "openpyxl"
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_rows, engine=engine)
    expected = [norm_text(x) for x in expected_terms]
    best_idx, best_score = 0, -1
    for i in range(len(preview)):
        vals = [norm_text(v) for v in preview.iloc[i].tolist() if pd.notna(v)]
        score = sum(any(e == v or e in v or v in e for e in expected) for v in vals)
        if score > best_score:
            best_idx, best_score = i, score
    if best_score < 2:
        raise ValueError(f"Could not detect header in {path.name}/{sheet_name}; best score={best_score}")
    return best_idx


def load_sheet_auto(path: Path, sheet_name: str, expected_terms: Iterable[str]) -> pd.DataFrame:
    header = detect_header(path, sheet_name, expected_terms)
    df = read_excel_sheet(path, sheet_name, header=header)
    return df.dropna(how="all").dropna(axis=1, how="all")
