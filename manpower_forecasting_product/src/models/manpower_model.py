from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.panel_forecaster import PanelForecaster
from src.models.exit_forecaster import HierarchicalExitForecaster


MANPOWER_DIMS = ["Channel", "Zone", "Vintage"]


def train_manpower_models(base: pd.DataFrame):
    closing = PanelForecaster("Closing", MANPOWER_DIMS, count_target=True)
    exit_model = HierarchicalExitForecaster()
    closing_result = closing.fit(base[MANPOWER_DIMS + ["Month", "Closing"]])
    exit_validation = exit_model.fit(base[MANPOWER_DIMS + ["Month", "Exit"]])
    return closing, exit_model, pd.concat([closing_result.validation, exit_validation], ignore_index=True)


def forecast_manpower(base: pd.DataFrame, closing_model, exit_model, start="2026-04-01", periods=12):
    closing_fc = closing_model.forecast(base[MANPOWER_DIMS + ["Month", "Closing"]], start, periods)
    exit_fc = exit_model.forecast(base[MANPOWER_DIMS + ["Month", "Exit"]], start, periods)
    keys = MANPOWER_DIMS + ["Month"]
    future = closing_fc.merge(exit_fc, on=keys, how="outer")

    # Opening at detailed grain is prior Closing; Apr forecast opens from actual Mar closing.
    last_actual = base[[*MANPOWER_DIMS, "Month", "Closing"]].copy()
    actual_lookup = {
        tuple(row[d] for d in MANPOWER_DIMS) + (pd.Timestamp(row["Month"]),): float(row["Closing"])
        for _, row in last_actual.iterrows()
    }
    future = future.sort_values(keys).reset_index(drop=True)
    future["ForecastOpening"] = np.nan
    for group_key, idx in future.groupby(MANPOWER_DIMS, observed=True).groups.items():
        g = future.loc[idx].sort_values("Month")
        if not isinstance(group_key, tuple): group_key = (group_key,)
        prev_month = (g["Month"].min() - pd.DateOffset(months=1)).to_period("M").to_timestamp()
        prev = actual_lookup.get(group_key + (prev_month,), np.nan)
        vals = []
        for ridx, row in g.iterrows():
            vals.append(prev)
            prev = row["ForecastClosing"]
        future.loc[g.index, "ForecastOpening"] = vals
    future["ForecastAttritionRate"] = np.where(
        future["ForecastOpening"].gt(0), future["ForecastExit"] / future["ForecastOpening"], np.nan
    )

    # Hiring is reconciled at Channel+Zone+Month.
    hiring = future.groupby(["Channel", "Zone", "Month"], observed=True, as_index=False).agg(
        Opening=("ForecastOpening", lambda x: x.sum(min_count=1)),
        Closing=("ForecastClosing", "sum"),
        Exit=("ForecastExit", "sum"),
    )
    hiring["ForecastHiring"] = np.where(
        hiring["Opening"].notna(), hiring["Closing"] - hiring["Opening"] + hiring["Exit"], np.nan
    )
    return future, hiring
