from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LAGS = [1, 2, 3, 6, 12]


def wape(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    den = np.abs(a).sum()
    return np.abs(a - p).sum() / den if den else np.nan


def metrics(actual, pred):
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": math.sqrt(mean_squared_error(actual, pred)),
        "WAPE": wape(actual, pred),
    }


@dataclass
class ModelResult:
    name: str
    validation: pd.DataFrame


class PanelForecaster:
    """
    Global panel forecaster for sparse monthly Channel/Zone/Vintage style data.
    Uses time-aware lag/rolling features and compares candidate models to Lag-12.
    No random sampling and no SMOTE.
    """

    def __init__(self, target: str, categorical_cols: Sequence[str], count_target: bool = True):
        self.target = target
        self.categorical_cols = list(categorical_cols)
        self.count_target = count_target
        self.numeric_cols = [
            "MonthNo", "Quarter", "TimeIndex", "Lag1", "Lag2", "Lag3", "Lag6", "Lag12",
            "Rolling3", "Rolling6", "Rolling12", "RollingStd3", "YoYChange",
        ]
        self.model = None
        self.model_name = None
        self.validation = None
        self.min_month = None

    def _features(self, history: pd.DataFrame):
        out = history.copy().sort_values(self.categorical_cols + ["Month"]).reset_index(drop=True)
        out["Month"] = pd.to_datetime(out["Month"]).dt.to_period("M").dt.to_timestamp()
        self.min_month = out["Month"].min() if self.min_month is None else self.min_month
        out["MonthNo"] = out["Month"].dt.month
        out["Quarter"] = out["Month"].dt.quarter
        out["TimeIndex"] = (out["Month"].dt.year - self.min_month.year) * 12 + (out["Month"].dt.month - self.min_month.month)
        g = out.groupby(self.categorical_cols, observed=True, dropna=False)[self.target]
        for lag in LAGS:
            out[f"Lag{lag}"] = g.shift(lag)
        out["Rolling3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        out["Rolling6"] = g.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
        out["Rolling12"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
        out["RollingStd3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
        out["YoYChange"] = np.where(out["Lag12"].notna(), out["Lag1"] - out["Lag12"], np.nan)
        return out

    def _pipeline(self, kind: str):
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
            ("num", "passthrough", self.numeric_cols),
        ])
        if kind == "hgb":
            estimator = HistGradientBoostingRegressor(
                learning_rate=0.045, max_iter=350, max_leaf_nodes=24,
                min_samples_leaf=15, l2_regularization=2.0, random_state=42,
                loss="poisson" if self.count_target else "squared_error",
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=500, random_state=42, n_jobs=-1,
                min_samples_leaf=2, max_features="sqrt",
            )
        return Pipeline([("preprocessing", prep), ("model", estimator)])

    def fit(self, history: pd.DataFrame, validation_months: int = 6):
        f = self._features(history)
        md = f[f[self.target].notna() & f["Lag12"].notna()].copy()
        if md.empty:
            raise ValueError(f"Not enough history for {self.target}; at least 13 months are needed")
        md[self.numeric_cols] = md[self.numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        months = sorted(pd.to_datetime(md["Month"]).unique())
        cut = pd.Timestamp(months[-validation_months])
        train = md[md["Month"] < cut].copy()
        valid = md[md["Month"] >= cut].copy()
        xcols = self.categorical_cols + self.numeric_cols
        rows, fitted = [], {}
        for name, kind in [("Random Forest", "rf"), ("HistGradientBoosting", "hgb")]:
            model = self._pipeline(kind)
            model.fit(train[xcols], train[self.target])
            pred = np.clip(model.predict(valid[xcols]), 0, None)
            m = metrics(valid[self.target], pred)
            rows.append({"Target": self.target, "Model": name, **m})
            fitted[name] = kind
        baseline = valid["Lag12"].clip(lower=0).values
        rows.append({"Target": self.target, "Model": "Lag12 Baseline", **metrics(valid[self.target], baseline)})
        self.validation = pd.DataFrame(rows).sort_values(["WAPE", "MAE", "RMSE"]).reset_index(drop=True)
        ml = self.validation[self.validation["Model"] != "Lag12 Baseline"]
        self.model_name = str(ml.iloc[0]["Model"])
        kind = fitted[self.model_name]
        self.model = self._pipeline(kind)
        self.model.fit(md[xcols], md[self.target])
        return ModelResult(self.model_name, self.validation.copy())

    def forecast(self, history: pd.DataFrame, start: str, periods: int = 12):
        if self.model is None:
            raise RuntimeError("fit() must be called before forecast()")
        h = history.copy()
        h["Month"] = pd.to_datetime(h["Month"]).dt.to_period("M").dt.to_timestamp()
        groups = h[self.categorical_cols].drop_duplicates().reset_index(drop=True)
        values = {}
        for key, g in h.groupby(self.categorical_cols, observed=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            values[key] = {pd.Timestamp(m): float(v) for m, v in zip(g["Month"], g[self.target]) if pd.notna(v)}
        months = pd.date_range(pd.Timestamp(start), periods=periods, freq="MS")
        out = []
        for month in months:
            rows, keys = [], []
            for record in groups.itertuples(index=False, name=None):
                key = tuple(record)
                d = values.get(key, {})
                def lag(n):
                    return d.get((month - pd.DateOffset(months=n)).to_period("M").to_timestamp(), np.nan)
                past = [v for m, v in sorted(d.items()) if m < month and pd.notna(v)]
                r3, r6, r12 = past[-3:], past[-6:], past[-12:]
                row = dict(zip(self.categorical_cols, key))
                row.update({
                    "MonthNo": month.month, "Quarter": month.quarter,
                    "TimeIndex": (month.year - self.min_month.year) * 12 + (month.month - self.min_month.month),
                    "Lag1": lag(1), "Lag2": lag(2), "Lag3": lag(3), "Lag6": lag(6), "Lag12": lag(12),
                    "Rolling3": np.mean(r3) if r3 else np.nan,
                    "Rolling6": np.mean(r6) if r6 else np.nan,
                    "Rolling12": np.mean(r12) if r12 else np.nan,
                    "RollingStd3": np.std(r3, ddof=1) if len(r3) >= 2 else np.nan,
                    "YoYChange": lag(1) - lag(12) if pd.notna(lag(1)) and pd.notna(lag(12)) else np.nan,
                })
                rows.append(row); keys.append(key)
            X = pd.DataFrame(rows)
            X[self.numeric_cols] = X[self.numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            pred = np.clip(self.model.predict(X[self.categorical_cols + self.numeric_cols]), 0, None)
            if self.count_target:
                pred = np.rint(pred)
            for key, y in zip(keys, pred):
                y = float(y)
                values.setdefault(key, {})[month] = y
                item = dict(zip(self.categorical_cols, key))
                item.update({"Month": month, f"Forecast{self.target}": y})
                out.append(item)
        return pd.DataFrame(out)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
