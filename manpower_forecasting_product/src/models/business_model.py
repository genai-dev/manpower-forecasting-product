from __future__ import annotations

import pandas as pd
from src.models.panel_forecaster import PanelForecaster

BUSINESS_DIMS = ["Channel", "Zone", "Vintage", "ProductMixSegment", "SubIDCode"]


def train_business_models(nop: pd.DataFrame, ats: pd.DataFrame):
    nop_model = PanelForecaster("NOP", BUSINESS_DIMS, count_target=True)
    ats_model = PanelForecaster("ATS", BUSINESS_DIMS, count_target=False)
    nres = nop_model.fit(nop[BUSINESS_DIMS + ["Month", "NOP"]])
    ares = ats_model.fit(ats[BUSINESS_DIMS + ["Month", "ATS"]])
    validation = pd.concat([nres.validation, ares.validation], ignore_index=True)
    return nop_model, ats_model, validation
