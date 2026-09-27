# Manpower + Business Forecasting Product

Production-oriented Flask project for FY24-FY26 historical manpower/business data, future monthly uploads, model retraining, scenario analysis and LLM insights.

## Historical source rules implemented

### FY24/FY25 manpower workbook
`YTD Mar FY'24 & FY'25 - PDC & Productivity_V6.xlsx`

- Closing: `Manpower_FOS`; filter FY first; use Channel Mapping, Vintage Slab, Month, Zone/Region, COUNT(Emp No). **No FOS filter on Manpower_FOS.**
- FY24 Exit: `Exit_fy_23_24`; filter `FOS/Non FOS = FOS`; Channel Mapping, Vintage Slab, Exit Month, Zone, COUNT(Emp No).
- FY25 Exit: `Exit_fy_24_25`; already FOS-filtered; do not require/reapply an FOS filter.

### FY26 manpower workbook
`IB Attrition Base File FY 25-26 31-03-26 Mar 26 YTD (1).xlsx`

- Closing `Manpower`: map Sub Channel -> Channel; filter FOS; Vintage Bracket, Zone, Month, COUNT(Emp No).
- Exit `Exit`: same channel mapping; filter FOS; Vintage Bracket, Zone, Exit Month, COUNT(Emp No).

### 3-year manpower collation
All three FYs are collated at `Channel + Zone + Vintage + Month` before training.
Opening = previous-month Closing. Attrition = Exit / Opening. Hiring is reconciled at `Channel + Zone + Month` as `Closing - Opening + Exit`.

## Step 4: NOP / Productivity
Reads the 3 accounting files from `D:\Business planning\Business`. Each workbook's two period sheets are concatenated into one FY dataset. Uses:
- Vertical -> Channel mapping
- Policy Ref -> COUNT = NOP
- Zone
- Vintage Slab
- Product Mix Segment
- Month
- Sub ID Code

The year is disambiguated from the sheet name while the row-level month comes from the Month column.

## Step 5: ATS
Uses Policy Ref, Product Mix Segment, Rated NB in Crs, Zone, Vertical, Sub ID Code and Vintage.
The Excel approximate-VLOOKUP vintage logic is reproduced in Python:

| Breakpoint | Bucket |
|---:|---|
| 0 | 0-3 M |
| 4 | 4-6 M |
| 7 | 7-12 M |
| 13 | 13-24 M |
| 24.0001 | 24+ M |

The pipeline keeps both `PolicyCount` and `RatedNBSum` and derives `ATS = RatedNBSum / PolicyCount`. This is auditable and uses the Rated NB field; if the business has another formal ATS formula, change it in `src/ingestion/business.py`.

## ML methodology
- No SMOTE: this is regression/count forecasting, not imbalanced classification.
- No random train/test split: time-aware last-period / rolling validation is used.
- Closing: global Random Forest / HistGradientBoosting candidate comparison plus Lag12 baseline.
- Exit: hierarchical features at Channel+Zone, Channel+Vintage, Channel, Zone, Vintage and portfolio levels; RF/RF-tuned/Poisson-HGB candidate comparison.
- NOP: global count forecast with time lags/rolling features.
- ATS: global continuous forecast with time lags/rolling features.

## One-time history vs future uploads
The business **does not upload the same 3 years every time**.
1. Run `/api/bootstrap` once to create versioned Apr-23..Mar-26 certified datasets.
2. Future business uploads only new month(s) with the standard workbook.
3. Upload validates/appends data but does **not** retrain automatically.
4. An authorized process calls `/api/train` monthly/quarterly/after approval.

## Future standard workbook
- `Manpower`: Month, Employee ID, Channel, Zone, Vintage, FOS/Non FOS(optional)
- `Exit`: Exit Month, Employee ID, Channel, Zone, Vintage, FOS/Non FOS(optional)
- `Business`: Month, Policy Ref, Channel, Zone, Vintage/Vintage Numeric, Product Mix Segment, Sub ID Code, Rated NB in Crs

This removes dependence on changing sheet/column names for new files. Legacy variations remain isolated in adapters.

## Scenario engine
Scenario values are temporary and never enter model training.
- Manpower: override Hiring, Exit or Attrition Rate at Channel+Zone+Month. The workforce identity is recalculated and Closing propagates into the next month's Opening.
- Business: override NOP or ATS by dimensions/month and compare baseline vs scenario Rated NB.

## LLM insights
The numeric forecast is produced by ML/formulas. The LLM receives computed facts only and explains trends, risks, gaps, scenario impacts and questions/actions for business. Configure Azure OpenAI via `.env`.

## Run
```bash
python -m venv ml_env
ml_env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Initial flow:
```text
POST /api/bootstrap
POST /api/train
POST /api/forecast
```

See `docs/architecture.md` for the architecture diagram.

## Actual + Forecast output
`src/services/exporter.py` produces one workbook containing 3-year Actual + future Forecast sheets for Closing, Exit, Opening, Hiring, Attrition, NOP and ATS. This keeps the business-facing output separate from raw ingestion/model objects.

## EDA / data quality API
`GET /api/data-quality` returns row counts, month coverage, missing values, unique categories and monthly totals for certified Manpower/NOP/ATS datasets.

## Download
After forecasting, `GET /api/export/latest` downloads `Actual_Forecast.xlsx` containing the combined 3-year actual history plus forecast sheets.

## NOP Productivity
Productivity is **derived**, not independently forecast:
`Productivity = NOP / Closing FLS` at Channel + Zone + Vintage + Month after aggregating Product Mix/Sub ID NOP. The exported workbook includes `Productivity A+F`.

## One-command non-Flask run
After installing requirements, run `run_pipeline.bat` or:
```bash
python scripts/bootstrap_train_forecast.py
```
