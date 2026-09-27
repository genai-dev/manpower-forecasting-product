from src.services.pipeline import ForecastingService

svc=ForecastingService()
print("1) Bootstrapping legacy 3-year actuals...")
print(svc.bootstrap_legacy()["dataset_ids"])
print("2) Training models...")
print(svc.train()["model_version"])
print("3) Forecasting Apr-26 -> Mar-27...")
print(svc.forecast("2026-04-01",12))
print("4) Exporting Actual + Forecast workbook...")
print(svc.export_latest())
