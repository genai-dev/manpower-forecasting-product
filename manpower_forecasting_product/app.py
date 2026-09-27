from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from src.services.pipeline import ForecastingService
from src.scenario.engine import apply_business_scenario, apply_manpower_scenario
from src.insights.llm import deterministic_summary, generate_llm_insights

app=Flask(__name__)
app.config["SECRET_KEY"]=os.getenv("SECRET_KEY","dev-change-me")
service=ForecastingService()
UPLOAD_DIR=service.repo.root/"uploads"; UPLOAD_DIR.mkdir(parents=True,exist_ok=True)

@app.get("/")
def home(): return render_template("index.html")

@app.get("/health")
def health(): return jsonify({"status":"ok"})

@app.post("/api/bootstrap")
def bootstrap():
    return jsonify(service.bootstrap_legacy())

@app.post("/api/upload/monthly")
def upload_monthly():
    if "file" not in request.files: return jsonify({"error":"file is required"}),400
    f=request.files["file"]; name=secure_filename(f.filename or "monthly.xlsx"); path=UPLOAD_DIR/name; f.save(path)
    return jsonify({"uploaded":name,"dataset_ids":service.ingest_monthly_upload(path)})

@app.post("/api/train")
def train(): return jsonify(service.train())

@app.post("/api/forecast")
def forecast():
    body=request.get_json(silent=True) or {}
    return jsonify(service.forecast(body.get("start","2026-04-01"),int(body.get("periods",12))))


@app.get("/api/data-quality")
def data_quality(): return jsonify(service.data_quality())

@app.get("/api/export/latest")
def export_latest():
    path=service.export_latest()
    return send_file(path,as_attachment=True,download_name="Actual_Forecast.xlsx")

@app.post("/api/scenario/manpower")
def scenario_manpower():
    body=request.get_json(force=True); forecast_id=body["hiring_forecast_id"]
    baseline=service.repo.load_forecast(forecast_id); out=apply_manpower_scenario(baseline,body.get("overrides",[])); sid,path=service.repo.save_scenario("manpower_scenario",out,body)
    return jsonify({"scenario_id":sid,"path":str(path)})

@app.post("/api/scenario/business")
def scenario_business():
    body=request.get_json(force=True); nop=service.repo.load_forecast(body["nop_forecast_id"]); ats=service.repo.load_forecast(body["ats_forecast_id"]); out=apply_business_scenario(nop,ats,body.get("overrides",[])); sid,path=service.repo.save_scenario("business_scenario",out,body)
    return jsonify({"scenario_id":sid,"path":str(path)})

@app.post("/api/insights")
def insights():
    body=request.get_json(force=True); mp=service.repo.load_forecast(body["manpower_forecast_id"]) if body.get("manpower_forecast_id") else None; scenario=service.repo.load_scenario(body["scenario_forecast_id"]) if body.get("scenario_forecast_id") else None; nop=service.repo.load_forecast(body["nop_forecast_id"]) if body.get("nop_forecast_id") else None; ats=service.repo.load_forecast(body["ats_forecast_id"]) if body.get("ats_forecast_id") else None
    facts=deterministic_summary(mp,scenario,nop,ats); return jsonify(generate_llm_insights(facts))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")),debug=False)
