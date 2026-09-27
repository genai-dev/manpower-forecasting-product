from __future__ import annotations

import os
from pathlib import Path
import joblib
import pandas as pd

from config.business_rules import BUSINESS_FILES, MANPOWER_FY24_25_FILE, MANPOWER_FY26_FILE
from src.ingestion.manpower import build_manpower_history
from src.ingestion.business import build_business_history
from src.ingestion.standard_upload import read_standard_monthly_workbook
from src.models.manpower_model import train_manpower_models, forecast_manpower
from src.models.business_model import train_business_models, BUSINESS_DIMS
from src.storage.repository import Repository
from src.validation.data_quality import validate_monthly_panel, manpower_reconciliation, eda_summary
from src.services.exporter import export_actual_forecast


def resolve_excel_path(path: Path):
    if path.exists(): return path
    for ext in [".xlsx",".xlsb",".xls"]:
        candidate=path.with_suffix(ext)
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"File not found in xlsx/xlsb variants: {path}")


class ForecastingService:
    def __init__(self, data_root=None):
        root=Path(data_root or os.getenv("DATA_ROOT", r"D:\Business planning\ForecastingProductData"))
        self.repo=Repository(root)
        self.model_dir=root/"models"; self.model_dir.mkdir(parents=True,exist_ok=True)

    def bootstrap_legacy(self):
        mp1=resolve_excel_path(Path(MANPOWER_FY24_25_FILE)); mp2=resolve_excel_path(Path(MANPOWER_FY26_FILE))
        business_specs=[{"fy":x["fy"],"path":resolve_excel_path(Path(x["path"]))} for x in BUSINESS_FILES]
        manpower,hiring=build_manpower_history(mp1,mp2)
        nop,ats=build_business_history(business_specs)
        ids={}
        ids["manpower"],_=self.repo.save_dataset("manpower_base",manpower,{"period":"Apr-23 to Mar-26","status":"CERTIFIED_CANDIDATE"})
        ids["hiring"],_=self.repo.save_dataset("hiring_base",hiring,{"period":"Apr-23 to Mar-26"})
        ids["nop"],_=self.repo.save_dataset("nop_base",nop,{"period":"Apr-23 to Mar-26"})
        ids["ats"],_=self.repo.save_dataset("ats_base",ats,{"period":"Apr-23 to Mar-26"})
        reports={
            "manpower_validation":validate_monthly_panel(manpower,["Channel","Zone","Vintage"],["Closing","Exit"],"2023-04-01","2026-03-01").to_dict("records"),
            "reconciliation":manpower_reconciliation(manpower,hiring).to_dict("records"),
            "nop_validation":validate_monthly_panel(nop,BUSINESS_DIMS,["NOP"],"2023-04-01","2026-03-01").to_dict("records"),
            "ats_validation":validate_monthly_panel(ats,BUSINESS_DIMS,["ATS"],"2023-04-01","2026-03-01").to_dict("records"),
        }
        return {"dataset_ids":ids,"reports":reports}

    @staticmethod
    def _merge_replace_months(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]):
        months=pd.to_datetime(new["Month"]).dt.to_period("M").dt.to_timestamp().unique()
        old=old[~pd.to_datetime(old["Month"]).dt.to_period("M").dt.to_timestamp().isin(months)].copy()
        return pd.concat([old,new],ignore_index=True).sort_values(keys+["Month"]).reset_index(drop=True)

    def ingest_monthly_upload(self, workbook_path: Path):
        frames=read_standard_monthly_workbook(workbook_path); result={}
        if "closing" in frames or "exit" in frames:
            current_id=self.repo.latest_dataset("manpower_base"); current=self.repo.load_dataset(current_id) if current_id else pd.DataFrame()
            keys=["Channel","Zone","Vintage","Month"]
            # Build new combined manpower rows for uploaded months.
            c=frames.get("closing",pd.DataFrame(columns=keys+["Closing"])); e=frames.get("exit",pd.DataFrame(columns=keys+["Exit"]))
            n=c.merge(e,on=keys,how="outer"); n["Closing"]=n["Closing"].fillna(0); n["Exit"]=n["Exit"].fillna(0)
            # Opening/Hiring need previous certified month; easiest robust path is recalc after merge.
            hist=pd.concat([current[keys+["Closing","Exit"]] if not current.empty else pd.DataFrame(columns=keys+["Closing","Exit"]),n],ignore_index=True)
            hist=hist.groupby(keys,observed=True,as_index=False)[["Closing","Exit"]].sum(); prev=hist[["Channel","Zone","Vintage","Month","Closing"]].copy(); prev["Month"]=pd.to_datetime(prev["Month"])+pd.offsets.MonthBegin(1); prev=prev.rename(columns={"Closing":"Opening"}); hist=hist.merge(prev,on=keys,how="left"); hist["AttritionRate"]=hist["Exit"]/hist["Opening"].where(hist["Opening"].gt(0))
            result["manpower"],_=self.repo.save_dataset("manpower_base",hist,{"source":"monthly_upload","status":"VALIDATED_PENDING_TRAIN"})
            hz=hist.groupby(["Channel","Zone","Month"],observed=True,as_index=False).agg(Opening=("Opening",lambda x:x.sum(min_count=1)),Closing=("Closing","sum"),Exit=("Exit","sum")); hz["Hiring"]=hz["Closing"]-hz["Opening"]+hz["Exit"]; result["hiring"],_=self.repo.save_dataset("hiring_base",hz,{"source":"monthly_upload"})
        for k,name,dims in [("nop","nop_base",BUSINESS_DIMS),("ats","ats_base",BUSINESS_DIMS)]:
            if k in frames:
                latest=self.repo.latest_dataset(name); old=self.repo.load_dataset(latest) if latest else pd.DataFrame(columns=frames[k].columns)
                merged=self._merge_replace_months(old,frames[k],dims); result[k],_=self.repo.save_dataset(name,merged,{"source":"monthly_upload","status":"VALIDATED_PENDING_TRAIN"})
        return result

    def train(self):
        mp_id=self.repo.latest_dataset("manpower_base"); nop_id=self.repo.latest_dataset("nop_base"); ats_id=self.repo.latest_dataset("ats_base")
        if not all([mp_id,nop_id,ats_id]): raise RuntimeError("Bootstrap/ingest datasets before training")
        mp=self.repo.load_dataset(mp_id); nop=self.repo.load_dataset(nop_id); ats=self.repo.load_dataset(ats_id)
        closing,exit_model,mval=train_manpower_models(mp); nop_model,ats_model,bval=train_business_models(nop,ats)
        model_version=pd.Timestamp.utcnow().strftime("model_%Y%m%dT%H%M%SZ")
        paths={
            "closing":self.model_dir/f"{model_version}_closing.joblib",
            "exit":self.model_dir/f"{model_version}_exit.joblib",
            "nop":self.model_dir/f"{model_version}_nop.joblib",
            "ats":self.model_dir/f"{model_version}_ats.joblib",
        }
        for obj,p in [(closing,paths["closing"]),(exit_model,paths["exit"]),(nop_model,paths["nop"]),(ats_model,paths["ats"])]: joblib.dump(obj,p)
        pd.concat([mval,bval],ignore_index=True).to_csv(self.model_dir/f"{model_version}_validation.csv",index=False)
        (self.model_dir/"LATEST").write_text(model_version)
        return {"model_version":model_version,"validation":pd.concat([mval,bval],ignore_index=True).to_dict("records")}


    def data_quality(self):
        result={}
        for name,dims,targets in [("manpower_base",["Channel","Zone","Vintage"],["Closing","Exit","Opening","AttritionRate"]),("nop_base",BUSINESS_DIMS,["NOP"]),("ats_base",BUSINESS_DIMS,["ATS"])]:
            did=self.repo.latest_dataset(name)
            if did:
                df=self.repo.load_dataset(did); summary,monthly=eda_summary(df,dims,targets); result[name]={"dataset_id":did,"summary":summary,"monthly_totals":monthly.to_dict("records")}
        return result

    def export_latest(self):
        ids={k:self.repo.latest_dataset(k) for k in ["manpower_base","hiring_base","nop_base","ats_base"]}
        if not all(ids.values()): raise RuntimeError("Missing base datasets")
        mp=self.repo.load_dataset(ids["manpower_base"]); hiring=self.repo.load_dataset(ids["hiring_base"]); nop=self.repo.load_dataset(ids["nop_base"]); ats=self.repo.load_dataset(ids["ats_base"])
        # latest forecast by prefix
        def latest_fc(prefix):
            files=sorted((self.repo.root/"forecasts").glob(f"{prefix}_*.csv"));
            if not files: raise RuntimeError(f"No {prefix} found; run forecast first")
            df=pd.read_csv(files[-1]);
            if "Month" in df: df["Month"]=pd.to_datetime(df["Month"]);
            return df
        mpfc=latest_fc("manpower_forecast"); hfc=latest_fc("hiring_forecast"); nfc=latest_fc("nop_forecast"); afc=latest_fc("ats_forecast")
        out=self.repo.root/"forecasts"/"Latest_Actual_Forecast.xlsx"
        validation=None
        latest_model=(self.model_dir/"LATEST").read_text().strip() if (self.model_dir/"LATEST").exists() else None
        vp=self.model_dir/f"{latest_model}_validation.csv" if latest_model else None
        if vp and vp.exists(): validation=pd.read_csv(vp)
        export_actual_forecast(out,mp,hiring,nop,ats,mpfc,hfc,nfc,afc,validation)
        return out

    def forecast(self, start="2026-04-01", periods=12):
        latest=(self.model_dir/"LATEST").read_text().strip(); mp=self.repo.load_dataset(self.repo.latest_dataset("manpower_base")); nop=self.repo.load_dataset(self.repo.latest_dataset("nop_base")); ats=self.repo.load_dataset(self.repo.latest_dataset("ats_base"))
        closing=joblib.load(self.model_dir/f"{latest}_closing.joblib"); exit_model=joblib.load(self.model_dir/f"{latest}_exit.joblib"); nop_model=joblib.load(self.model_dir/f"{latest}_nop.joblib"); ats_model=joblib.load(self.model_dir/f"{latest}_ats.joblib")
        mp_detail,hiring=forecast_manpower(mp,closing,exit_model,start,periods); nop_fc=nop_model.forecast(nop[BUSINESS_DIMS+["Month","NOP"]],start,periods); ats_fc=ats_model.forecast(ats[BUSINESS_DIMS+["Month","ATS"]],start,periods)
        ids={}
        ids["manpower"],_=self.repo.save_forecast("manpower_forecast",mp_detail,{"model_version":latest})
        ids["hiring"],_=self.repo.save_forecast("hiring_forecast",hiring,{"model_version":latest})
        ids["nop"],_=self.repo.save_forecast("nop_forecast",nop_fc,{"model_version":latest})
        ids["ats"],_=self.repo.save_forecast("ats_forecast",ats_fc,{"model_version":latest})
        return {"forecast_ids":ids,"model_version":latest}
