from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


class Repository:
    def __init__(self, root: Path):
        self.root=Path(root)
        for name in ["datasets","models","forecasts","scenarios","reports"]:
            (self.root/name).mkdir(parents=True,exist_ok=True)

    @staticmethod
    def _version(prefix):
        return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def save_dataset(self, name: str, df: pd.DataFrame, metadata=None):
        v=self._version(name); path=self.root/"datasets"/f"{v}.csv"; df.to_csv(path,index=False)
        meta={"id":v,"name":name,"rows":len(df),"created_at":datetime.now(timezone.utc).isoformat(),**(metadata or {})}
        (self.root/"datasets"/f"{v}.json").write_text(json.dumps(meta,indent=2,default=str)); return v,path

    def load_dataset(self, dataset_id: str):
        p=self.root/"datasets"/f"{dataset_id}.csv"
        df=pd.read_csv(p)
        if "Month" in df: df["Month"]=pd.to_datetime(df["Month"])
        return df

    def latest_dataset(self, name: str):
        files=sorted((self.root/"datasets").glob(f"{name}_*.csv"))
        return files[-1].stem if files else None

    def save_forecast(self, name, df, metadata=None):
        v=self._version(name); p=self.root/"forecasts"/f"{v}.csv"; df.to_csv(p,index=False)
        (self.root/"forecasts"/f"{v}.json").write_text(json.dumps({"id":v,**(metadata or {})},indent=2,default=str)); return v,p

    def load_forecast(self, forecast_id):
        df=pd.read_csv(self.root/"forecasts"/f"{forecast_id}.csv")
        if "Month" in df: df["Month"]=pd.to_datetime(df["Month"])
        return df

    def save_scenario(self, name, df, request):
        v=self._version(name); p=self.root/"scenarios"/f"{v}.csv"; df.to_csv(p,index=False)
        (self.root/"scenarios"/f"{v}.json").write_text(json.dumps({"id":v,"request":request},indent=2,default=str)); return v,p

    def load_scenario(self, scenario_id):
        df=pd.read_csv(self.root/"scenarios"/f"{scenario_id}.csv")
        if "Month" in df: df["Month"]=pd.to_datetime(df["Month"])
        return df
