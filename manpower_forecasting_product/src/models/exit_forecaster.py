from __future__ import annotations

import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DIMS = ["Channel", "Zone", "Vintage"]
LEVELS = {"CZ": ["Channel", "Zone"], "CV": ["Channel", "Vintage"], "CH": ["Channel"], "ZN": ["Zone"], "VT": ["Vintage"]}
BASE_NUM = ["MonthNo", "Quarter", "TimeIndex", "Lag1", "Lag2", "Lag3", "Lag6", "Lag12", "Rolling3", "Rolling6", "Rolling12", "RollingStd3", "YoYChange"]
EXTRA = ["ExitSum3", "ExitSum6", "ExitSum12", "ExitNonZero6", "ExitNonZero12", "ExitTrend12"] + [
    f"{p}_{m}" for p in ["CZ", "CV", "CH", "ZN", "VT", "ALL"] for m in ["Lag1", "Lag3", "Lag12", "Roll3", "Roll6", "Roll12", "Trend12", "YoY"]
]
NUM = BASE_NUM + EXTRA


def slope(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if len(a) < 2: return 0.0
    x = np.arange(len(a), dtype=float); x -= x.mean(); den = np.sum(x*x)
    return float(np.sum(x * (a-a.mean())) / den) if den else 0.0


def _wape(a, p):
    a=np.asarray(a,float); p=np.asarray(p,float); den=np.abs(a).sum()
    return np.abs(a-p).sum()/den if den else np.nan


def _metrics(a,p):
    return {"MAE":mean_absolute_error(a,p),"RMSE":math.sqrt(mean_squared_error(a,p)),"WAPE":_wape(a,p)}


def _level_features(df, cols, prefix):
    m = df.groupby(cols+["Month"], observed=True, as_index=False)["Exit"].sum().sort_values(cols+["Month"])
    g = m.groupby(cols, observed=True)["Exit"]
    m[f"{prefix}_Lag1"] = g.shift(1); m[f"{prefix}_Lag3"] = g.shift(3); m[f"{prefix}_Lag12"] = g.shift(12)
    m[f"{prefix}_Roll3"] = g.transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean())
    m[f"{prefix}_Roll6"] = g.transform(lambda x:x.shift(1).rolling(6,min_periods=1).mean())
    m[f"{prefix}_Roll12"] = g.transform(lambda x:x.shift(1).rolling(12,min_periods=1).mean())
    m[f"{prefix}_Trend12"] = g.transform(lambda x:x.shift(1).rolling(12,min_periods=4).apply(slope, raw=True))
    m[f"{prefix}_YoY"] = m[f"{prefix}_Lag1"] - m[f"{prefix}_Lag12"]
    keep=cols+["Month"]+[f"{prefix}_{z}" for z in ["Lag1","Lag3","Lag12","Roll3","Roll6","Roll12","Trend12","YoY"]]
    return m[keep]


class HierarchicalExitForecaster:
    def __init__(self):
        self.model=None; self.model_name=None; self.validation=None; self.min_month=None

    def _features(self, history):
        out=history.copy().sort_values(DIMS+["Month"]).reset_index(drop=True)
        out["Month"]=pd.to_datetime(out["Month"]).dt.to_period("M").dt.to_timestamp(); self.min_month=out["Month"].min()
        out["MonthNo"]=out["Month"].dt.month; out["Quarter"]=out["Month"].dt.quarter
        out["TimeIndex"]=(out["Month"].dt.year-self.min_month.year)*12+(out["Month"].dt.month-self.min_month.month)
        g=out.groupby(DIMS, observed=True)["Exit"]
        for lag in [1,2,3,6,12]: out[f"Lag{lag}"]=g.shift(lag)
        out["Rolling3"]=g.transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean())
        out["Rolling6"]=g.transform(lambda x:x.shift(1).rolling(6,min_periods=1).mean())
        out["Rolling12"]=g.transform(lambda x:x.shift(1).rolling(12,min_periods=1).mean())
        out["RollingStd3"]=g.transform(lambda x:x.shift(1).rolling(3,min_periods=2).std())
        out["YoYChange"]=out["Lag1"]-out["Lag12"]
        out["ExitSum3"]=g.transform(lambda x:x.shift(1).rolling(3,min_periods=1).sum())
        out["ExitSum6"]=g.transform(lambda x:x.shift(1).rolling(6,min_periods=1).sum())
        out["ExitSum12"]=g.transform(lambda x:x.shift(1).rolling(12,min_periods=1).sum())
        out["ExitNonZero6"]=g.transform(lambda x:x.shift(1).gt(0).rolling(6,min_periods=1).mean())
        out["ExitNonZero12"]=g.transform(lambda x:x.shift(1).gt(0).rolling(12,min_periods=1).mean())
        out["ExitTrend12"]=g.transform(lambda x:x.shift(1).rolling(12,min_periods=4).apply(slope, raw=True))
        for p,cols in LEVELS.items(): out=out.merge(_level_features(history,cols,p),on=cols+["Month"],how="left")
        ov=history.groupby("Month",as_index=False)["Exit"].sum().sort_values("Month"); s=ov["Exit"]
        ov["ALL_Lag1"]=s.shift(1); ov["ALL_Lag3"]=s.shift(3); ov["ALL_Lag12"]=s.shift(12)
        ov["ALL_Roll3"]=s.shift(1).rolling(3,min_periods=1).mean(); ov["ALL_Roll6"]=s.shift(1).rolling(6,min_periods=1).mean(); ov["ALL_Roll12"]=s.shift(1).rolling(12,min_periods=1).mean()
        ov["ALL_Trend12"]=s.shift(1).rolling(12,min_periods=4).apply(slope,raw=True); ov["ALL_YoY"]=ov["ALL_Lag1"]-ov["ALL_Lag12"]
        return out.merge(ov[["Month"]+[f"ALL_{z}" for z in ["Lag1","Lag3","Lag12","Roll3","Roll6","Roll12","Trend12","YoY"]]],on="Month",how="left")

    def _pipe(self, kind):
        prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),DIMS),("num","passthrough",NUM)])
        if kind=="poisson": est=HistGradientBoostingRegressor(loss="poisson",learning_rate=.045,max_iter=450,max_leaf_nodes=20,min_samples_leaf=15,l2_regularization=2.0,random_state=42)
        elif kind=="rf_tuned": est=RandomForestRegressor(n_estimators=700,random_state=42,n_jobs=-1,max_depth=14,min_samples_leaf=2,max_features=.65)
        else: est=RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1,min_samples_leaf=2,max_features="sqrt")
        return Pipeline([("preprocessing",prep),("model",est)])

    def fit(self, history, validation_months=6):
        f=self._features(history); md=f[f["Exit"].notna() & f["Lag12"].notna()].copy(); md[NUM]=md[NUM].replace([np.inf,-np.inf],np.nan).fillna(0)
        months=sorted(md["Month"].unique()); cut=pd.Timestamp(months[-validation_months]); tr=md[md["Month"]<cut]; va=md[md["Month"]>=cut]; xcols=DIMS+NUM
        rows=[]; kinds={}
        for name,kind in [("Hierarchical Random Forest","rf"),("Hierarchical RF Tuned","rf_tuned"),("Hierarchical Poisson HGB","poisson")]:
            m=self._pipe(kind); m.fit(tr[xcols],tr["Exit"]); p=np.clip(m.predict(va[xcols]),0,None); rows.append({"Target":"Exit","Model":name,**_metrics(va["Exit"],p)}); kinds[name]=kind
        rows.append({"Target":"Exit","Model":"Lag12 Baseline",**_metrics(va["Exit"],va["Lag12"].clip(lower=0))})
        self.validation=pd.DataFrame(rows).sort_values(["WAPE","MAE","RMSE"]).reset_index(drop=True); ml=self.validation[self.validation.Model!="Lag12 Baseline"]; self.model_name=str(ml.iloc[0].Model)
        self.model=self._pipe(kinds[self.model_name]); self.model.fit(md[xcols],md["Exit"]); return self.validation.copy()

    @staticmethod
    def _dict_val(d,m,lag): return d.get((m-pd.DateOffset(months=lag)).to_period("M").to_timestamp(),np.nan)
    @staticmethod
    def _recent(d,m,n): return [float(d.get((m-pd.DateOffset(months=i)).to_period("M").to_timestamp())) for i in range(n,0,-1) if pd.notna(d.get((m-pd.DateOffset(months=i)).to_period("M").to_timestamp(),np.nan))]
    def _hvals(self,d,m,p):
        l1=self._dict_val(d,m,1); l3=self._dict_val(d,m,3); l12=self._dict_val(d,m,12); r3=self._recent(d,m,3); r6=self._recent(d,m,6); r12=self._recent(d,m,12)
        return {f"{p}_Lag1":l1,f"{p}_Lag3":l3,f"{p}_Lag12":l12,f"{p}_Roll3":np.mean(r3) if r3 else np.nan,f"{p}_Roll6":np.mean(r6) if r6 else np.nan,f"{p}_Roll12":np.mean(r12) if r12 else np.nan,f"{p}_Trend12":slope(r12) if len(r12)>=2 else 0.0,f"{p}_YoY":l1-l12 if pd.notna(l1) and pd.notna(l12) else np.nan}

    def forecast(self, history, start, periods=12):
        h=history.copy(); h["Month"]=pd.to_datetime(h["Month"]).dt.to_period("M").dt.to_timestamp(); combos=h[DIMS].drop_duplicates().reset_index(drop=True)
        combo={k:{pd.Timestamp(m):float(v) for m,v in zip(g.Month,g.Exit)} for k,g in h.groupby(DIMS,observed=True)}
        def level_hist(cols):
            mon=h.groupby(cols+["Month"],observed=True,as_index=False).Exit.sum(); result={}
            for k,g in mon.groupby(cols,observed=True):
                if not isinstance(k,tuple): k=(k,)
                result[k]={pd.Timestamp(m):float(v) for m,v in zip(g.Month,g.Exit)}
            return result
        levels={p:level_hist(c) for p,c in LEVELS.items()}; ovm=h.groupby("Month",as_index=False).Exit.sum(); overall={pd.Timestamp(m):float(v) for m,v in zip(ovm.Month,ovm.Exit)}; outputs=[]
        for month in pd.date_range(pd.Timestamp(start),periods=periods,freq="MS"):
            rows=[]; keys=[]
            for vals in combos.itertuples(index=False,name=None):
                key=tuple(vals); d=combo.get(key,{}); l=lambda n:self._dict_val(d,month,n); r3=self._recent(d,month,3); r6=self._recent(d,month,6); r12=self._recent(d,month,12)
                row=dict(zip(DIMS,key)); row.update({"MonthNo":month.month,"Quarter":month.quarter,"TimeIndex":(month.year-self.min_month.year)*12+(month.month-self.min_month.month),"Lag1":l(1),"Lag2":l(2),"Lag3":l(3),"Lag6":l(6),"Lag12":l(12),"Rolling3":np.mean(r3) if r3 else np.nan,"Rolling6":np.mean(r6) if r6 else np.nan,"Rolling12":np.mean(r12) if r12 else np.nan,"RollingStd3":np.std(r3,ddof=1) if len(r3)>=2 else np.nan,"YoYChange":l(1)-l(12) if pd.notna(l(1)) and pd.notna(l(12)) else np.nan,"ExitSum3":np.sum(r3) if r3 else np.nan,"ExitSum6":np.sum(r6) if r6 else np.nan,"ExitSum12":np.sum(r12) if r12 else np.nan,"ExitNonZero6":np.mean(np.asarray(r6)>0) if r6 else 0.0,"ExitNonZero12":np.mean(np.asarray(r12)>0) if r12 else 0.0,"ExitTrend12":slope(r12) if len(r12)>=2 else 0.0})
                ch,z,v=key
                for p,lk in {"CZ":(ch,z),"CV":(ch,v),"CH":(ch,),"ZN":(z,),"VT":(v,)}.items(): row.update(self._hvals(levels[p].get(lk,{}),month,p))
                row.update(self._hvals(overall,month,"ALL")); rows.append(row); keys.append(key)
            X=pd.DataFrame(rows); X[NUM]=X[NUM].replace([np.inf,-np.inf],np.nan).fillna(0); pred=np.rint(np.clip(self.model.predict(X[DIMS+NUM]),0,None)); level_sums={p:{} for p in LEVELS}; total=0.0
            for key,y in zip(keys,pred):
                y=float(y); combo.setdefault(key,{})[month]=y; ch,z,v=key; total+=y
                for p,lk in {"CZ":(ch,z),"CV":(ch,v),"CH":(ch,),"ZN":(z,),"VT":(v,)}.items(): level_sums[p][lk]=level_sums[p].get(lk,0.0)+y
                outputs.append({"Channel":ch,"Zone":z,"Vintage":v,"Month":month,"ForecastExit":y})
            for p,sums in level_sums.items():
                for lk,val in sums.items(): levels[p].setdefault(lk,{})[month]=val
            overall[month]=total
        return pd.DataFrame(outputs)

    def save(self,path): joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
