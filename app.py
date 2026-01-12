import sys
import types
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import duckdb
import joblib
import gradio as gr
from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.base import BaseEstimator, TransformerMixin
import threading, types, duckdb
import os
from importlib.resources import files


REPO_ID = "Carson-Shively/uber-fares"
REV = "main" 

ROOT = Path(__file__).resolve().parent

MODEL_PKL_PATH  = ROOT / "artifacts" / "stacked_fares.pkl"
FEATS_JSON_PATH = ROOT / "artifacts" / "feature_columns.json"
SQL_PKG = "uber_fares.data_layers.gold"
MACROS_SQL_FILE = "macros.sql"
ONLINE_SQL_FILE = "online.sql"

def _read_pkg_sql(pkg: str, filename: str) -> str:
    return (files(pkg) / filename).read_text(encoding="utf-8")



state = types.SimpleNamespace()
_duck_lock = threading.Lock()

def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

def stack_preds_to_df(Z, column_names):
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(column_names)
    if Z.shape[1] != n:
        Z = Z[:, -n:]
    return pd.DataFrame(Z, columns=column_names)

def to_df_all_fn(Z, *, n_bases, pred_names):
    Z = np.asarray(Z);  import pandas as pd
    return pd.DataFrame(Z[:, -n_bases:], columns=pred_names)

def add_quantile_widths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"lgbm_q10","lgbm_q90"}.issubset(df.columns):
        df["lgbm_qw"] = (df["lgbm_q90"] - df["lgbm_q10"]).clip(lower=0.0)
    if {"xgb_q10","xgb_q90"}.issubset(df.columns):
        df["xgb_qw"] = (df["xgb_q90"] - df["xgb_q10"]).clip(lower=0.0)
    return df

def add_pred_mean_gmd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    a = df["pred_lgbm"].to_numpy(); b = df["pred_xgb"].to_numpy()
    df["pred_mean"] = (a + b) / 2.0
    df["gmd_pred"] = np.abs(a - b)
    return df

def select_meta_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["pred_lgbm","pred_xgb","lgbm_qw","xgb_qw","gmd_pred","pred_mean"] if c in df.columns]
    return df[keep]

class TapDF(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.last_ = X
        return self
    def transform(self, X):
        self.last_ = X
        return X

m = sys.modules.get('__main__') or types.ModuleType('__main__')
sys.modules['__main__'] = m
for name, fn in {
    'X_to_float32': X_to_float32,
    'to_df_all_fn': to_df_all_fn,
    'add_quantile_widths': add_quantile_widths,
    'add_pred_mean_gmd': add_pred_mean_gmd,
    'select_meta_cols': select_meta_cols,
}.items():
    setattr(m, name, fn)



def init_connection(con) -> None:
    con.execute(_read_pkg_sql(SQL_PKG, MACROS_SQL_FILE))


def load_model_and_schema():
    try:
        model = joblib.load(MODEL_PKL_PATH)
        feature_columns = json.loads(FEATS_JSON_PATH.read_text(encoding="utf-8"))
        return model, feature_columns
    except Exception as e:
        raise RuntimeError("Failed model/schema load") from e


def init_app():
    state.con = duckdb.connect()
    init_connection(state.con)
    state.MODEL, state.FEATURE_COLUMNS = load_model_and_schema()
    state.FEATURE_COLUMNS = tuple(state.FEATURE_COLUMNS)
    with _duck_lock:
        state.con.execute(_read_pkg_sql(SQL_PKG, ONLINE_SQL_FILE))


def collect_raw_inputs(pc, dt, plon, plat, dlon, dlat):
    raw = {
        "passenger_count":   pc, 
        "pickup_datetime":   dt, 
        "pickup_longitude":  plon,
        "pickup_latitude":   plat,
        "dropoff_longitude": dlon,
        "dropoff_latitude":  dlat,
    }
    return raw, "Collected raw inputs. (Not validated yet.)"

class OnlineRequired(BaseModel):
    model_config = ConfigDict(strict=True) 
    passenger_count: float
    pickup_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float

def run_gold(df: pd.DataFrame) -> pd.DataFrame:
    row = df.iloc[0]

    anchor_ts = row.get("anchor_ts", row.get("pickup_datetime"))

    pc_raw = row.get("passenger_count", row.get("passenger_count_clean", 0.0))
    pc = float(pc_raw) if pd.notna(pc_raw) else 0.0

    plat = float(row["pickup_latitude"])
    plon = float(row["pickup_longitude"])
    dlat = float(row["dropoff_latitude"])
    dlon = float(row["dropoff_longitude"])

    with _duck_lock:
        return state.con.execute(
            "SELECT * FROM online_q_all(?::TIMESTAMPTZ, ?::DOUBLE, ?::DOUBLE, ?::DOUBLE, ?::DOUBLE, ?::DOUBLE)",
            [anchor_ts, pc, plat, plon, dlat, dlon],
        ).fetchdf()

def make_one_row_df(payload) -> pd.DataFrame:
    return pd.DataFrame([payload.model_dump()])

def _prepare_X_for_model(X_gold: pd.DataFrame) -> pd.DataFrame:
    cols = list(state.FEATURE_COLUMNS)  
    missing = [c for c in cols if c not in X_gold.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return X_gold.loc[:, cols]

def predict_from_raw(pc, dt, plon, plat, dlon, dlat) -> float:
    raw, _ = collect_raw_inputs(pc, dt, plon, plat, dlon, dlat)
    try:
        payload = OnlineRequired.model_validate(raw)   
    except ValidationError as e:
        raise gr.Error(e.errors()[0]["msg"])

    X_gold = run_gold(make_one_row_df(payload))
    print("X_gold columns:", list(X_gold.columns))

    X = _prepare_X_for_model(X_gold)
    yhat = float(state.MODEL.predict(X)[0])
    return round(yhat, 2)

def stamp_and_predict(pc, plon, plat, dlon, dlat):
    ts = datetime.now(timezone.utc)  
    return predict_from_raw(pc, ts, plon, plat, dlon, dlat)

PC_MIN, PC_MAX = 1, 6
LON_MIN, LON_MAX = -75, -72
LAT_MIN, LAT_MAX = 40, 42

with gr.Blocks(title="NYC Taxi Fare Predictor") as demo:
    gr.Markdown("# NYC Taxi Fare Predictor")
    pc = gr.Slider(
        PC_MIN, PC_MAX, step=1,
        label=f"Passengers ({PC_MIN}–{PC_MAX})",
    )
    plon = gr.Number(
        label=f"Pickup longitude [{LON_MIN}, {LON_MAX}]",
        minimum=LON_MIN, maximum=LON_MAX, precision=6,
    )
    plat = gr.Number(
        label=f"Pickup latitude [{LAT_MIN}, {LAT_MAX}]",
        minimum=LAT_MIN, maximum=LAT_MAX, precision=6,
    )
    dlon = gr.Number(
        label=f"Dropoff longitude [{LON_MIN}, {LON_MAX}]",
        minimum=LON_MIN, maximum=LON_MAX, precision=6,
    )
    dlat = gr.Number(
        label=f"Dropoff latitude [{LAT_MIN}, {LAT_MAX}]",
        minimum=LAT_MIN, maximum=LAT_MAX, precision=6,
    )

    btn_predict = gr.Button("Predict Fare")
    yhat_out = gr.Number(label="Estimated Fare ($)", interactive=False)

    btn_predict.click(
        stamp_and_predict,
        inputs=[pc, plon, plat, dlon, dlat],
        outputs=[yhat_out],
    )

if __name__ == "__main__":
    init_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )