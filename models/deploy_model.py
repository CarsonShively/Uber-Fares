import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
import json
from joblib import dump
from huggingface_hub import hf_hub_download
from pathlib import Path

train_data = hf_hub_download(
        repo_id="Carson-Shively/uber-fares",
        filename="gold_uf",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

X = df.drop(columns=["label", "pickup_datetime"])
y = df["label"]


def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

ensure_float32 = FunctionTransformer(
    X_to_float32, validate=False, feature_names_out="one-to-one"
)

lgbm_model = lgb.LGBMRegressor(
    objective="gamma",
    metric="rmsle",   
    n_estimators=1536,
    learning_rate=0.011455971173294806,
    max_depth=9,
    num_leaves=102,                  
    min_split_gain=0.045754591757354676,
    min_child_samples=65,
    min_child_weight=0.7572941145680039,
    reg_lambda=1.0137565838511768,
    reg_alpha=0.4270391365932309,
    subsample=0.8966012485934016,
    subsample_freq=3,
    colsample_bytree=0.6763000379498749,
    max_bin=488,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)

xgb_model = xgb.XGBRegressor(
    objective="reg:gamma",
    eval_metric="rmsle",
    n_estimators=1722,
    learning_rate=0.01748081755915497,
    max_depth=7,
    min_child_weight=0.04553167349901099,
    subsample=0.8914920536901483,
    colsample_bytree=0.6726178383908462,
    gamma=0.022060794038152958,   
    reg_lambda=7.519941123236492,
    reg_alpha=1.8212239178825262,
    max_bin=469,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

lgbm_q10 = lgb.LGBMRegressor(
    objective="quantile", alpha=0.10, metric="quantile",
    n_estimators=1049,
    learning_rate=0.04738897994799516,
    max_depth=9,
    num_leaves=318,
    min_split_gain=0.04680559213273095,
    min_child_samples=56,
    min_child_weight=0.0017073967431528124,
    reg_lambda=7.5504986209563665,
    reg_alpha=1.2022300234864176,
    subsample=0.9562108866694068,
    subsample_freq=0,
    colsample_bytree=0.939468448256698,
    max_bin=468,
    n_jobs=-1, random_state=42, verbosity=-1
)

lgbm_q90 = lgb.LGBMRegressor(
    objective="quantile", alpha=0.90, metric="quantile",
    n_estimators=1728,
    learning_rate=0.036264212405738856,
    max_depth=7,
    num_leaves=474,
    min_split_gain=0.02654775061557585,
    min_child_samples=63,
    min_child_weight=0.0015167330688076208,
    reg_lambda=0.02861167865082195,
    reg_alpha=0.777354579378964,
    subsample=0.8907023547660844,
    subsample_freq=6,
    colsample_bytree=0.7248636643427562,
    max_bin=327,
    n_jobs=-1, random_state=42, verbosity=-1
)

xgb_q10 = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.10,
    n_estimators=968,
    learning_rate=0.0518695140025758,
    max_depth=10,
    min_child_weight=5.727904470799623,
    subsample=0.9842241025641474,
    colsample_bytree=0.8092649925838797,
    gamma=0.27656227050693505,      
    reg_lambda=0.002489955937346358,
    reg_alpha=0.3919657248382904,
    max_bin=267,
    tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0
)

xgb_q90 = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.90,
    n_estimators=733,
    learning_rate=0.020165400229396,
    max_depth=5,
    min_child_weight=0.0010521761868451127,
    subsample=0.9723192142682251,
    colsample_bytree=0.847400070346666,
    gamma=0.21870215041229618,   
    reg_lambda=2.838382119353614,
    reg_alpha=0.14808930346818072,
    max_bin=348,
    tree_method="hist",
    n_jobs=-1, random_state=42, verbosity=0
)

base_learners = [
    ("lgbm",     lgbm_model),  
    ("xgb",      xgb_model),   
    ("lgbm_q10", lgbm_q10),
    ("lgbm_q90", lgbm_q90),
    ("xgb_q10",  xgb_q10),
    ("xgb_q90",  xgb_q90),
]


inner_cv = KFold(n_splits=5, shuffle=False)

names_in  = [name for name, _ in base_learners]
pred_names = [f"pred_{n}" if n in ("lgbm","xgb") else n for n in names_in]
n_bases    = len(pred_names)


def stack_preds_to_df(Z, column_names):
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(column_names)
    if Z.shape[1] != n:
        Z = Z[:, -n:]
    return pd.DataFrame(Z, columns=column_names)

to_df_all = FunctionTransformer(
    stack_preds_to_df, kw_args={"column_names": pred_names}, validate=False
)

def add_quantile_widths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"lgbm_q10","lgbm_q90"}.issubset(df.columns):
        df["lgbm_qw"] = (df["lgbm_q90"] - df["lgbm_q10"]).clip(lower=0.0)
    if {"xgb_q10","xgb_q90"}.issubset(df.columns):
        df["xgb_qw"] = (df["xgb_q90"] - df["xgb_q10"]).clip(lower=0.0)
    return df
add_qw = FunctionTransformer(add_quantile_widths, validate=False)

def add_pred_mean_gmd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    a = df["pred_lgbm"].to_numpy()
    b = df["pred_xgb"].to_numpy()
    df["pred_mean"] = (a + b) / 2.0
    df["gmd_pred"] = np.abs(a - b)  
    return df

add_feats_point = FunctionTransformer(add_pred_mean_gmd, validate=False)

def select_meta_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in [
        "pred_lgbm", "pred_xgb",
        "lgbm_qw", "xgb_qw",
        "gmd_pred", "pred_mean",
    ] if c in df.columns]
    return df[keep]

keep_meta = FunctionTransformer(select_meta_cols, validate=False)


class TapDF(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.last_ = X
        return self
    def transform(self, X):
        self.last_ = X
        return X

meta_core = HistGradientBoostingRegressor(
    loss="squared_error", max_depth=2, learning_rate=0.03,
    max_iter=800, min_samples_leaf=40, l2_regularization=0.2,
    max_bins=64, early_stopping=True, validation_fraction=0.12,
    n_iter_no_change=20, random_state=42
)
meta_model = TransformedTargetRegressor(
    regressor=meta_core, func=np.log1p, inverse_func=np.expm1
)

final_est = Pipeline([
    ("to_df_all",   to_df_all), 
    ("add_point",   add_feats_point),
    ("add_qwidths", add_qw),         
    ("keep_meta",   keep_meta),
    ("tap",         TapDF()),        
    ("meta",        meta_model), 
])

stacked_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=final_est,
    passthrough=False,
    cv=inner_cv,
    n_jobs=-1,
)

full_pipeline = Pipeline([
    ("to_float32", ensure_float32),
    ("model", stacked_model)
]).set_output(transform="pandas")

full_pipeline.fit(X, y)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

dump(full_pipeline, ARTIFACTS_DIR / "stacked_fares.pkl", compress=3)

with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
