import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/uber-fares",
        filename="gold_uf",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

Xy = df.sort_values('pickup_datetime').reset_index(drop=True)

cut = Xy['pickup_datetime'].quantile(0.80)
train = Xy[Xy['pickup_datetime'] <  cut]
test  = Xy[Xy['pickup_datetime'] >= cut]

y_train = train['label'].astype(float)
y_test  = test['label'].astype(float)
X_train = train.drop(columns=['label','pickup_datetime'])
X_test  = test.drop(columns=['label','pickup_datetime'])

def to_float32(df):
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 1e-12, None))))

Xf = to_float32(X_train.copy())
yf = y_train.astype(float)

tscv = TimeSeriesSplit(n_splits=3, gap=1)

def objective(trial):
    params = {
        "objective": "gamma",
        "n_estimators": trial.suggest_int("n_estimators", 800, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 350),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.85, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "max_bin": trial.suggest_int("max_bin", 255, 511),
        "n_jobs": -1, "random_state": 42, "verbosity": -1,
        "bagging_seed": 42, "feature_fraction_seed": 42, "data_random_seed": 42,
    }
    if params["max_depth"] > 0:
        params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"] - 1)
    if params["subsample"] < 1.0 and params["subsample_freq"] == 0:
        params["subsample_freq"] = 1

    scores = []
    for tr_idx, va_idx in tscv.split(Xf):
        X_tr_i, X_va_i = Xf.iloc[tr_idx], Xf.iloc[va_idx]
        y_tr_i, y_va_i = yf.iloc[tr_idx],  yf.iloc[va_idx]
        m = LGBMRegressor(**params)
        m.fit(
            X_tr_i, y_tr_i,
            eval_set=[(X_va_i, y_va_i)],
            eval_metric="rmsle",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        y_hat = m.predict(X_va_i, num_iteration=m.best_iteration_ or None)
        scores.append(rmsle(y_va_i, y_hat))
    return float(np.mean(scores))

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lgbm_gamma_rmsle_tscv")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_score = study.best_value

print("Best RMSLE (CV):", best_score)
print("Best Params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Best RMSLE (CV): 0.2331382494704827
# Best Params:
#   n_estimators: 1536
#   learning_rate: 0.011455971173294806
#   max_depth: 9
#   num_leaves: 102
#   min_child_samples: 65
#   min_child_weight: 0.7572941145680039
#   min_split_gain: 0.045754591757354676
#   reg_lambda: 1.0137565838511768
#   reg_alpha: 0.4270391365932309
#   subsample: 0.8966012485934016
#   subsample_freq: 3
#   colsample_bytree: 0.6763000379498749
#   max_bin: 488