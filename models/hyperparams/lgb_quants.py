import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/uber-fares",
        filename="data/gold/gold_uf.parquet",
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

Xf = to_float32(X_train.copy())
yf = y_train.astype(float)

tscv = TimeSeriesSplit(n_splits=3, gap=1)

def tune_lgbm_quantile(alpha: float, n_trials: int = 10, seed: int = 42):
    def objective(trial):
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "n_estimators": trial.suggest_int("n_estimators", 600, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.85, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
            "max_bin": trial.suggest_int("max_bin", 255, 511),
            "n_jobs": -1, "random_state": seed, "verbosity": -1,
            "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed,
        }

        if params["max_depth"] is not None and params["max_depth"] > 0:
            allowed_max_leaves = 1 << params["max_depth"]
            params["num_leaves"] = int(np.clip(params["num_leaves"], 2, allowed_max_leaves))

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
                eval_metric="quantile",                
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            scores.append(m.best_score_["valid_0"]["quantile"])

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"lgbm_q{int(alpha*100)}_tscv")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
  
    best_params.update({
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "n_jobs": -1, "random_state": seed, "verbosity": -1
    })

    print(f"[alpha={alpha:.2f}] Best pinball loss (CV):", study.best_value)
    print(f"[alpha={alpha:.2f}] Best Params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return LGBMRegressor(**best_params)

lgbm_q10 = tune_lgbm_quantile(alpha=0.10, n_trials=10, seed=42)
lgbm_q90 = tune_lgbm_quantile(alpha=0.90, n_trials=10, seed=42)