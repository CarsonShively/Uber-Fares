import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
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

if (yf <= 0).any():
    raise ValueError("Gamma objective requires strictly positive targets.")

tscv = TimeSeriesSplit(n_splits=3, gap=1)

def objective(trial):
    params = {
        "objective": "reg:gamma",
        "eval_metric": "rmsle",
        "eta": trial.suggest_float("learning_rate", 5e-4, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.85, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
        "gamma": trial.suggest_float("min_split_loss", 0.0, 0.3),
        "lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "tree_method": "hist",
        "max_bin": trial.suggest_int("max_bin", 256, 512),
        "seed": 42,
    }
    num_boost_round = trial.suggest_int("n_estimators", 600, 2000)
    early_stopping_rounds = 200

    scores = []
    for tr_idx, va_idx in tscv.split(Xf):
        X_tr_i, X_va_i = Xf.iloc[tr_idx].values, Xf.iloc[va_idx].values
        y_tr_i, y_va_i = yf.iloc[tr_idx].values,  yf.iloc[va_idx].values

        dtr = xgb.DMatrix(X_tr_i, label=y_tr_i)
        dva = xgb.DMatrix(X_va_i, label=y_va_i)

        bst = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        best_it = bst.best_iteration
        if best_it is None:
            best_it = num_boost_round - 1  

        y_hat = bst.predict(dva, iteration_range=(0, best_it + 1))
        scores.append(rmsle(y_va_i, y_hat)) 
    return float(np.mean(scores))

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgb_gamma_rmsle_tscv")
study.optimize(objective, n_trials=25, show_progress_bar=True)

best_params = study.best_params
best_score = study.best_value

print("Best RMSLE (CV):", best_score)
print("Best Params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")