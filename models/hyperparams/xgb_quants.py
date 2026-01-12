import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

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

Xf = to_float32(X_train.copy())
yf = y_train.astype(float)

tscv = TimeSeriesSplit(n_splits=3, gap=1)

def pinball_loss(y_true, y_pred, alpha: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))

def tune_xgb_quantile(alpha: float, n_trials: int = 10, seed: int = 42):
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 5e-4, 0.06, log=True)
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": alpha,
            "eta": lr,
            "learning_rate": lr,
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.85, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
            "gamma": trial.suggest_float("min_split_loss", 0.0, 0.3),
            "lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "tree_method": "hist",
            "max_bin": trial.suggest_int("max_bin", 256, 512),
            "seed": seed,
        }
        num_boost_round = trial.suggest_int("n_estimators", 600, 1800)
        early_stopping_rounds = 200

        fold_losses, fold_best_rounds = [], []

        for tr_idx, va_idx in tscv.split(Xf):
            X_tr, X_va = Xf.iloc[tr_idx].values, Xf.iloc[va_idx].values
            y_tr, y_va = yf.iloc[tr_idx].values,  yf.iloc[va_idx].values

            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)

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
            fold_losses.append(pinball_loss(y_va, y_hat, alpha))
            fold_best_rounds.append(int(best_it + 1))

        trial.set_user_attr("best_rounds_median", int(np.median(fold_best_rounds)))
        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"xgb_q{int(alpha*100)}_tscv")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_n_estimators = study.best_trial.user_attrs.get(
        "best_rounds_median", best.get("n_estimators", 1000)
    )

    best_est = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        learning_rate=best["learning_rate"],
        max_depth=best["max_depth"],
        min_child_weight=best["min_child_weight"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        gamma=best["min_split_loss"],
        reg_lambda=best["reg_lambda"],
        reg_alpha=best["reg_alpha"],
        tree_method="hist",
        max_bin=best["max_bin"],
        n_estimators=int(best_n_estimators),
        random_state=seed,
        n_jobs=-1,
    )

    print(f"[alpha={alpha:.2f}] Best pinball loss (CV): {study.best_value:.6f}")
    print(f"[alpha={alpha:.2f}] Chosen n_estimators (median of best rounds): {best_n_estimators}")
    print(f"[alpha={alpha:.2f}] Best Params:")
    for k in ["learning_rate","max_depth","min_child_weight","subsample","colsample_bytree",
              "min_split_loss","reg_lambda","reg_alpha","max_bin"]:
        print(f"  {k}: {best[k]}")

    return best_est

xgb_q10 = tune_xgb_quantile(alpha=0.10, n_trials=10, seed=42)
xgb_q90 = tune_xgb_quantile(alpha=0.90, n_trials=10, seed=42)