import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "H_roll_gf", "H_roll_ga", "H_roll_xg",
    "A_roll_gf", "A_roll_ga", "A_roll_xg",
    "H2H_HomeGoals_Avg", "H2H_AwayGoals_Avg", "H2H_Count",
    "HomePPDA", "AwayPPDA"
]

N_TRIALS = 30  # Reduced slightly for quicker iterative CLI runs
N_SPLITS = 3

def objective(trial, X, y):
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
    return np.mean(maes)

def train_lgbm():
    print("Loading clean dataset for LightGBM...")
    df = pd.read_csv(PROCESSED / "features_clean.csv", parse_dates=["Date"])
    X = df[FEATURE_COLS]

    for target_col, model_name in [("HomeGoals", "lgbm_home"), ("AwayGoals", "lgbm_away")]:
        y = df[target_col]
        print(f"\n=== Optimizing {model_name} ({N_TRIALS} trials) ===")
        # Suppress Optuna convergence prints to keep terminal clean
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=True)

        best = study.best_params
        print(f"Best MAE: {study.best_value:.4f}")
        
        print("Training final full model...")
        final_model = lgb.LGBMRegressor(**best, verbosity=-1)
        final_model.fit(X, y)
        joblib.dump(final_model, MODELS / f"{model_name}.pkl")
        print(f"Saved -> models/{model_name}.pkl")

if __name__ == "__main__":
    train_lgbm()
