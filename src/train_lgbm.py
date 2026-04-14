import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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
TEST_RATIO = 0.15  # Hold out last 15% for metrics

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
    df = df.sort_values("Date")

    # Hold out last TEST_RATIO for evaluation metrics
    split_idx = int(len(df) * (1 - TEST_RATIO))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    X_train_full = df_train[FEATURE_COLS]
    X_test = df_test[FEATURE_COLS]

    metrics = {"eval_date": str(pd.Timestamp.today().date()), "n_test_matches": len(df_test)}

    for target_col, model_name in [("HomeGoals", "lgbm_home"), ("AwayGoals", "lgbm_away")]:
        y_train = df_train[target_col]
        y_test = df_test[target_col]

        print(f"\n=== Optimizing {model_name} ({N_TRIALS} trials) ===")
        # Suppress Optuna convergence prints to keep terminal clean
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, X_train_full, y_train), n_trials=N_TRIALS, show_progress_bar=True)

        best = study.best_params
        print(f"Best CV MAE: {study.best_value:.4f}")

        # Train final model on full training set with best params
        print("Training final model on training set...")
        final_model = lgb.LGBMRegressor(**best, verbosity=-1)
        final_model.fit(X_train_full, y_train)

        # Evaluate on held-out test set
        preds = final_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        prefix = "home_goals" if target_col == "HomeGoals" else "away_goals"
        metrics[f"{prefix}_mae"] = round(float(mae), 4)
        metrics[f"{prefix}_rmse"] = round(float(rmse), 4)
        print(f"Test MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")

        # Retrain on ALL data for the production model
        print("Retraining on full dataset for production model...")
        prod_model = lgb.LGBMRegressor(**best, verbosity=-1)
        prod_model.fit(df[FEATURE_COLS], df[target_col])
        joblib.dump(prod_model, MODELS / f"{model_name}.pkl")
        print(f"Saved -> models/{model_name}.pkl")

    with open(MODELS / "lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> models/lgbm_metrics.json")

if __name__ == "__main__":
    train_lgbm()
