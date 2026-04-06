from sklearn.ensemble import StackingRegressor
import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import joblib

from utils_reg.config import RANDOM_STATE, CV_FOLDS, N_TRIALS, DEVICE


def optimize_model_regression(X_train_proc, y, model_name):
    """
    Optimize a regression model using Optuna.

    X_train must already be preprocessed (imputed + scaled).
    y must be continuous (e.g., pLD50).
    """
    if len(X_train_proc) != len(y):
        raise ValueError(
            f"X and y length mismatch: {len(X_train_proc)} vs {len(y)}"
        )

    # ----------------------------------------------------------------
    # Cross-validation
    # ----------------------------------------------------------------
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ----------------------------------------------------------------
    # Base model
    # ----------------------------------------------------------------
    if model_name == "xgb":
        base_model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=1 if DEVICE == "cuda" else -1,
            device=DEVICE,
            tree_method="hist"   # works with both CPU and CUDA
        )
        if DEVICE == "cuda":
            print("[XGB] Using GPU (CUDA)")

    elif model_name == "rf":
        base_model = RandomForestRegressor(
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_name == "svm":
        base_model = SVR(kernel="rbf")

    elif model_name == "ridge":
        base_model = Ridge()    # Ridge has no random_state

    else:
        raise ValueError(f"Model not supported: {model_name}")

    # ----------------------------------------------------------------
    # Optuna objective
    # ----------------------------------------------------------------
    def objective(trial):
        if model_name == "xgb":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 300, 800),
                "max_depth":       trial.suggest_int("max_depth", 3, 6),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":       trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "gamma":           trial.suggest_float("gamma", 0.0, 0.5),
                "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":       trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        elif model_name == "rf":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 300, 800),
                "max_depth":       trial.suggest_int("max_depth", 6, 15),
                "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 5),
                "min_samples_split":trial.suggest_int("min_samples_split", 2, 10),
                "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        elif model_name == "svm":
            params = {
                "C":      trial.suggest_float("C", 0.1, 100, log=True),
                "gamma":  trial.suggest_float("gamma", 1e-4, 0.1, log=True),
                "epsilon":trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            }
        elif model_name == "ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}

        rmses = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_proc)):
            m = clone(base_model)
            m.set_params(**params)

            X_tr  = X_train_proc[train_idx]
            X_val = X_train_proc[valid_idx]
            y_tr  = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_val = y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]

            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            rmse  = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)

            trial.report(rmse, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(rmses)

    # ----------------------------------------------------------------
    # Study
    # ----------------------------------------------------------------
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=1, interval_steps=1
    )
    study = optuna.create_study(
        study_name=f"{model_name}_regression_study",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # ----------------------------------------------------------------
    # Final model with best params
    # ----------------------------------------------------------------
    final_model = clone(base_model)
    final_model.set_params(**study.best_params)
    final_model.fit(X_train_proc, y)

    final_model.best_params_ = study.best_params
    final_model.best_score_  = study.best_value

    return final_model


def train_stacking_model_regression(X_train_proc, y):
    """
    Stacking Regressor: RF + XGB + SVM base models, Ridge meta-model.
    All base models are Optuna-optimized.
    """
    print("StackingRegressor training:")
    estimators = []

    for name in ["rf", "xgb", "svm"]:
        print(f"  → Optimizing {name.upper()} with Optuna...")
        model = optimize_model_regression(X_train_proc, y, model_name=name)
        estimators.append((name, model))

    cv_stack = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    meta_model = Ridge(alpha=1.0)

    print("Creating StackingRegressor...")
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv_stack,
        n_jobs=-1,
        passthrough=False
    )
    stacking_model.fit(X_train_proc, y)
    return stacking_model


def save_model(BASE_DIR, target, model_name, final_model,
               full_descriptor_list, filtered_features, selected_features, preprocessor):
    """
    Serialize model and all components needed for prediction.

    Prediction pipeline for a new molecule:
      1. Calculate 217 descriptors → select full_descriptor_list columns
      2. Filter to filtered_features → apply preprocessor.transform()
      3. Subset to selected_features by index → predict
    """
    model_filename = (
        f"{BASE_DIR}/outputs_reg/{target}/models/best_model_{target}_{model_name}.pkl"
    )

    model_components = {
        "model":               final_model,
        "full_descriptor_list":full_descriptor_list,
        "filtered_features":   filtered_features,
        "selected_features":   selected_features,
        "preprocessor":        preprocessor,
        "target":              target,
        "model_name":          model_name,
        "task":                "regression",
        "output_label":        "pLD50",
    }

    joblib.dump(model_components, model_filename)
    print(f"\nModel saved to: {model_filename}")
