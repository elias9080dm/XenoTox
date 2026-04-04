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


def optimize_model_regression(X_train_proc, y, model_name):
    """
    Optimize a regression model using Optuna.
    X_train must be already preprocessed
    y must be continuous (e.g., log LD50)
    """

    # 0. Input validation
    if len(X_train_proc) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got {len(X_train_proc)} and {len(y)}"
        )

    # 1. CV (NO stratification in regression)
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 2. Base models
    if model_name == "xgb":
        base_model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )

    elif model_name == "rf":
        base_model = RandomForestRegressor(
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "svm":
        base_model = SVR(
            kernel="rbf"
        )

    elif model_name == "ridge":
        base_model = Ridge(
            random_state=42
        )

    else:
        raise ValueError(f"Model not supported: {model_name}")

    # 3. Objective function
    def objective(trial):

        if model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "max_depth": trial.suggest_int("max_depth", 6, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }

        elif model_name == "svm":
            params = {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 0.1, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            }

        elif model_name == "ridge":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)
            }

        rmses = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_proc)):

            model = clone(base_model)
            model.set_params(**params)

            X_tr = X_train_proc[train_idx]
            X_val = X_train_proc[valid_idx]

            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_val = y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]

            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)

            # pruning
            trial.report(rmse, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(rmses)

    # 4. Pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1
    )

    # 5. Study
    study = optuna.create_study(
        study_name=f"{model_name}_regression_study",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    study.optimize(objective, n_trials=25)

    # 6. Final model
    final_model = clone(base_model)
    final_model.set_params(**study.best_params)
    final_model.fit(X_train_proc, y)

    final_model.best_params_ = study.best_params
    final_model.best_score_ = study.best_value

    return final_model


def train_stacking_model_regression(
    X_train_proc,
    y
):
    """
    Train a Stacking Regressor using optimized base models and a Ridge meta-model.

    X_train must be already preprocessed
    y must be continuous (e.g., log LD50)
    """

    print("StackingRegressor training:")

    # 1. Base models
    base_model_names = ["rf", "xgb", "svm"]
    estimators = []

    print("Training individual models:")

    for name in base_model_names:
        print(f"  → Optimizing {name.upper()} with Optuna...")
        model = optimize_model_regression(
            X_train_proc,
            y,
            model_name=name
        )
        estimators.append((name, model))

    # 2. CV for stacking
    cv_stack = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 3. Meta-model
    print("Training Meta-model...")
    meta_model = Ridge(
        alpha=1.0,
        random_state=42
    )

    # 4. Final Stacking
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
