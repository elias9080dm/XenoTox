import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (make_scorer, matthews_corrcoef)
from sklearn.base import clone
from collections import Counter
import joblib

from utils_clf.config import RANDOM_STATE, CV_FOLDS, N_TRIALS


def optimize_model(X_train_proc, y_encoded, model_name):
    """
    Optimize a base model using Optuna.
    X_train must be already preprocessed
    y_train must be already encoded
    """
    # 0. Input validation
    if len(X_train_proc) != len(y_encoded):
        raise ValueError(
            f"X and y must have the same length. Got {len(X_train_proc)} and {len(y_encoded)}")

    # 2. CV
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # 3. Base models
    if model_name == "xgb":
        counter = Counter(y_encoded)
        if len(counter) < 2:
            raise ValueError(
                "Data must contain both classes (0 and 1) for XGBClassifier")
        if counter[1] == 0:
            raise ValueError("No positive class (1) samples in y_encoded")
        scale_pos_weight = counter[0] / counter[1]

        base_model = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight
        )

    elif model_name == "rf":
        base_model = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif model_name == "svm":
        base_model = SVC(
            probability=True,
            class_weight="balanced",
            kernel="rbf",
            random_state=RANDOM_STATE
        )
    elif model_name == "lr":
        base_model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Model not supported: {model_name}")

    # 4. Optuna objective function
    def objective(trial):
        current_model = clone(base_model)

        if model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "gamma": trial.suggest_float("gamma", 0.0, 0.3),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }

        elif model_name == "svm":
            params = {
                "C": trial.suggest_float("C", 0.1, 50, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 0.1, log=True),
            }

        elif model_name == "lr":
            params = {
                "C": trial.suggest_float("C", 1e-2, 100, log=True)
            }

        params_to_set = params.copy()
        scores = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_proc, y_encoded)):
            # Clone model inside loop to avoid state carryover between folds
            current_model = clone(base_model)
            current_model.set_params(**params_to_set)

            X_tr = X_train_proc[train_idx]
            X_val = X_train_proc[valid_idx]

            y_tr = y_encoded.iloc[train_idx]
            y_val = y_encoded.iloc[valid_idx]

            current_model.fit(X_tr, y_tr)

            preds = current_model.predict(X_val)

            score = matthews_corrcoef(y_val, preds)

            scores.append(score)

            # Reportar progreso al pruner
            trial.report(score, step=fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1
    )

    # 5. Optuna study
    study = optuna.create_study(
        study_name=f"{model_name}_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=pruner,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # 6. Final training with BEST parameters
    final_model = clone(base_model)
    final_model.set_params(**study.best_params)
    final_model.fit(X_train_proc, y_encoded)

    # Save useful metadata
    final_model.best_params_ = study.best_params
    final_model.best_score_ = study.best_value

    return final_model


def train_stacking_model(
    X_train_proc,
    y_train_enc
):
    """
    Train a Stacking Classifier using optimized base models and a Logistic Regression meta-model.
    X_train must be already preprocessed
    y_train must be already encoded
    """
    print("StackingClassifier training: ")

    # 1. Base models
    base_model_names = ["rf", "xgb", "svm"]
    estimators = []

    print("Training individual models:")

    for name in base_model_names:
        print(f"  → Optimizing {name.upper()} with Optuna...")
        model = optimize_model(
            X_train_proc,
            y_train_enc,
            model_name=name
        )
        estimators.append((name, model))

    # 2. CV stacking
    cv_stack = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # 3. Meta-model
    print("Training Meta-model...")
    meta_model = LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    # 4. Final Stacking
    print("Creating StackingClassifier...")
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        stack_method="predict_proba",
        cv=cv_stack,
        n_jobs=-1,
        passthrough=False
    )

    stacking_model.fit(X_train_proc, y_train_enc)

    return stacking_model


def save_model(BASE_DIR, target, model_name, final_model,
               full_descriptor_list, filtered_features, selected_features, preprocessor):
    """
    Save model and all components needed for prediction.

    Prediction pipeline for a new molecule:
      1. Calculate descriptors → select full_descriptor_list columns
      2. Filter to filtered_features → apply preprocessor.transform()
      3. Subset to selected_features → predict
    """
    model_filename = f"{BASE_DIR}/outputs_clf/{target}/models/best_model_{target}_{model_name}.pkl"

    model_components = {
        "model": final_model,
        "full_descriptor_list": full_descriptor_list,
        "filtered_features": filtered_features,
        "selected_features": selected_features,
        "preprocessor": preprocessor,
        "target": target,
        "model_name": model_name,
        "class_mapping": {0: "inactive", 1: "active"},
        "positive_class": 1
    }

    joblib.dump(model_components, model_filename)
    print(f"\nModel and components saved to: {model_filename}")
