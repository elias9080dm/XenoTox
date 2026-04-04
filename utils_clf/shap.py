import shap
import numpy as np
import pandas as pd
import tempfile
import os


def shap_top20(
    estimator,
    model_name,
    X_train_proc,
    feature_names,
    positive_class=1,
    max_samples=100,
    top_n=20
):

    # ==========================================================
    # 1. Omitir stacking (meta-modelo no usa descriptores)
    # ==========================================================
    if model_name.lower() == "stacking":
        print(
            f"SHAP omitido para {model_name}: "
            "el meta-modelo opera sobre salidas de modelos base."
        )
        return

    # ============================================
    # 3. Background (submuestreo estable desde TRAIN)
    # ============================================
    rng = np.random.default_rng(42)

    bg_size = min(100, X_train_proc.shape[0])
    bg_idx = rng.choice(X_train_proc.shape[0], bg_size, replace=False)

    X_background_proc = X_train_proc[bg_idx]
    print(f"--- SHAP Top-{top_n}: {model_name} ---")

    rng = np.random.default_rng(42)

    # ==========================================================
    # 2. Submuestreo estable
    # ==========================================================
    n_samples = min(max_samples, X_train_proc.shape[0])
    idx = rng.choice(X_train_proc.shape[0], size=n_samples, replace=False)
    X_shap = X_train_proc[idx]

    model_cls = estimator.__class__.__name__.lower()

    # ==========================================================
    # 3. TREE MODELS
    # ==========================================================
    if "xgb" in model_cls or "randomforest" in model_cls:

        # Fix XGBoost base_score parsing issue
        if "xgb" in model_cls:
            try:
                booster = estimator.get_booster()
                # Use tempfile for cross-platform compatibility
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                    tmp_path = tmp.name
                # Force reload of model parameters to fix serialization issues
                booster.save_model(tmp_path)
                booster.load_model(tmp_path)
                os.remove(tmp_path)
            except Exception:
                pass  # Continue if fix fails, may still work

        explainer = shap.TreeExplainer(
            estimator,
            data=X_background_proc,  # Added background data
            model_output="probability",
            feature_perturbation="interventional"  # Explicitly set to 'interventional'
        )

        shap_values = explainer(X_shap)

        # Extraer clase positiva si es binario
        if shap_values.values.ndim == 3:
            class_index = list(estimator.classes_).index(positive_class)
            shap_vals = shap_values.values[:, :, class_index]
        else:
            shap_vals = shap_values.values

    # ==========================================================
    # 4. SVM (Kernel SHAP sobre probabilidades)
    # ==========================================================
    elif "svc" in model_cls:

        background = X_background_proc[:50]

        pos_index = list(estimator.classes_).index(positive_class)

        def model_proba_pos(X):
            return estimator.predict_proba(X)[:, pos_index]

        explainer = shap.KernelExplainer(model_proba_pos, background)

        shap_vals = explainer.shap_values(
            X_shap,
            nsamples=100
        )

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

    # ==========================================================
    # 5. LOGISTIC REGRESSION
    # ==========================================================
    elif "logisticregression" in model_cls:

        explainer = shap.LinearExplainer(
            estimator,
            X_background_proc,
            feature_perturbation="interventional"
        )

        shap_values = explainer(X_shap)

        if shap_values.values.ndim == 3:
            class_index = list(estimator.classes_).index(positive_class)
            shap_vals = shap_values.values[:, :, class_index]
        else:
            shap_vals = shap_values.values

    else:
        raise ValueError(
            f"Modelo no soportado para SHAP: {estimator.__class__.__name__}")

    # ==========================================================
    # 6. Importancia media absoluta
    # ==========================================================
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    importance_df = pd.DataFrame({
        "Descriptor": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    top_df = importance_df.head(top_n)
    top_idx = top_df.index.to_numpy()

    # ==========================================================
    # 7. Plot resumen (solo top features)
    # ==========================================================
    fig = shap.summary_plot(
        shap_vals[:, top_idx],
        X_shap[:, top_idx],
        feature_names=top_df["Descriptor"].values,
        show=True
    )
    return importance_df, fig
