import numpy as np
import matplotlib.pyplot as plt
import shap


def shap_analysis(model, model_name, X_train_proc, X_test_proc, feature_names,
                  n_background=100, n_explain=500):
    """
    SHAP analysis for regression models.

    Generates a 4-panel figure:
      1. Bar plot — top 20 features by mean |SHAP|
      2. Beeswarm summary — top 10 features
      3. Dependence plot — top-1 feature (colored by top-2)
      4. Dependence plot — top-2 feature (colored by top-1)

    Parameters
    ----------
    model          : fitted sklearn-compatible regressor
    model_name     : 'xgb', 'rf', 'svm', 'ridge', or 'stacking'
    X_train_proc   : numpy array, training features (scaled)
    X_test_proc    : numpy array, test features (scaled)
    feature_names  : list of str
    n_background   : int, background sample size for KernelExplainer
    n_explain      : int, max test samples to explain

    Returns
    -------
    fig        : matplotlib Figure  (None for unsupported models)
    shap_dict  : dict with shap_values, mean_abs_shap, top_features
    """
    X_tr = X_train_proc if not hasattr(X_train_proc, "values") else X_train_proc.values
    X_te = X_test_proc  if not hasattr(X_test_proc,  "values") else X_test_proc.values

    np.random.seed(42)

    # Background sample
    bg_idx = np.random.choice(len(X_tr), min(n_background, len(X_tr)), replace=False)
    X_bg = X_tr[bg_idx]

    # Explain sample
    te_idx = np.random.choice(len(X_te), min(n_explain, len(X_te)), replace=False)
    X_exp = X_te[te_idx]

    # ----------------------------------------------------------------
    # Choose explainer
    # ----------------------------------------------------------------
    if model_name in ("xgb", "rf"):
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_exp)

    elif model_name == "ridge":
        explainer  = shap.LinearExplainer(model, X_bg)
        shap_vals  = explainer.shap_values(X_exp)

    elif model_name == "svm":
        print("SVR: using KernelExplainer (slow — limited to 100 samples).")
        X_exp = X_exp[:100]
        explainer  = shap.KernelExplainer(model.predict, X_bg)
        shap_vals  = explainer.shap_values(X_exp)

    elif model_name == "stacking":
        print("Stacking: SHAP not computed directly. Run on base models individually.")
        return None, None

    else:
        print(f"SHAP not supported for model: {model_name}")
        return None, None

    # ----------------------------------------------------------------
    # Derived quantities
    # ----------------------------------------------------------------
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top20_idx]
    top_vals  = mean_abs[top20_idx]

    # ----------------------------------------------------------------
    # 4-panel figure
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    # ---- Panel 1: Mean |SHAP| bar plot ----
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, 20))
    ax1.barh(range(20), top_vals[::-1], color=colors[::-1])
    ax1.set_yticks(range(20))
    ax1.set_yticklabels(top_names[::-1], fontsize=8)
    ax1.set_xlabel("Mean |SHAP value|  (pLD50 units)", fontsize=10)
    ax1.set_title("Top 20 Features\n(Global Importance)", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # ---- Panel 2: Beeswarm (top 10) ----
    ax2 = fig.add_subplot(gs[0, 1])
    top10_idx = top20_idx[:10]
    for rank, fi in enumerate(reversed(top10_idx)):
        sv = shap_vals[:, fi]
        fv = X_exp[:, fi]
        vmin, vmax = fv.min(), fv.max()
        norm = (fv - vmin) / (vmax - vmin + 1e-10)
        jitter = np.random.uniform(-0.3, 0.3, len(sv))
        sc = ax2.scatter(sv, rank + jitter, c=norm, cmap="RdYlBu_r",
                         s=8, alpha=0.55, vmin=0, vmax=1)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([feature_names[i] for i in reversed(top10_idx)], fontsize=9)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("SHAP value  (impact on pLD50)", fontsize=10)
    ax2.set_title("SHAP Impact — Top 10\n(color: low ← feature value → high)",
                  fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax2, label="Feature value", shrink=0.75)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    # ---- Panel 3: Dependence plot — top-1 feature ----
    ax3 = fig.add_subplot(gs[1, 0])
    fi1, fi2 = top20_idx[0], top20_idx[1]
    sc3 = ax3.scatter(
        X_exp[:, fi1], shap_vals[:, fi1],
        c=X_exp[:, fi2], cmap="viridis", s=12, alpha=0.55
    )
    ax3.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax3.set_xlabel(feature_names[fi1], fontsize=10)
    ax3.set_ylabel(f"SHAP({feature_names[fi1]})", fontsize=10)
    ax3.set_title(
        f"Dependence: {feature_names[fi1]}\n(color = {feature_names[fi2]})",
        fontsize=11, fontweight="bold"
    )
    plt.colorbar(sc3, ax=ax3, label=feature_names[fi2], shrink=0.75)
    ax3.grid(alpha=0.3, linestyle="--")

    # ---- Panel 4: Dependence plot — top-2 feature ----
    ax4 = fig.add_subplot(gs[1, 1])
    sc4 = ax4.scatter(
        X_exp[:, fi2], shap_vals[:, fi2],
        c=X_exp[:, fi1], cmap="viridis", s=12, alpha=0.55
    )
    ax4.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax4.set_xlabel(feature_names[fi2], fontsize=10)
    ax4.set_ylabel(f"SHAP({feature_names[fi2]})", fontsize=10)
    ax4.set_title(
        f"Dependence: {feature_names[fi2]}\n(color = {feature_names[fi1]})",
        fontsize=11, fontweight="bold"
    )
    plt.colorbar(sc4, ax=ax4, label=feature_names[fi1], shrink=0.75)
    ax4.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"SHAP Analysis — {model_name.upper()}  (n={len(X_exp)} test samples)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    shap_dict = {
        "shap_values":   shap_vals,
        "mean_abs_shap": mean_abs,
        "top_features":  top_names,
        "X_explained":   X_exp,
    }

    return fig, shap_dict
