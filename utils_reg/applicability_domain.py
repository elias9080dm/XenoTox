import numpy as np
import matplotlib.pyplot as plt


def applicability_domain_analysis(model_name, X_train, X_test, y_test, y_pred):
    """
    Leverage-based Applicability Domain for regression (QSAR OECD standard).

    Generates a Williams plot: standardized residuals vs leverage.
    Thresholds:
        h* = 3(k+1)/n  — OECD leverage threshold
        |std_res| > 3   — outlier threshold

    Parameters
    ----------
    model_name : str
    X_train    : numpy array, training set (preprocessed)
    X_test     : numpy array, test set (preprocessed)
    y_test     : array-like, observed values
    y_pred     : numpy array, model predictions

    Returns
    -------
    fig      : matplotlib Figure
    ad_stats : dict
    """
    X_tr = X_train if not hasattr(X_train, "values") else X_train.values
    X_te = X_test  if not hasattr(X_test,  "values") else X_test.values
    y_true = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    # ----------------------------------------------------------------
    # Leverage (hat diagonal)
    # h_i = x_i^T (X^T X)^{-1} x_i
    # ----------------------------------------------------------------
    XtX_inv = np.linalg.pinv(X_tr.T @ X_tr)
    h_test = np.array([float(x @ XtX_inv @ x) for x in X_te])

    # ----------------------------------------------------------------
    # OECD threshold
    # ----------------------------------------------------------------
    k = X_te.shape[1]
    n = X_tr.shape[0]
    h_star = 3.0 * (k + 1) / n

    # ----------------------------------------------------------------
    # Standardized residuals
    # ----------------------------------------------------------------
    residuals = y_pred - y_true
    s = np.std(residuals)
    std_residuals = residuals / (s + 1e-10)

    # ----------------------------------------------------------------
    # AD flags
    # ----------------------------------------------------------------
    outside_h  = h_test > h_star
    outside_r  = np.abs(std_residuals) > 3
    outside_ad = outside_h | outside_r
    pct_outside = 100.0 * outside_ad.sum() / len(outside_ad)

    inside_ad = ~outside_ad

    # ----------------------------------------------------------------
    # Williams plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        h_test[inside_ad], std_residuals[inside_ad],
        color="#2E86AB", alpha=0.6, s=25, edgecolors="none",
        label=f"Inside AD (n={inside_ad.sum()})"
    )
    ax.scatter(
        h_test[outside_ad], std_residuals[outside_ad],
        color="#E84855", alpha=0.85, s=50, marker="^", edgecolors="none",
        label=f"Outside AD (n={outside_ad.sum()}, {pct_outside:.1f}%)"
    )

    # Thresholds
    ax.axvline(h_star, color="k", linestyle="--", lw=1.8, label=f"h* = {h_star:.4f}")
    ax.axhline( 3.0,  color="#555555", linestyle=":", lw=1.5, label="|std. residual| = 3")
    ax.axhline(-3.0,  color="#555555", linestyle=":", lw=1.5)
    ax.axhline( 0.0,  color="#AAAAAA", linestyle="-",  lw=0.8, alpha=0.5)

    ax.set_xlabel("Leverage  h", fontsize=12)
    ax.set_ylabel("Standardized Residual", fontsize=12)
    ax.set_title(
        f"Williams Plot — Applicability Domain ({model_name.upper()})\n"
        f"{pct_outside:.1f}% of test compounds outside AD | "
        f"h* = {h_star:.4f}  (k={k}, n={n})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    ad_stats = {
        "h_star":         h_star,
        "h_test":         h_test,
        "std_residuals":  std_residuals,
        "outside_h":      outside_h,
        "outside_r":      outside_r,
        "outside_ad":     outside_ad,
        "pct_outside_ad": pct_outside,
    }

    return fig, ad_stats
