import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics_regression(model, X_test, y_test, model_name, data_type="internal"):
    """
    Compute regression metrics and generate a 3-panel diagnostic plot.

    Parameters
    ----------
    model      : fitted sklearn-compatible regressor
    X_test     : numpy array, preprocessed test features
    y_test     : array-like, true pLD50 values
    model_name : str
    data_type  : 'internal' or 'external'

    Returns
    -------
    fig        : matplotlib Figure
    results    : dict with metrics_df, y_pred, residuals, rmse, mae, r2
    """
    y_pred = model.predict(X_test)
    y_true = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    sderr = float(np.std(y_pred - y_true))

    metrics_df = pd.DataFrame([{
        "Model":     model_name,
        "Data_type": data_type,
        "RMSE":      round(rmse, 4),
        "MAE":       round(mae, 4),
        "R2":        round(r2, 4),
        "Pearson_r": round(pearson_r, 4),
        "Bias":      round(bias, 4),
        "SD_error":  round(sderr, 4),
    }])

    print(metrics_df.to_string(index=False))

    residuals = y_pred - y_true

    # ----------------------------------------------------------------
    # 3-panel figure
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Predicted vs Observed
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.35, color="#2E86AB", s=15, edgecolors="none")
    mn = min(y_true.min(), y_pred.min()) - 0.2
    mx = max(y_true.max(), y_pred.max()) + 0.2
    ax1.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="y = x")
    slope, intercept, *_ = stats.linregress(y_true, y_pred)
    x_fit = np.linspace(mn, mx, 300)
    ax1.plot(x_fit, slope * x_fit + intercept, color="#E84855", lw=2,
             label=f"Linear fit (r={pearson_r:.3f})")
    ax1.set_xlabel("Observed pLD50", fontsize=11)
    ax1.set_ylabel("Predicted pLD50", fontsize=11)
    ax1.set_title(
        f"Predicted vs Observed\nR²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(mn, mx)
    ax1.set_ylim(mn, mx)

    # 2) Residuals vs Predicted
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.35, color="#A23B72", s=15, edgecolors="none")
    ax2.axhline(0, color="k", linestyle="--", lw=1.5, label="Zero")
    ax2.axhline(rmse,  color="#E84855", linestyle=":", lw=1.5, label=f"±RMSE ({rmse:.3f})")
    ax2.axhline(-rmse, color="#E84855", linestyle=":", lw=1.5)
    ax2.set_xlabel("Predicted pLD50", fontsize=11)
    ax2.set_ylabel("Residual (pred − obs)", fontsize=11)
    ax2.set_title(
        f"Residuals vs Predicted\nBias={bias:.4f} | SD={sderr:.4f}",
        fontsize=11, fontweight="bold"
    )
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # 3) Residuals distribution + normality
    ax3 = axes[2]
    ax3.hist(residuals, bins=50, color="#3D9970", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.8, label="Residuals")
    mu_r, sigma_r = residuals.mean(), residuals.std()
    x_norm = np.linspace(mu_r - 4 * sigma_r, mu_r + 4 * sigma_r, 300)
    ax3.plot(x_norm, stats.norm.pdf(x_norm, mu_r, sigma_r),
             "k-", lw=2, label=f"N(μ={mu_r:.3f}, σ={sigma_r:.3f})")
    # Shapiro-Wilk test (sample up to 5000)
    sample = residuals if len(residuals) <= 5000 else residuals[
        np.random.choice(len(residuals), 5000, replace=False)]
    _, p_sw = stats.shapiro(sample)
    ax3.set_xlabel("Residual", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title(
        f"Residuals Distribution\nShapiro-Wilk p={p_sw:.3e}",
        fontsize=11, fontweight="bold"
    )
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Regression Validation — {model_name.upper()} | {data_type.upper()}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    results = {
        "metrics_df": metrics_df,
        "y_pred":     y_pred,
        "residuals":  residuals,
        "rmse":       rmse,
        "mae":        mae,
        "r2":         r2,
        "pearson_r":  pearson_r,
    }

    return fig, results
