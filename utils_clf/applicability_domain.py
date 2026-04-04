import numpy as np
import matplotlib.pyplot as plt


def applicability_domain_analysis(target, model_name, X_train_proc, X_test_proc, y_test_enc, y_proba):
    """
    Perform applicability domain analysis using leverage (hat diagonal) and visualize with a plot (leverage vs probability).

    Parameters:
    - target: target name for file saving
    - model_name: model name for file saving
    - X_train_proc: preprocessed training data (for AD calculation)
    - X_test_proc: preprocessed test data (for AD calculation)
    - y_test_enc: encoded test labels (for coloring points)
    - y_proba: predicted probabilities for test data (for y-axis in plot)
    """
    # Leverage (descriptor space AD)
    n_train, p = X_train_proc.shape

    epsilon = 1e-6
    XtX_inv = np.linalg.pinv(
        X_train_proc.T @ X_train_proc + epsilon * np.eye(p)
    )

    # Leverage for TRAIN and TEST (unclipped for metrics)
    leverage_train = np.sum(
        X_train_proc @ XtX_inv * X_train_proc,
        axis=1
    )

    leverage_test = np.sum(
        X_test_proc @ XtX_inv * X_test_proc,
        axis=1
    )

    # AD threshold (h*) - OECD recommended: 3(p+1)/n
    # Alternative: 2.5 or 3.5 depending on stringency required
    h_star = 3 * (p + 1) / n_train

    # IMPORTANT: Calculate AD percentage BEFORE clipping
    outside_ad = leverage_test > h_star
    pct_outside = outside_ad.mean() * 100

    print(f"Training set size: {n_train}")
    print(f"Number of descriptors: {p}")
    print(f"AD threshold (h*): {h_star:.4f}")
    print(
        f"Compounds outside AD: {pct_outside:.1f}% ({outside_ad.sum()}/{len(leverage_test)})")
    print(
        f"Leverage range - Train: [{leverage_train.min():.4f}, {leverage_train.max():.4f}]")
    print(
        f"Leverage range - Test:  [{leverage_test.min():.4f}, {leverage_test.max():.4f}]")

    # Williams plot (adapted for classification QSAR)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Clip only for visualization
    leverage_test_vis = np.clip(leverage_test, 0, 10 * h_star)

    active_mask = y_test_enc == 1
    inactive_mask = y_test_enc == 0

    # Plot points colored by AD membership
    inside_ad = ~outside_ad

    scatter1 = ax.scatter(
        leverage_test_vis[inside_ad & active_mask],
        y_proba[inside_ad & active_mask],
        alpha=0.7, s=60, c='green', label='Active (inside AD)', edgecolors='darkgreen', linewidth=0.5
    )

    scatter2 = ax.scatter(
        leverage_test_vis[inside_ad & inactive_mask],
        y_proba[inside_ad & inactive_mask],
        alpha=0.6, s=60, c='lightblue', label='Inactive (inside AD)', edgecolors='darkblue', linewidth=0.5
    )

    scatter3 = ax.scatter(
        leverage_test_vis[outside_ad & active_mask],
        y_proba[outside_ad & active_mask],
        alpha=0.8, s=100, marker='^', c='red', label='Active (outside AD)', edgecolors='darkred', linewidth=1
    )

    scatter4 = ax.scatter(
        leverage_test_vis[outside_ad & inactive_mask],
        y_proba[outside_ad & inactive_mask],
        alpha=0.8, s=100, marker='^', c='orange', label='Inactive (outside AD)', edgecolors='darkorange', linewidth=1
    )

    # Reference lines
    ax.axvline(h_star, linestyle='--', linewidth=2, color='red',
               alpha=0.8, label=f'AD threshold (h* = {h_star:.3f})')

    # Confidence interval for AD (optional)
    ax.axvspan(0, h_star, alpha=0.1, color='green',
               label='Applicability Domain')

    # Text annotation
    ax.text(0.98, 0.02, f'{pct_outside:.1f}% outside AD',
            transform=ax.transAxes, fontsize=11, horizontalalignment='right', verticalalignment='bottom')

    ax.set_xlabel('Leverage (hat diagonal)', fontsize=12)
    ax.set_ylabel('Predicted probability (Active)', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Applicability Domain Analysis ({target} - {model_name})',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    return fig
