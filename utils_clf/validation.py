from sklearn.metrics import (
    make_scorer, accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, RocCurveDisplay,
    auc, roc_curve, PrecisionRecallDisplay, precision_recall_curve
)
import matplotlib.pyplot as plt
import pandas as pd


def compute_metrics(final_model, X_test, y_test, model_name, target, data_type="internal"):
    """
    X_test must be preprocessed and have the same features used in training.
    y_test must be encoded as inactive=0, active=1
    """
    # 1. Probabilities
    classes = list(final_model.classes_)
    if 1 not in classes:
        raise ValueError("Positive class (1) is not in final_model.classes_")

    pos_index = classes.index(1)
    y_proba = final_model.predict_proba(X_test)[:, pos_index]

    # Threshold
    y_pred = (y_proba >= 0.5).astype(int)

    # 2. Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # 3. Save metrics
    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "Target": target,
        "Threshold": 0.5,
        "Accuracy": acc,
        "Bal_Accuracy": bal_acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "MCC": mcc
    }])

    print(metrics_df)

    # 4. Curves with personalized styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        ax=ax1,
        name='',
        color='#2E86AB',
        linewidth=3,
        marker='o',
        markersize=0
    )

    # ROC styling
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2,
             alpha=0.5, label='Random Classifier')
    ax1.set_title(f'ROC Curve - {target.upper()} ({model_name})\nAUC = {roc_auc:.4f}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.text(0.5, -0.18, f'{data_type.upper()} VALIDATION',
             ha='center', va='top', transform=ax1.transAxes, fontsize=10)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    # Add model name to legend
    ax1.plot([], [], color='#2E86AB', linewidth=3, label=model_name)
    ax1.legend(loc='lower right', fontsize=10,
               framealpha=0.95, edgecolor='black')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_proba,
        ax=ax2,
        name='',
        color='#A23B72',
        linewidth=3,
        marker='o',
        markersize=0
    )

    # PR styling
    baseline = (y_test == 1).sum() / len(y_test)
    ax2.axhline(baseline, color='k', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax2.set_title(f'Precision-Recall Curve - {target.upper()} ({model_name})\nAUC = {pr_auc:.4f}',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.text(0.5, -0.18, f'{data_type.upper()} VALIDATION',
             ha='center', va='top', transform=ax2.transAxes, fontsize=10)
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    # Add model name to legend
    ax2.plot([], [], color='#A23B72', linewidth=3, label=model_name)
    ax2.legend(loc='lower left', fontsize=10,
               framealpha=0.95, edgecolor='black')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # Figure-level styling
    fig.suptitle(f'Model Evaluation - {target.upper()} | {model_name.upper()}',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.show()

    return fig, {
        "metrics_df": metrics_df,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
