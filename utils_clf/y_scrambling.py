import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns


def y_scrambling(final_model, X_train_preprocessed, y_train_enc, X_test_preprocessed, y_test_enc):
    scores_scramble = []

    print("Initialazing Y-scrambling...")

    for i in tqdm(range(20), desc="Y-Scrambling"):

        # 1. Label scrambling
        y_scrambled = np.random.permutation(y_train_enc)

        # 2. Clone model
        scramble_model = clone(final_model)

        # 3. Train on scrambled data
        scramble_model.fit(X_train_preprocessed, y_scrambled)

        # 4. Test validation
        y_pred_scramble = scramble_model.predict(X_test_preprocessed)

        # 5. Metric
        mcc = matthews_corrcoef(y_test_enc, y_pred_scramble)
        scores_scramble.append(mcc)

    # Results
    yscramble_results_df = pd.DataFrame({
        "Iteration": range(1, len(scores_scramble) + 1),
        "MCC": scores_scramble
    })
    return yscramble_results_df


def plot_yscrambling_results(yscrambled_results, real_mcc, target, model_name):

    mean_yscramble_mcc = yscrambled_results['MCC'].mean()

    # Gráfica
    fig = plt.figure(figsize=(6, 5))
    sns.histplot(yscrambled_results['MCC'],
                 color='skyblue', label='Y-Scrambled MCCs')
    plt.axvline(0.0, linestyle=":", linewidth=1.5,
                label="Random performance (MCC = 0)")
    plt.axvline(mean_yscramble_mcc, color='orange', linestyle='--',
                label=f'Mean Y-Scrambled MCC ({mean_yscramble_mcc:.4f})')
    plt.axvline(real_mcc, color='red', linestyle='-',
                label=f'Real MCC ({real_mcc:.4f})')
    plt.title(
        f'Real MCC vs. Y-Scrambled MCCs ({target} - {model_name})', fontsize=14, fontweight='bold'),
    plt.xlabel('Matthews Correlation Coefficient (MCC)'), plt.ylabel(
        'Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return fig
