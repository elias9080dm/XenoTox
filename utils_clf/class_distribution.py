import matplotlib.pyplot as plt
import seaborn as sns


def plot_dist(df_curated, activity_col, target):
    fig, ax = plt.subplots(figsize=(4, 4))

    sns.countplot(x=activity_col, data=df_curated,
                  edgecolor='black', hue=activity_col, palette={'active': 'orange', 'inactive': 'blue'},
                  alpha=0.7, legend=False, ax=ax)
    ax.set_title(
        f"Curated class distribution ({target})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            xytext=(0, -1),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.show()

    return fig
