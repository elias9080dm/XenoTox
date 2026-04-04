import matplotlib.pyplot as plt
import seaborn as sns


def plot_pld50_distribution(df_curated):
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df_curated,
        x=df_curated['pLD50'],
        kde=True,
        bins=50,
        color='cornflowerblue'
    )
    plt.xlabel('-log(pLD50) [mol/kg]', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of -log(pLD50)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig
