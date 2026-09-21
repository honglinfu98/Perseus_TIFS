from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from perseus.settings import PROJECT_ROOT


def plot_fraud_trends(base_size=12, start_year=2018):
    """
    Plots the percentage of pump-and-dump fraud cases per year with a single parameter to control font sizes.

    Parameters:
        base_size (int): Base font size for all plot elements.
        start_year (int): The starting year for filtering data.
    """
    with open(path.join(PROJECT_ROOT, "data", "comparison.pkl"), "rb") as file:
        comparison = pickle.load(file)

    df = pd.DataFrame(comparison)
    df["source_posted_at"] = pd.to_datetime(df["source_posted_at"])
    df["year"] = df["source_posted_at"].dt.year

    pivot_df = df.pivot_table(
        index="year", columns="fraud_type", aggfunc="size", fill_value=0
    )

    pivot_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    plt.rcParams.update({"font.size": base_size})
    filtered_df = pivot_percentage[pivot_percentage.index >= start_year]

    filtered_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 9),
        color={"timepump": "C1", "crowdpump": "C0"},
        alpha=0.5,
    )

    plt.xlabel("")
    plt.ylabel("Proportion", fontsize=base_size)
    plt.xticks(rotation=45, fontsize=base_size)
    plt.yticks(fontsize=base_size)
    plt.legend(
        # title="Fraud Type",
        title_fontsize=base_size,
        fontsize=base_size,
        labels=["Crowd Pump", "Time Pump"],
        loc="lower right",
    )
    plt.tight_layout()

    plt.savefig(path.join(PROJECT_ROOT, "results", "march_comparison_plot.pdf"))
    plt.show()


# Example usage:
plot_fraud_trends(base_size=40, start_year=2018)
