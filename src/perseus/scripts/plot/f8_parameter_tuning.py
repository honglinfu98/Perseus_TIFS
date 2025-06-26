import os
import pandas as pd
import matplotlib.pyplot as plt
from perseus.settings import PROJECT_ROOT


def visualize_results(
    metric=None,
    csv_path=os.path.join(PROJECT_ROOT, "data", "buffer", "hp_search_results_a.csv"),
    fontsize=20,
):
    df = pd.read_csv(csv_path)
    available_metrics = [col for col in df.columns if col.startswith("test_")]
    print("Available metrics for visualization:", available_metrics)

    # Default to the first available metric if not specified
    if metric is None:
        if available_metrics:
            metric = available_metrics[0]
            print(f"Defaulting to metric: {metric}")
        else:
            raise ValueError(
                "No metric columns starting with 'test_' found in the CSV."
            )

    for data_name in df["data"].unique():
        for model_name in df["model"].unique():
            for wd in df["weight_decay"].unique():
                sub = df[
                    (df["data"] == data_name)
                    & (df["model"] == model_name)
                    & (df["weight_decay"] == wd)
                ]
                if sub.empty or metric not in sub.columns:
                    continue
                pivot = sub.pivot_table(
                    index="hidden_channels", columns="lr", values=metric
                )
                plt.figure()
                plt.imshow(pivot.values, aspect="auto", cmap="Greens")
                plt.xticks(range(len(pivot.columns)), pivot.columns, fontsize=fontsize)
                plt.yticks(range(len(pivot.index)), pivot.index, fontsize=fontsize)
                plt.xlabel("Learning Rate", fontsize=fontsize)
                plt.ylabel("Hidden Channels", fontsize=fontsize)
                plt.title(
                    f"{data_name} - {model_name} - {wd} - {metric}", fontsize=fontsize
                )

                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=fontsize)
                plt.tight_layout()

    # save the plot to data/tuning.pdf
    plt.savefig(os.path.join(PROJECT_ROOT, "data", "tuning.pdf"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # To visualize, set the metric you want, e.g.:
    # "test_f1", "test_precision", "test_recall", "test_acc", "test_mcc"
    visualize_results(metric="test_f1")
