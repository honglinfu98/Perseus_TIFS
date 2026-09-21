"""
This script visualizes the results of hyperparameter tuning for GNN models.
It loads parameter tuning results from a pickle file and generates a heatmap plot
showing the performance metric (e.g., best validation F1 score) as a function of
learning rate and hidden channel size for a specified model, dataset, and batch size.
"""

import os
import pickle
import matplotlib.pyplot as plt
from perseus.settings import PROJECT_ROOT
import numpy as np


def visualize_results(
    metric: str = "best_val_f1",
    fontsize: int = 23,
    save_path: str = os.path.join(PROJECT_ROOT, "results", "tuning_grid.pdf"),
    batch_size: int = 8,
    data_name_filter: str = "DDM",
    model_name_filter: str = "MultiGAT",
    results_path: str = os.path.join(
        PROJECT_ROOT, "results", "parameter_search_results.pkl"
    ),
):
    """
    Visualize hyperparameter tuning results as a heatmap.

    Loads parameter tuning results from a pickle file and plots a heatmap of the specified metric
    (e.g., best validation F1 score) for different combinations of learning rate and hidden channel size,
    filtered by batch size, dataset name, and model name.

    The pickle is a flat list of dicts produced by
    ``perseus.evaluation.run_fusion_experiments.run_parameter_search`` with keys
    ``batch_size``, ``lr``, ``hidden_channels``, ``model``, ``data`` and ``best_val_f1``.

    Args:
        metric (str): The key in the result dict to plot (e.g., 'best_val_f1').
        fontsize (int): Font size for plot labels and ticks.
        save_path (str): Path to save the generated plot PDF.
        batch_size (int): Batch size to filter results by.
        data_name_filter (str): Dataset name to filter results by (e.g., 'DDM').
        model_name_filter (str): Model name to filter results by (e.g., 'MultiGAT').
        results_path (str): Path to the parameter search results pickle.
    """
    # Load flat param tuning results
    with open(results_path, "rb") as file:
        param_tuning_results = pickle.load(file)

    points = []
    for entry in param_tuning_results:
        if entry.get("data") != data_name_filter:
            continue
        if entry.get("model") != model_name_filter:
            continue
        if entry.get("batch_size") != batch_size:
            continue
        lr = entry.get("lr")
        hidden_channels = entry.get("hidden_channels")
        metric_val = entry.get(metric)
        if lr is not None and hidden_channels is not None and metric_val is not None:
            points.append((lr, hidden_channels, metric_val))

    if not points:
        print("No points found for plotting.")
    else:
        lrs = sorted(set(p[0] for p in points))
        hiddens = sorted(set(p[1] for p in points))
        metric_grid = np.full((len(hiddens), len(lrs)), np.nan)
        for lr, hidden, val in points:
            i = hiddens.index(hidden)
            j = lrs.index(lr)
            metric_grid[i, j] = val

        def format_lr(lr):
            mantissa, exponent = f"{lr:.0e}".split("e")
            return f"{mantissa}e{int(exponent)}"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(metric_grid, aspect="auto", cmap="Greens")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([format_lr(lr) for lr in lrs], fontsize=fontsize)
        ax.set_yticks(range(len(hiddens)))
        ax.set_yticklabels([str(h) for h in hiddens], fontsize=fontsize)
        ax.set_xlabel("Learning Rate", fontsize=fontsize)
        ax.set_ylabel("Hidden Channels", fontsize=fontsize)

        # Add red cross at (32, 0.005)
        target_hidden = 32
        target_lr = 0.005
        if target_hidden in hiddens and target_lr in lrs:
            i_cross = hiddens.index(target_hidden)
            j_cross = lrs.index(target_lr)
            ax.plot(j_cross, i_cross, "rx", markersize=35, markeredgewidth=3)

        # Removed the title as requested
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
        plt.show()


if __name__ == "__main__":
    # To visualize, set the metric you want, e.g.:
    # "best_val_f1", or other keys present in the pickle
    visualize_results(metric="best_val_f1", batch_size=8)
