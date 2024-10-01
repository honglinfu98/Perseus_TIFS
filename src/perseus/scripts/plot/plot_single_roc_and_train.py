"""
This function is used to get the global minimum and maximum training times
"""

from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker


import matplotlib.pyplot as plt
import numpy as np

from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]


# open the results files
with open(path.join(PROJECT_ROOT, "data", "results_tr.pkl"), "rb") as file:
    results_tr = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_f.pkl"), "rb") as file:
    results_f = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_t.pkl"), "rb") as file:
    results_t = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_fv.pkl"), "rb") as file:
    results_fv = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_e.pkl"), "rb") as file:
    results_e = pickle.load(file)


def get_global_min_max(results: dict, results1: dict, method: str):
    """
    This function is used to get the global minimum and maximum training times
    """
    all_times = []
    for res in [results, results1]:
        for dataset in datasets:
            all_times.extend(res[dataset][method]["train_times"])
    return min(all_times), max(all_times)


global_min_max = {}
for method in ["GCN", "GAT", "GraphSAGE"]:
    global_min_max[method] = get_global_min_max(results_t, results_t, method)


def plot_cdf(data: dict, ax, title, colors):
    """
    This function is used to plot the CDF of the training times for a data set.
    """
    label_map = {
        "DDINA": "Directed Diffusion",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted Diffusion",
    }

    for dataset in datasets:
        sorted_times = np.sort(data[dataset])
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        dataset_label = label_map.get(dataset, dataset)
        sns.lineplot(
            x=sorted_times,
            y=cdf,
            ax=ax,
            label=f"{dataset_label}",
            color=colors[dataset],
            drawstyle="steps-post",
        )

    ax.set_xscale("log")
    ax.set_xlim(min(sorted_times), max(sorted_times))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=5))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.3f}" if x < 1 else f"{int(x)}")
    )
    ax.tick_params(axis="x", which="major", labelrotation=45)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()


def save_cdf_plots(fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks):
    """
    This function is used to save the individual subfigures for the cdf plots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 24))  # Adjusted for vertical layout
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results_t[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(method_train_times, axs[j], method, dataset_colors)

        axs[j].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        axs[j].set_xlabel(axs[j].get_xlabel(), fontsize=fontsize_labels)
        axs[j].set_ylabel(axs[j].get_ylabel(), fontsize=fontsize_labels)
        axs[j].set_title(axs[j].get_title(), fontsize=fontsize_title)
        axs[j].legend(fontsize=fontsize_legend, loc="lower right")

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "cdf_plots.pdf"))
    plt.show()
    plt.close(fig)


# Call the updated function to generate and save the CDF plots
save_cdf_plots(
    fontsize_title=50, fontsize_labels=45, fontsize_legend=35, fontsize_ticks=35
)


def plot_roc(results: dict, model_name: str, ax, title, xlabel, ylabel):
    """
    This function is used to plot the FPR vs TPR for each model.
    """
    label_map = {
        "DDINA": "Directed Diffusion",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted Diffusion",
    }

    for dataset in datasets:
        fpr = results[dataset][model_name]["fpr"][0]
        tpr = results[dataset][model_name]["tpr"][0]
        dataset_label = label_map.get(dataset, dataset)
        sns.lineplot(
            x=fpr,
            y=tpr,
            ax=ax,
            label=f"{dataset_label}",
            color=dataset_colors[dataset],
        )

    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def save_roc_plots(fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks):
    """
    This function is used to save the individual subfigures for the ROC plots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 24))  # Adjusted for vertical layout
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        plot_roc(
            results_t,
            method,
            axs[j],
            method,
            "False Positive Rate",
            "True Positive Rate",
        )

        axs[j].tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        axs[j].set_xlabel(axs[j].get_xlabel(), fontsize=fontsize_labels)
        axs[j].set_ylabel(axs[j].get_ylabel(), fontsize=fontsize_labels)
        axs[j].set_title(axs[j].get_title(), fontsize=fontsize_title)
        axs[j].legend(fontsize=fontsize_legend, loc="lower right")

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "roc_plots.pdf"))
    plt.show()
    plt.close(fig)


# Call the updated function to generate and save the ROC plots
save_roc_plots(
    fontsize_title=50, fontsize_labels=45, fontsize_legend=35, fontsize_ticks=35
)
