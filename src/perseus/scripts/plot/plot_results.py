"""
This function is used to get the global minimum and maximum training times
"""

from os import path
import pickle
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
import torch
from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1


with open(path.join(PROJECT_ROOT, "data", "results.pkl"), "rb") as file:
    results = pickle.load(file)


with open(path.join(PROJECT_ROOT, "data", "results1.pkl"), "rb") as file:
    results1 = pickle.load(file)


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
    global_min_max[method] = get_global_min_max(results, results1, method)


def plot_cdf(data: dict, ax, title, colors, global_min, global_max):
    """
    This function is used to plot the CDF of the training times for each method
    """
    label_map = {
        "DDINA": "Directed DANI",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted DANI",
    }
    for dataset, times in data.items():
        sorted_times = np.sort(times)
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
    ax.set_xlim(global_min, global_max)

    if global_max / global_min > 1000:
        numticks = 3
    else:
        numticks = 5

    ax.xaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs="auto", numticks=numticks)
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.3f}" if x < 1 else f"{int(x)}")
    )

    ax.tick_params(axis="x", which="major", labelrotation=45)

    ax.set_ylim(0, 1)  # This ensures the y-axis starts at 0 and ends at 1
    ax.set_xlabel("Time per Epoch (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()


def plot_fpr_tpr(
    results: dict,
    model_name: str,
    ax,
    datasets: str,
    label: int,
    dataset_colors: str,
    title,
    xlabel,
    ylabel,
):
    """
    This function is used to plot the FPR vs TPR for each model
    """
    label_map = {
        "DDINA": "Directed DANI",
        "COSS": "Cosine Similarity",
        "DDM": "Weighted DANI",
    }

    for dataset in datasets:
        fpr = results[dataset][model_name]["fpr"][label]
        tpr = results[dataset][model_name]["tpr"][label]

        dataset_label = label_map.get(dataset, dataset)

        sns.lineplot(
            x=fpr, y=tpr, ax=ax, label=f"{dataset_label}", color=dataset_colors[dataset]
        )

    ax.set_ylim(0, 1)  # This ensures the y-axis starts at 0 and ends at 1
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def save_individual_subfigures(
    fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks
):
    """
    This function is used to save the individual subfigures for the cdf and roc plots
    """

    def set_style(ax, show_title=True, show_xlabel=True, show_ylabel=True):
        if show_title:
            ax.set_title(ax.get_title(), fontsize=fontsize_title)
        else:
            ax.set_title("")
        if show_xlabel:
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
        else:
            ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
        if ax.get_legend():
            ax.legend(
                title=ax.get_legend().get_title().get_text(),
                fontsize=fontsize_legend,
                title_fontsize=fontsize_legend,
                loc="lower right",  # Set the legend location to the bottom right
            )

    fig_a, axs_a = plt.subplots(1, 3, figsize=(36, 8))
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_a[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )
        if j < 2:  # Remove legends for the first two subplots
            axs_a[j].get_legend().remove()
    set_style(axs_a[0], show_title=True, show_xlabel=False, show_ylabel=True)
    for ax in axs_a[1:]:
        set_style(ax, show_title=True, show_xlabel=False, show_ylabel=False)
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_a.pdf"))
    plt.close(fig_a)

    fig_b, axs_b = plt.subplots(1, 3, figsize=(36, 8))
    for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
        method_train_times = {
            dataset: results1[dataset][method]["train_times"] for dataset in datasets
        }
        plot_cdf(
            method_train_times,
            axs_b[j],
            method,
            dataset_colors,
            *global_min_max[method],
        )
        axs_b[j].get_legend().remove()  # Remove legend for each subplot
    set_style(axs_b[0], show_title=False, show_xlabel=True, show_ylabel=True)
    for ax in axs_b[1:]:
        set_style(ax, show_title=False, show_xlabel=True, show_ylabel=False)
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_b.pdf"))
    plt.close(fig_b)

    # For subfigure_c
    fig_c, axs_c = plt.subplots(1, len(models), figsize=(36, 8))
    for i, model_name in enumerate(models):
        plot_fpr_tpr(
            results,
            model_name,
            axs_c[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        if i < len(models) - 1:  # Remove legends for the first two models
            axs_c[i].get_legend().remove()
    set_style(axs_c[0], show_title=True, show_xlabel=False, show_ylabel=True)
    for ax in axs_c[1:]:
        set_style(ax, show_title=True, show_xlabel=False, show_ylabel=False)
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_c.pdf"))
    plt.close(fig_c)
    # For subfigure_d
    fig_d, axs_d = plt.subplots(1, len(models), figsize=(36, 8))
    for i, model_name in enumerate(models):
        plot_fpr_tpr(
            results1,
            model_name,
            axs_d[i],
            datasets,
            label,
            dataset_colors,
            title=f"{model_name}",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axs_d[i].get_legend().remove()  # Remove legends for each subplot
    set_style(axs_d[0], show_title=False, show_xlabel=True, show_ylabel=True)
    for ax in axs_d[1:]:
        set_style(ax, show_title=False, show_xlabel=True, show_ylabel=False)

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "subfigure_d.pdf"))
    plt.close(fig_d)


save_individual_subfigures(
    fontsize_title=50, fontsize_labels=45, fontsize_legend=35, fontsize_ticks=35
)


# def create_plots(fontsize_title, fontsize_labels, fontsize_legend, fontsize_ticks):
#     """
#     This function is used to create the main figure with subfigures for each of the sections
#     """
#     # Create the main figure with desired size
#     fig = plt.figure(figsize=(36, 15))

#     # Create subfigures for each of the sections (a to d)
#     subfigs = fig.subfigures(2, 2, wspace=0.07, hspace=0.07)

#     # Helper function to set the styles
#     def set_style(ax, show_title=True, show_xlabel=True, show_ylabel=True):
#         if show_title:
#             ax.set_title(ax.get_title(), fontsize=fontsize_title)
#         else:
#             ax.set_title("")
#         if show_xlabel:
#             ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
#         else:
#             ax.set_xlabel("")
#         if show_ylabel:
#             ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
#         else:
#             ax.set_ylabel("")
#         ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
#         if ax.get_legend():
#             ax.legend(
#                 title=ax.get_legend().get_title().get_text(),
#                 fontsize=fontsize_legend,
#                 title_fontsize=fontsize_legend,
#             )

#     # Configure each subfigure
#     # Subfigure a (top left)
#     axs_a = subfigs[0, 0].subplots(1, 3)
#     for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
#         method_train_times = {
#             dataset: results[dataset][method]["train_times"] for dataset in datasets
#         }
#         plot_cdf(
#             method_train_times,
#             axs_a[j],
#             method,
#             dataset_colors,
#             *global_min_max[method],
#         )

#     set_style(
#         axs_a[0], show_title=True, show_xlabel=False, show_ylabel=True
#     )  # Only leftmost y-label
#     for ax in axs_a[1:]:
#         set_style(
#             ax, show_title=True, show_xlabel=False, show_ylabel=False
#         )  # No y-label

#     # Subfigure c (top right) - Adjusted to use all models
#     axs_c = subfigs[0, 1].subplots(
#         1, len(models)
#     )  # Ensure the subplot layout matches the number of models
#     for i, model_name in enumerate(models):
#         print(model_name)

#         plot_fpr_tpr(
#             results,
#             model_name,
#             axs_c[i],
#             datasets,
#             label,
#             dataset_colors,
#             title=f"{model_name}",
#             xlabel="False Positive Rate",
#             ylabel="True Positive Rate",
#         )
#         set_style(axs_c[i], show_title=True, show_xlabel=False, show_ylabel=(i == 0))

#     # Subfigure b (bottom left)
#     axs_b = subfigs[1, 0].subplots(1, 3)
#     for j, method in enumerate(["GCN", "GAT", "GraphSAGE"]):
#         method_train_times = {
#             dataset: results1[dataset][method]["train_times"] for dataset in datasets
#         }
#         plot_cdf(
#             method_train_times,
#             axs_b[j],
#             method,
#             dataset_colors,
#             *global_min_max[method],
#         )

#     set_style(
#         axs_b[0], show_title=False, show_xlabel=True, show_ylabel=True
#     )  # Only leftmost y-label
#     for ax in axs_b[1:]:
#         set_style(
#             ax, show_title=False, show_xlabel=True, show_ylabel=False
#         )  # No y-label

#     # Subfigure d (bottom right) - Adjusted to use all models
#     axs_d = subfigs[1, 1].subplots(
#         1, len(models)
#     )  # Ensure the subplot layout matches the number of models
#     for i, model_name in enumerate(models):
#         print(model_name)
#         plot_fpr_tpr(
#             results1,
#             model_name,
#             axs_d[i],
#             datasets,
#             label,
#             dataset_colors,
#             title=f"{model_name}",
#             xlabel="False Positive Rate",
#             ylabel="True Positive Rate",
#         )
#         set_style(axs_d[i], show_title=False, show_xlabel=True, show_ylabel=(i == 0))

#     # Display the figure
#     # plt.tight_layout()  # Adjust layout

#     plt.savefig(path.join(PROJECT_ROOT, "data", "comparisons.pdf"))
#     plt.show()


# create_plots(
#     fontsize_title=30, fontsize_labels=25, fontsize_legend=20, fontsize_ticks=20
# )


# data = []
# metrics_columns = sorted(results[datasets[0]][models[0]]["metrics"].keys())

# # Loop through each dataset and model to gather metrics
# for dataset in datasets:
#     for model_name in models:
#         # Check if the dataset and model_name keys exist in the results dictionary
#         if dataset in results and model_name in results[dataset]:
#             # Retrieve the stored metrics for the current dataset and model
#             metrics = results[dataset][model_name]["metrics"]
#             # Append a new record including the dataset, model, and all metrics
#             data.append(
#                 [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
#             )
#         else:
#             # Handle the missing dataset/model combination by appending NaNs or placeholders
#             data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# # Assuming 'metrics_columns' is predefined or you define it based on your known metrics
# columns = ["Dataset", "Model"] + metrics_columns
# df = pd.DataFrame(data, columns=columns)
# filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]
# highlighted_df = filtered_df.style.highlight_max(
#     subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
# )
# highlighted_df


# data = []
# metrics_columns = sorted(results1[datasets[0]][models[0]]["metrics"].keys())
# for dataset in datasets:
#     for model_name in models:
#         # Check if the dataset and model_name keys exist in the results dictionary
#         if dataset in results1 and model_name in results1[dataset]:
#             # Retrieve the stored metrics for the current dataset and model
#             metrics = results1[dataset][model_name]["metrics"]
#             # Append a new record including the dataset, model, and all metrics
#             data.append(
#                 [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
#             )
#         else:
#             # Handle the missing dataset/model combination by appending NaNs or placeholders
#             data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# columns = ["Dataset", "Model"] + metrics_columns
# df = pd.DataFrame(data, columns=columns)
# filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]
# highlighted_df1 = filtered_df.style.highlight_max(
#     subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
# )
# highlighted_df1


def one_hot_to_indices(labels):
    indices = []
    for label in labels:
        index_array = np.where(label == 1)[0]
        if index_array.size > 0:
            indices.append(index_array[0])
        else:
            # Handle the case where no '1' is found; assign a default class, or handle as error
            indices.append(
                -1
            )  # Using -1 or any specific label to indicate the error/mislabeling
    return np.array(indices)


def visualize_tsne(embeddings, labels, title="t-SNE Visualization of GNN Embeddings"):
    tsne = TSNE(n_components=2, random_state=33)
    transformed_embeddings = tsne.fit_transform(embeddings)
    xs, ys = transformed_embeddings[:, 0], transformed_embeddings[:, 1]

    label_indices = one_hot_to_indices(
        labels
    )  # Convert one-hot encoded labels to indices

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xs, ys, c=label_indices, cmap="viridis")
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


# Call the visualization function
def plot_all_tsne(results, datasets, models):
    for dataset in datasets:
        for model in models:
            # Check if the model and dataset combination exists in results
            if dataset in results and model in results[dataset]:
                # Concatenate tensors for embeddings and labels
                all_embeddings = torch.cat(results[dataset][model]["embs"])
                all_labels = torch.cat(results[dataset][model]["labels"])

                # Convert labels from one-hot to indices if they are one-hot encoded
                all_labels_indices = one_hot_to_indices(all_labels)

                # Call the visualization function
                print(f"Visualizing t-SNE for {dataset} - {model}")
                visualize_tsne(
                    all_embeddings,
                    all_labels_indices,
                    title=f"t-SNE for {dataset} - {model}",
                )
            else:
                print(f"No results available for {dataset} - {model}")


plot_all_tsne(results, datasets, models)
