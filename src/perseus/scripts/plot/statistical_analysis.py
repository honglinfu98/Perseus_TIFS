"""
This script is used to perform statistical analysis on the graph features and plot the distributions of the features by group
"""

from os import path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from perseus.dataset.dataset_preparation import combine_features, graph_features
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)

# from perseus.dataset.preprocess.groudtruth_labeling import create_label_mapping
from perseus.dataset.preprocess.process import (
    calculate_effsize_efficiency,
    compute_weighted_in_out_ratios,
    out_ego_graph,
    in_ego_graph,
)
from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
)

from perseus.dataset.preprocess.groudtruth_labeling import (
    create_label_mapping,
    read_labeling_csv_back_to_dict,
)
from perseus.settings import PROJECT_ROOT


def aggregate_and_compare_combined(gs_ls: dict, label_mapping_ls: dict):
    """
    This function aggregates the graph features and seperate them by group, then performs a T-test to compare the groups
    """

    # Initialize containers for centrality measures and additional features by group
    metrics_by_group = {
        "degree_centrality": [[], []],
        "betweenness_centrality": [[], []],
        "closeness_centrality": [[], []],
        "pagerank": [[], []],
        "in_ratio": [[], []],
        "out_ratio": [[], []],
        "out_nodes": [[], []],
        "eff_size": [[], []],
        "efficiency": [[], []],
        "density": [[], []],
        "clustering_coeff": [[], []],
    }

    # Iterate over each network and its corresponding labels
    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})

        # Calculate centrality measures for each node
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)

        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                # Aggregate centrality measures
                metrics_by_group["degree_centrality"][group].append(
                    degree_centrality.get(node, 0)
                )
                metrics_by_group["betweenness_centrality"][group].append(
                    betweenness_centrality.get(node, 0)
                )
                metrics_by_group["closeness_centrality"][group].append(
                    closeness_centrality.get(node, 0)
                )
                metrics_by_group["pagerank"][group].append(pagerank.get(node, 0))

                # Calculate and aggregate additional graph features
                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = len(in_ego.nodes) / len(graph.nodes)
                out_ratio = len(out_ego.nodes) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                clustering_coeff = nx.clustering(graph, node)

                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["out_ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["eff_size"][group].append(eff_size)
                metrics_by_group["efficiency"][group].append(efficiency)
                metrics_by_group["density"][group].append(density)
                metrics_by_group["clustering_coeff"][group].append(clustering_coeff)

    # Perform T-tests on the aggregated data for each metric
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        ttest_results[metric] = stats.ttest_ind(
            groups[0], groups[1], equal_var=False, nan_policy="omit"
        )

    return ttest_results, metrics_by_group


# Function to process signals and return necessary components for analysis
def process_signals_and_get_components(signal_function: pd.DataFrame):
    """
    This script is used to process the signals and return the necessary components for analysis
    """
    signals = signal_function()  # Get signals using the provided function
    processed_signals = process_dataframe(signals)  # Process signals
    ided_signals = assign_event_ids(processed_signals)  # Assign event IDs
    cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(
        ided_signals
    )  # Aggregate data
    gs, results, As, P_dict, P_com = get_graphs(
        cascade, no_nodes, id_mapping
    )  # Get graphs
    # graph_feature = graph_features(gs)  # Get graph features
    # market_feature = features_engineer(processed_signals)  # Engineer market features
    # weighted_feature = compute_weighted_in_out_ratios(P_com)

    # combine_feature = combine_features(
    #     market_feature, graph_feature, weighted_feature
    # )  # Combine features
    label_mapping = read_labeling_csv_back_to_dict(
        "test"
    )  # Read label mapping (adjust as needed)
    return gs, label_mapping


# Define a function to assign significance levels
def assign_significance(p: float):
    """
    This script is used to assign significance levels to p-values
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def plot_distribution(
    gs_ls: dict,
    label_mapping_ls: dict,
    font_size=10,
    title_size=12,
    legend_size=5,
    tick_label_size=10,
    output_dir=path.join(PROJECT_ROOT, "data"),
):
    """
    This function is used to plot the distribution of graph features by group, including mean and variance in the legend.
    """
    metrics_by_group = {
        "degree_centrality": [[], []],
        "betweenness_centrality": [[], []],
        "Closeness Centrality": [[], []],
        "pagerank": [[], []],
        "In Ratio": [[], []],
        "Out Ratio": [[], []],
        "out_nodes": [[], []],
        "eff_size": [[], []],
        "Efficiency": [[], []],
        "density": [[], []],
        "clustering_coeff": [[], []],
    }

    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)

        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                metrics_by_group["degree_centrality"][group].append(
                    degree_centrality.get(node, 0)
                )
                metrics_by_group["betweenness_centrality"][group].append(
                    betweenness_centrality.get(node, 0)
                )
                metrics_by_group["Closeness Centrality"][group].append(
                    closeness_centrality.get(node, 0)
                )
                metrics_by_group["pagerank"][group].append(pagerank.get(node, 0))

                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = len(in_ego.nodes) / len(graph.nodes)
                out_ratio = len(out_ego.nodes) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                clustering_coeff = nx.clustering(graph, node)

                metrics_by_group["In Ratio"][group].append(in_ratio)
                metrics_by_group["Out Ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["eff_size"][group].append(eff_size)
                metrics_by_group["Efficiency"][group].append(efficiency)
                metrics_by_group["density"][group].append(density)
                metrics_by_group["clustering_coeff"][group].append(clustering_coeff)

    metrics_to_plot = ["Closeness Centrality", "In Ratio", "Out Ratio", "Efficiency"]

    # Set global settings
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=title_size, labelsize=font_size)
    plt.rc("xtick", labelsize=tick_label_size)
    plt.rc("ytick", labelsize=tick_label_size)
    plt.rc("legend", fontsize=legend_size)

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(6, 5))  # Explicitly creating a figure with axes
        group0 = metrics_by_group[metric][0]
        group1 = metrics_by_group[metric][1]
        mean0, var0 = np.mean(group0), np.var(group0)
        mean1, var1 = np.mean(group1), np.var(group1)
        sns.kdeplot(
            group0, ax=ax, label=f"Mastermind:\nMean:{mean0:.2f}, Variance:{var0:.2f}"
        )
        sns.kdeplot(
            group1,
            ax=ax,
            label=f"Non-mastermind:\nMean:{mean1:.2f}, Variance:{var1:.2f}",
        )
        # ax.set_title(metric)
        ax.set_ylabel("Density")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1
        )  # Adjust legend position and ncol for vertical layout
        plt.tight_layout()  # Adjust layout
        plt.savefig(path.join(output_dir, f"distribution_{metric}.pdf"))
        plt.close(fig)  # Close the figure to free memory

    return metrics_by_group


def plot_closeness_roc(metrics_by_group):
    closeness_group_0 = metrics_by_group["closeness_centrality"][0]
    closeness_group_1 = metrics_by_group["closeness_centrality"][1]

    labels = np.array([0] * len(closeness_group_0) + [1] * len(closeness_group_1))
    scores = np.array(closeness_group_0 + closeness_group_1)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Check if AUC is less than 0.5 and invert scores if so
    if roc_auc < 0.5:
        scores = 1 - scores  # Invert the scores
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)  # Recalculate AUC with inverted scores

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


# Main execution block
if __name__ == "__main__":
    # Process each dataset
    gs_train, label_mapping_train = process_signals_and_get_components(
        get_train_scored_signals
    )
    gs_valid, label_mapping_valid = process_signals_and_get_components(
        get_valid_scored_signals
    )
    gs_test, label_mapping_test = process_signals_and_get_components(
        get_test_scored_signals
    )

    # Combine all graphs and label mappings
    all_gs = {**gs_train, **gs_valid, **gs_test}
    all_label_mappings = {
        **label_mapping_train,
        **label_mapping_valid,
        **label_mapping_test,
    }

    # Running the pooled analysis
    results, c = aggregate_and_compare_combined(all_gs, all_label_mappings)
    plot_distribution(all_gs, all_label_mappings, 25, 25, 17, 25)

    # Transform the t-test results into a format suitable for DataFrame construction
    data = {
        "Metric": [],
        "Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "Significance": [],  # New column for significance marks
    }

    for metric, result in results.items():
        data["Metric"].append(metric)
        data["Statistic"].append(result.statistic)
        data["P-value"].append(f"{result.pvalue:.3g}")  # Format p-value for readability
        # Include degrees of freedom if available
        data["Degrees of Freedom"].append(
            getattr(result, "df", None)
        )  # Use getattr for compatibility
        # Assign significance level
        data["Significance"].append(assign_significance(result.pvalue))

    # Create a DataFrame
    df_results = pd.DataFrame(data)

    # Optionally, set the Metric column as the index for better presentation
    df_results.set_index("Metric", inplace=True)

    # Show or return the DataFrame
    print(df_results)

    plot_closeness_roc(c)
