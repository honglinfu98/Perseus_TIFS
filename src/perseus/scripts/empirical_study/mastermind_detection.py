"""
This function is used to create a case study for the empirical study. It is used to demonstrate how to use the trained model to make predictions on a single cascade.
"""

from os import path
import pickle
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats
import pandas as pd
import seaborn as sns


from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
    combine_features,
    graph_features,
    out_ego_graph,
    in_ego_graph,
    calculate_effsize_efficiency,
    compute_weighted_graph_features,
)
from perseus.dataset.preprocess.train_test_validate import (
    get_new_detection,
    get_test_scored_signals,
    get_train_scored_signals,
)
from perseus.settings import PROJECT_ROOT


# with open(path.join(PROJECT_ROOT, "data", "results_f.pkl"), "rb") as file:
#     results_t = pickle.load(file)


def aggregate_and_compare_combined(
    gs_ls: dict, label_mapping_ls: dict, font_size: 24, tick_size: 24, bw: 0.5
):
    """
    This function aggregates the graph features and seperate them by group, then performs a T-test to compare the groups
    """

    # Initialize containers for centrality measures and additional features by group
    metrics_by_group = {
        # "degree_centrality": [[], []],
        "betweenness_centrality": [[], []],
        "closeness_centrality": [[], []],
        "pagerank": [[], []],
        "in_ratio": [[], []],
        "Out Ratio": [[], []],
        "out_nodes": [[], []],
        "eff_size": [[], []],
        "Efficiency": [[], []],
        "Density": [[], []],
        "Clustering Coefficient": [[], []],
    }

    # Iterate over each network and its corresponding labels
    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})

        # Calculate centrality measures for each node
        # degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        clustering_coeff = nx.clustering(graph)

        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                # Aggregate centrality measures
                # metrics_by_group["degree_centrality"][group].append(
                #     degree_centrality.get(node, 0)
                # )
                metrics_by_group["betweenness_centrality"][group].append(
                    betweenness_centrality.get(node, 0)
                )
                metrics_by_group["closeness_centrality"][group].append(
                    closeness_centrality.get(node, 0)
                )
                metrics_by_group["pagerank"][group].append(pagerank.get(node, 0))

                metrics_by_group["Clustering Coefficient"][group].append(
                    clustering_coeff.get(node, 0)
                )

                # Calculate and aggregate additional graph features
                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = (len(in_ego.nodes) - 1) / len(graph.nodes)
                out_ratio = (len(out_ego.nodes) - 1) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                # clustering_coeff = nx.clustering(graph, node)

                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["Out Ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["eff_size"][group].append(eff_size)
                metrics_by_group["Efficiency"][group].append(efficiency)
                metrics_by_group["Density"][group].append(density)
                # metrics_by_group["clustering_coeff"][group].append(clustering_coeff)

    # Perform T-tests on the aggregated data for each metric
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        ttest_results[metric] = stats.ttest_ind(
            groups[0], groups[1], equal_var=False, nan_policy="omit"
        )

    metrics_to_plot = [
        # "Density",
        # "Clustering Coefficient",
        # "Out Ratio",
        # "Efficiency",
        # # "eff_size",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
        "in_ratio",
        "Out Ratio",
        "out_nodes",
        "eff_size",
        "Efficiency",
        "Density",
        "Clustering Coefficient",
    ]

    # Set global settings
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=font_size, labelsize=font_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("legend", fontsize=font_size)

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 8))  # Explicitly creating a figure with axes
        group0 = metrics_by_group[metric][0]
        group1 = metrics_by_group[metric][1]
        mean0, var0 = np.mean(group0), np.var(group0)
        mean1, var1 = np.mean(group1), np.var(group1)
        sns.kdeplot(
            group0,
            ax=ax,
            label=f"Accomplices:\nMean:{mean0:.2f}, Variance:{var0:.2f}",
            bw_adjust=0.7,
            clip=(0, np.inf),
        )
        sns.kdeplot(
            group1,
            ax=ax,
            label=f"Mastermind:\nMean:{mean1:.2f}, Variance:{var1:.2f}",
            bw_adjust=0.7,
            clip=(0, np.inf),
        )

        # ax.set_title(metric)
        ax.set_ylabel(f"{metric}", fontsize=font_size)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1
        )  # Adjust legend position and ncol for vertical layout
        plt.tight_layout()  # Adjust layout
        plt.savefig(
            path.join(path.join(PROJECT_ROOT, "data", f"distribution_{metric}.pdf"))
        )
        plt.show()
        plt.close(fig)  # Close the figure to free memory

    return ttest_results, metrics_by_group


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


if __name__ == "__main__":

    signals = get_train_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = (
        aggregate_data(ided_signals)
    )
    gs, results, As, P_dict, P_com = get_graphs(
        cascade_buffer, no_nodes_buffer, id_mapping_buffer
    )
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    weighted_feature = compute_weighted_graph_features(P_com)
    combine_feature = combine_features(market_feature, graph_feature, weighted_feature)
    all_predictions = {}

    for k, v in gs.items():
        try:

            features_buffer = combine_feature[k]
            features_buffer = features_buffer[
                features_buffer["telegram_chat_id"].isin(v.nodes)
            ]

            # Selecting feature columns for normalization
            feature_columns = [
                # "average_speed",  # market
                # "sum_total_targets",  # osn
                "average_increase_percentage",  # market
                "number_of_signals",  # osn
                "sum_targets_achieved",  # osn
                "rating",  # topological
                # "in_ratio",  # topological
                # "out_ratio",  # topological
                # "out_nodes",  # topological
                # "density",  # topological
                # "clustering_coeff",  # topological
                # "closeness_centrality",  # topological
                # "eff_size",  # topological
                # "efficiency",  # topological
                # "betweenness_centrality",
                # "pagerank",
                "ego_in_ratio",
                "ego_out_ratio",
                "ego_out_nodes",
                "eff_size",
                "efficiency",
                "density",
                "clustering_coeff",
                "closeness_centrality",
                "pagerank",
                "betweenness_centrality",
                # "ego_weighted_in_ratio",
                # "ego_weighted_out_ratio",
                # "ego_out_weights",
                # "weighted_closeness_centrality",
                # "weighted_betweenness_centrality",
                # "weighted_pagerank",
                # "ego_weighted_eff_size",
                # "ego_weighted_efficiency",
                # "weighted_clustering_coefficient",
                # "ego_weighted_density",
            ]

            features_to_normalize = features_buffer[feature_columns]
            std = features_to_normalize.std()
            mean = features_to_normalize.mean()

            # Normalize, but handle cases where std is zero
            normalized_features = features_to_normalize.copy()

            # Only normalize features where std is non-zero
            non_zero_std = std != 0
            normalized_features.loc[:, non_zero_std] = (
                features_to_normalize.loc[:, non_zero_std] - mean[non_zero_std]
            ) / std[non_zero_std]

            # For features with zero std, assign them a constant value (like 0)
            normalized_features.loc[:, ~non_zero_std] = 0

            nan_columns = normalized_features.columns[
                normalized_features.isnull().any()
            ].tolist()
            if nan_columns:
                print(f"NaN values detected in key: {k}, Columns: {nan_columns}")
                continue

            node_attributes = torch.tensor(
                normalized_features.values, dtype=torch.float
            )

            # Map node IDs to indices
            id_to_index = {
                telegram_chat_id: index
                for index, telegram_chat_id in enumerate(
                    features_buffer["telegram_chat_id"]
                )
            }

            # Create edge index
            edge_index = []
            for source_node, target_node in v.edges():
                source = id_to_index[source_node]
                target = id_to_index[target_node]
                edge_index.append([source, target])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            data = Data(x=node_attributes, edge_index=edge_index)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with open(
                path.join(PROJECT_ROOT, "data", "saved", "results_l.pkl"), "rb"
            ) as file:
                results_t = pickle.load(file)

            model_1 = results_t["DDINA"]["GraphSAGE"]["model"]
            model_1.load_state_dict(results_t["DDINA"]["GraphSAGE"]["model_weights"])
            model_1.to(device)

            # Extract node IDs in the order they are being processed
            node_ids = features_buffer["telegram_chat_id"].tolist()

            model_1.eval()  # Set the model to evaluation mode

            all_predictions[k] = {}

            with torch.no_grad():
                data = data.to(device)
                output, _ = model_1(data.x, data.edge_index)
                # probs = torch.cat(output, dim=0).sigmoid().numpy()

                predictions = (output > 0.55).int()

                for i, pred in enumerate(predictions.cpu().numpy()):
                    node_id = node_ids[i]
                    # actual_label = label_mapping[k].get(
                    # #     node_id, 0
                    # # )  # Default to 0 if not found
                    all_predictions[k][node_id] = pred

            # Now `all_predictions` is a list of tuples, each containing (node_id, predicted_label, actual_label)
            # print(all_predictions)
            # print(k)
        except Exception as e:
            print(e)
            continue

    # Transformation logic
    transformed_predictions = {}
    for asset, nodes in all_predictions.items():
        transformed_predictions[asset] = {
            node: prediction[0] for node, prediction in nodes.items()
        }

    # Display the transformed data
    transformed_predictions

    # The number of masterminds in total
    total_sum = sum(
        sum(inner_dict.values()) for inner_dict in transformed_predictions.values()
    )
    total_sum

    # The number of accomplices in total
    zero_count = 0
    # Iterating over the tokens
    for token, inner_dict in transformed_predictions.items():
        zero_count += list(inner_dict.values()).count(0)
    zero_count

    mastermind_set = {}
    for asset, nodes in transformed_predictions.items():
        mastermind_set[asset] = {
            node for node, prediction in nodes.items() if prediction == 1
        }

    # # Initialize the output dictionary
    output_dict = {}

    # Loop through each key in mastermind_set
    for key, marked_nodes in mastermind_set.items():
        # Check if the key exists in gs and get the corresponding nodes
        if key in gs:
            nodes = [i for i in gs[key].nodes]
            # Create a sub-dictionary for each node, marking it as 1 or 0
            output_dict[key] = {
                node: (1 if node in marked_nodes else 0) for node in nodes
            }

        # Print or return the output_dict
        # print(output_dict)

    # Running the pooled analysis
    results, c = aggregate_and_compare_combined(gs, output_dict, 24, 24, 0.9)
    # plot_distribution(gs, output_dict, 25, 25, 17, 25)

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

    # Summing up all the values in the dictionaries nested inside the main dictionary.
    total_sum = sum(
        sum(inner_dict.values()) for inner_dict in transformed_predictions.values()
    )
    total_sum

    # Dictionary to hold the count of tokens with 1, 2, 3, etc., '1's
    ones_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    # Iterating over the tokens
    for token, inner_dict in transformed_predictions.items():
        count_of_ones = list(inner_dict.values()).count(1)
        if (
            1 <= count_of_ones <= 5
        ):  # We are only interested in those tokens with 1 to 5 ones
            ones_count[count_of_ones] += 1

    ones_count
