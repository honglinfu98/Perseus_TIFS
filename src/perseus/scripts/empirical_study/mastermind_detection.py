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
    get_valid_scored_signals,
)
from perseus.settings import PROJECT_ROOT


# with open(path.join(PROJECT_ROOT, "data", "results_f.pkl"), "rb") as file:
#     results_t = pickle.load(file)
def aggregate_and_compare_combined(
    gs_ls: dict, label_mapping_ls: dict, font_size: int, tick_size: int, bw: float
):
    """
    This function aggregates the graph features and separates them by group, then performs a T-test
    to compare the groups.

    The plotting here has been modified so that the curves have a fixed width (linewidth=2),
    the font sizes match the provided style, and the KDE plots:
      - do not include mean values in the legend,
      - show the mean as a vertical dashed bold line with an annotation placed to its left,
      - and have the legend placed right above the plot outside the plotting area with:
            * for Efficiency, only the Accomplices legend is shown (and the Mastermind mean is annotated to the right),
            * for Clustering Coefficient, only the Mastermind legend is shown.
    """

    # (Assuming that in_ego_graph, out_ego_graph, calculate_effsize_efficiency, and PROJECT_ROOT
    #  are defined elsewhere in your code.)

    # Initialize containers for centrality measures and additional features by group
    metrics_by_group = {
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
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        clustering_coeff = nx.clustering(graph)

        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                # Aggregate centrality measures
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

                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["Out Ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["eff_size"][group].append(eff_size)
                metrics_by_group["Efficiency"][group].append(efficiency)
                metrics_by_group["Density"][group].append(density)

    # Perform T-tests on the aggregated data for each metric
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        ttest_results[metric] = stats.ttest_ind(
            groups[0], groups[1], equal_var=False, nan_policy="omit"
        )

    # We will plot only these two metrics
    metrics_to_plot = [
        "Efficiency",
        "Clustering Coefficient",
    ]

    # Set global settings (without bold styling)
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=font_size, labelsize=font_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("legend", fontsize=font_size)

    # Define colors for the two groups so that lines and vertical means match
    group_colors = {"Accomplices": "lightblue", "Mastermind": "lightcoral"}

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        vertical_lines_info = (
            []
        )  # To store (mean, color, group_label) for annotation later

        # Define ordering for plotting.
        if metric == "Efficiency":
            groups_info = [
                (metrics_by_group[metric][1], "Mastermind"),
                (metrics_by_group[metric][0], "Accomplices"),
            ]
        else:  # Clustering Coefficient
            groups_info = [
                (metrics_by_group[metric][0], "Accomplices"),
                (metrics_by_group[metric][1], "Mastermind"),
            ]

        # Loop over the groups and plot the KDE curves
        for data, group_label in groups_info:
            mean_val = np.mean(data)
            color = group_colors[group_label]
            # Determine the legend label based on the metric:
            # For Efficiency, show only Accomplices; for Clustering Coefficient, only Mastermind.
            if metric == "Efficiency":
                legend_label = (
                    group_label if group_label == "Accomplices" else "_nolegend_"
                )
            else:
                legend_label = (
                    group_label if group_label == "Mastermind" else "_nolegend_"
                )

            sns.kdeplot(
                data,
                ax=ax,
                label=legend_label,
                bw_adjust=0.9,
                clip=(0, np.inf),
                linewidth=5,
                color=color,
            )
            # Draw a vertical dashed bold line at the mean
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=2)
            # Save the mean info along with the group label for later annotation
            vertical_lines_info.append((mean_val, color, group_label))

        # Annotate vertical lines with mean values.
        y_max = ax.get_ylim()[1]
        for mean_val, color, group_label in vertical_lines_info:
            # For Efficiency metric, place the Mastermind annotation to the right of the line.
            if metric == "Efficiency" and group_label == "Mastermind":
                offset = (5, 0)
                horizontal_alignment = "left"
            else:
                offset = (-5, 0)
                horizontal_alignment = "right"
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(mean_val, y_max * 0.95),
                xytext=offset,  # offset left or right
                textcoords="offset points",
                color=color,
                weight="bold",
                va="center",
                ha=horizontal_alignment,
            )

        # Set axis labels (without bold font weight)
        ax.set_ylabel(metric, fontsize=font_size)
        ax.set_xlabel("Value", fontsize=font_size)

        # Ensure tick labels are not bold
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("normal")
        ax.tick_params(axis="both", which="major", labelsize=font_size)

        # Add a grid with the desired style
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

        # Place the legend right above the plot outside the plotting area.
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=1,
            prop={"weight": "normal", "size": font_size},
            title_fontsize=font_size,
        )

        # Adjust layout so the legend does not overlap the plot
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        # Save and show the plot
        plt.savefig(path.join(PROJECT_ROOT, "data", f"distribution_{metric}.pdf"))
        plt.show()
        plt.close(fig)

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

    signals = get_new_detection()
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
                # "number_of_signals",  # osn
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

            with open(path.join(PROJECT_ROOT, "data", "results_wn.pkl"), "rb") as file:
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
    results, c = aggregate_and_compare_combined(gs, output_dict, 28, 28, 0.9)
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

    # top 10 keys have the highest number of ones in trandformed_predictions with their corresponding number of ones
    top_10 = sorted(
        [
            (key, sum(inner_dict.values()))
            for key, inner_dict in transformed_predictions.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    print(top_10)

    # I want to know the co appreace of inner dict keys with values of 1. Give me a matrix of the co appearance for the inner keys
    co_appearance = {}
    for token, inner_dict in transformed_predictions.items():
        for node, prediction in inner_dict.items():
            if prediction == 1:
                if node not in co_appearance:
                    co_appearance[node] = {}
                for other_node, other_prediction in inner_dict.items():
                    if other_prediction == 1 and other_node != node:
                        co_appearance[node][other_node] = (
                            co_appearance[node].get(other_node, 0) + 1
                        )

    # Display the co_appearance dictionary in a mrtrix form,
    # make the column and row names in the same order
    # filter the matrix with a threshold of 10 co appearance
    co_appearance_matrix = pd.DataFrame(co_appearance).fillna(0)
    co_appearance_matrix = co_appearance_matrix.reindex(
        sorted(co_appearance_matrix.columns), axis=1
    )
    co_appearance_matrix = co_appearance_matrix.reindex(
        sorted(co_appearance_matrix.columns), axis=0
    )
    co_appearance_matrix = co_appearance_matrix[
        co_appearance_matrix.columns[co_appearance_matrix.sum() > 10]
    ]
    co_appearance_matrix = co_appearance_matrix.loc[
        co_appearance_matrix.index[co_appearance_matrix.sum(axis=1) > 10]
    ]
    co_appearance_matrix

    # extract the nodes from co_appearance_matrix
    nodes = co_appearance_matrix.index

    # # Go over gs and create subgraphs that contain the nodes and the nodes neighbors
    # subgraphs = {}
    # for key, graph in gs.items():
    #     # find the neighbors of the nodes in the co_appearance_matrix in the graph
    #     nodes = co_appearance_matrix.index
    #     neighbors = set()
    #     for node in nodes:
    #         neighbors.update(graph.neighbors(node))
    #     nodes.update(neighbors)
    #     subgraph = graph.subgraph(nodes)
    #     subgraphs[key] = subgraph

    # # Plot the subgraphs
    # for key, subgraph in subgraphs.items():
    #     plt.figure(figsize=(10, 10))
    #     pos = nx.spring_layout(subgraph)
    #     nx.draw(subgraph, pos, with_labels=True, node_size=500, font_size=10)
    #     plt.title(key)
    #     plt.show()
