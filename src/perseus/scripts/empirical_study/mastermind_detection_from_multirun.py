import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from os import path
from perseus.settings import PROJECT_ROOT
from perseus.dataset.preprocess.train_test_validate import get_new_detection
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
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_curve
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)

# --- Get feature columns for DDM from dataset_preparation.py ---
DDM_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",
    "sum_targets_achieved",
    "rating",
    "ego_weighted_in_ratio",
    "ego_weighted_out_ratio",
    "ego_out_weights",
    "weighted_closeness_centrality",
    "weighted_betweenness_centrality",
    "weighted_pagerank",
    "ego_weighted_eff_size",
    "ego_weighted_efficiency",
    "weighted_clustering_coefficient",
    "ego_weighted_density",
]

DDINA_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",  # market
    "sum_targets_achieved",  # osn
    "rating",  # topological
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
]

# --- Load the trained model and outputs ---
with open(path.join(PROJECT_ROOT, "data", "sp", "model_saved.pkl"), "rb") as f:
    results = pickle.load(f)
model_info = results["DDINA"]["GraphSAGE"]
batch_size_key = list(model_info.keys())[0]
model_dict = model_info[batch_size_key]
model = model_dict["model"]
model.load_state_dict(model_dict["model_weights"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Find the best threshold using saved outputs ---
all_probs = model_dict["all_probs"]
all_labels = model_dict["all_labels"]

if all_probs.ndim > 1 and all_probs.shape[1] == 1:
    all_probs = all_probs.ravel()
    all_labels = all_labels.ravel()

precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

# Set your desired minimum precision
min_precision = 0.85  # adjust as needed

# Find indices where precision is above the minimum
high_precision_indices = np.where(precisions[:-1] >= min_precision)[0]

if len(high_precision_indices) > 0:
    # Compute F1 for these thresholds
    f1s = []
    for idx in high_precision_indices:
        t = thresholds[idx]
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        f1s.append(f1)
    f1s = np.array(f1s)
    best_idx = high_precision_indices[f1s.argmax()]
    best_threshold = thresholds[best_idx]
    best_macro_f1 = f1s.max()
    print(
        f"Best threshold with precision >= {min_precision}: {best_threshold:.3f}, Macro F1: {best_macro_f1:.3f}, Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}"
    )
else:
    print(
        f"No threshold found with precision >= {min_precision}. Consider lowering min_precision."
    )

# --- Get new detection signals ---
signals = get_new_detection()
signals["buffer_price_increase"] = signals["price_increase"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
)
signals["btc_base_return"] = signals["buffer_price_increase"].apply(
    lambda x: x["price_increase"] if isinstance(x, dict) else np.nan
)
signals.drop(columns=["buffer_price_increase"], inplace=True)
processed_signals = process_dataframe(signals)
ided_signals = assign_event_ids(processed_signals)
cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = aggregate_data(
    ided_signals
)
gs, P_theta = get_graphs(cascade_buffer, no_nodes_buffer, id_mapping_buffer)
graph_feature = graph_features(gs)
market_feature = features_engineer(processed_signals)
weighted_feature = compute_weighted_graph_features(P_theta)
combine_feature = combine_features(market_feature, graph_feature, weighted_feature)

all_predictions = {}

# Prepare a list of (asset_key, Data, node_ids) for batching
batch_data = []
for k, v in gs.items():
    try:
        features_buffer = combine_feature[k]
        features_buffer = features_buffer[
            features_buffer["telegram_chat_id"].isin(v.nodes)
        ]
        features_to_normalize = features_buffer[DDINA_FEATURE_COLUMNS]
        std = features_to_normalize.std()
        mean = features_to_normalize.mean()
        normalized_features = features_to_normalize.copy()
        non_zero_std = std != 0
        normalized_features.loc[:, non_zero_std] = (
            features_to_normalize.loc[:, non_zero_std] - mean[non_zero_std]
        ) / std[non_zero_std]
        normalized_features.loc[:, ~non_zero_std] = 0
        nan_columns = normalized_features.columns[
            normalized_features.isnull().any()
        ].tolist()
        if nan_columns:
            print(f"NaN values detected in key: {k}, Columns: {nan_columns}")
            continue
        node_attributes = torch.tensor(normalized_features.values, dtype=torch.float)
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(
                features_buffer["telegram_chat_id"]
            )
        }
        edge_index = []
        for source_node, target_node in v.edges():
            source = id_to_index[source_node]
            target = id_to_index[target_node]
            edge_index.append([source, target])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=node_attributes, edge_index=edge_index)
        node_ids = features_buffer["telegram_chat_id"].tolist()
        batch_data.append((k, data, node_ids))
    except Exception as e:
        print(e)
        continue

# Process in batches of 8
batch_size = 8
for i in range(0, len(batch_data), batch_size):
    batch = batch_data[i : i + batch_size]
    data_list = [item[1].to(device) for item in batch]
    asset_keys = [item[0] for item in batch]
    node_ids_list = [item[2] for item in batch]
    with torch.no_grad():
        output_list = model(data_list)
        for j, output in enumerate(output_list):
            predictions = (output > best_threshold).int()
            all_predictions[asset_keys[j]] = {}
            for idx, pred in enumerate(predictions.cpu().numpy()):
                node_id = node_ids_list[j][idx]
                all_predictions[asset_keys[j]][node_id] = pred

# --- Continue with mastermind detection as before ---
transformed_predictions = {}
for asset, nodes in all_predictions.items():
    transformed_predictions[asset] = {
        node: prediction for node, prediction in nodes.items()
    }

mastermind_set = {}
for asset, nodes in transformed_predictions.items():
    mastermind_set[asset] = {
        node for node, prediction in nodes.items() if prediction == 1
    }

output_dict = {}

for key, marked_nodes in mastermind_set.items():
    if key in gs:
        nodes = [i for i in gs[key].nodes]
        output_dict[key] = {node: (1 if node in marked_nodes else 0) for node in nodes}
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


# --- Example: Run your analysis/plotting functions as before ---
def aggregate_and_compare_combined(
    gs_ls: dict, label_mapping_ls: dict, font_size: int, tick_size: int, bw: float
):
    metrics_by_group = {
        "betweenness_centrality": [[], []],
        "closeness_centrality": [[], []],
        "pagerank": [[], []],
        "in_ratio": [[], []],
        "Out Ratio": [[], []],
        "out_nodes": [[], []],
        "Effective Size": [[], []],
        "Efficiency": [[], []],
        "Density": [[], []],
        "Clustering Coefficient": [[], []],
    }
    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        clustering_coeff = nx.clustering(graph)
        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
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
                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = (len(in_ego.nodes) - 1) / len(graph.nodes)
                out_ratio = (len(out_ego.nodes) - 1) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["Out Ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["Effective Size"][group].append(eff_size)
                metrics_by_group["Efficiency"][group].append(efficiency)
                metrics_by_group["Density"][group].append(density)
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        ttest_results[metric] = stats.ttest_ind(
            groups[0], groups[1], equal_var=False, nan_policy="omit"
        )

    # --- Plotting logic from mastermind_detection.py ---
    metrics_to_plot = [
        "Efficiency",
        "Out Ratio",
    ]
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=font_size, labelsize=font_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("legend", fontsize=font_size)
    group_colors = {"Accomplices": "lightblue", "Mastermind": "lightcoral"}
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        vertical_lines_info = []
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
        for data, group_label in groups_info:
            mean_val = np.mean(data)
            color = group_colors[group_label]
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
                bw_adjust=bw,
                clip=(0, np.inf),
                linewidth=5,
                color=color,
            )
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=2)
            vertical_lines_info.append((mean_val, color, group_label))
        y_max = ax.get_ylim()[1]
        for mean_val, color, group_label in vertical_lines_info:
            offset = (5, 0)
            horizontal_alignment = "left"
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(mean_val, y_max * 0.95),
                xytext=offset,
                textcoords="offset points",
                color=color,
                weight="bold",
                va="center",
                ha=horizontal_alignment,
            )
        ax.set_xlabel(metric, fontsize=font_size)
        ax.set_ylabel("", fontsize=font_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("normal")
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=1,
            prop={"weight": "normal", "size": font_size},
            title_fontsize=font_size,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(path.join(PROJECT_ROOT, "data", f"distribution_{metric}.pdf"))
        plt.show()
        plt.close(fig)
    return ttest_results, metrics_by_group


def assign_significance(p: float):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# --- Run the pooled analysis ---
results, c = aggregate_and_compare_combined(gs, output_dict, 28, 28, 0.9)

data = {
    "Metric": [],
    "Statistic": [],
    "P-value": [],
    "Degrees of Freedom": [],
    "Significance": [],
    "Group 0 Mean": [],
    "Group 1 Mean": [],
}
for metric, result in results.items():
    data["Metric"].append(metric)
    data["Statistic"].append(result.statistic)
    data["P-value"].append(f"{result.pvalue:.3g}")
    data["Degrees of Freedom"].append(getattr(result, "df", None))
    data["Significance"].append(assign_significance(result.pvalue))
    mean_group0 = np.mean(c[metric][0]) if c[metric][0] else np.nan
    mean_group1 = np.mean(c[metric][1]) if c[metric][1] else np.nan
    data["Group 0 Mean"].append(mean_group0)
    data["Group 1 Mean"].append(mean_group1)
df_results = pd.DataFrame(data)
df_results.set_index("Metric", inplace=True)
print(df_results)

print("Detection complete. Example output_dict:", output_dict)

# --- Save predictions and labels to pickle file (case_study style) ---
label_mapping = read_labeling_csv_back_to_dict("test")
combined_predictions = {}
for asset, nodes in all_predictions.items():
    combined_predictions[asset] = {}
    for node_id, pred in nodes.items():
        # Get label from label_mapping if available, else 0
        label = label_mapping.get(asset, {}).get(node_id, 0)
        # Save as tuple (prediction, label)
        combined_predictions[asset][node_id] = (int(pred), int(label))

with open(path.join(PROJECT_ROOT, "data", "sp", "predictions.pkl"), "wb") as file:
    pickle.dump(combined_predictions, file)
