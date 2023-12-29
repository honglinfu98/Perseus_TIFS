from os import path
import pickle
import pandas as pd

from torch_geometric.data import Data
import torch
from pygod.detector import DOMINANT, AnomalyDAE, CONAD, OCGNN
import scipy.stats as stats


from clotho.settings import PROJECT_ROOT
from clotho.dataset.extract.cloudburst_connection import get_scored_signals

from clotho.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
    get_graphs,
)

import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# import pickle

# # Load features.pkl
# with open('features.pkl', 'rb') as file:
#     features = pickle.load(file)


def calculate_effsize_efficiency(G, ego):
    # Get the ego network
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network (excluding ego)
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)
    # Inside your calculate_effsize_efficiency function:
    avg_degree = 0  # Default to 0
    if num_alters > 0:
        avg_degree = (
            sum(
                ego_net.degree(n) - (1 if ego_net.has_edge(n, ego) else 0)
                for n in alters
            )
            / num_alters
        )

    # Calculate effective size
    eff_size = num_alters - avg_degree

    # Calculate efficiency
    efficiency = eff_size / num_alters if num_alters > 0 else 0

    return eff_size, efficiency


def out_ego_graph(G, node, radius=1):
    """
    Extract the out-ego network of a specified node in a directed graph.
    """
    # Extract the out-ego network
    out_ego = nx.ego_graph(G, node, radius=radius)
    return out_ego


def in_ego_graph(G, node, radius=1):
    """
    Extract the in-ego network of a specified node in a directed graph.
    """
    # Reverse the graph
    G_reverse = G.reverse(copy=True)
    # Extract the in-ego network
    in_ego = nx.ego_graph(G_reverse, node, radius=radius)
    return in_ego


# # Load graph.pkl
# with open('graph.pkl', 'rb') as file:
#     graph_nx = pickle.load(file)
# Function to split data and calculate mean
def calculate_mean(data):
    count = len(data)
    top_25_index = int(count * 0.25)

    # head_data = data.nlargest(top_25_index, column)
    # tail_data = data.nsmallest(count - top_25_index, column)

    head_data = data.iloc[:top_25_index]
    tail_data = data.iloc[count - top_25_index :]

    return head_data.mean(), tail_data.mean()


def plot_aggregated_data(dfs, label_fontsize=28, tick_fontsize=10, title_fontsize=14):
    # Storing mean values (assuming aggregate_data_eff is defined elsewhere)
    aggregate_means, first_ranked_values = aggregate_data_eff(dfs)

    # Plotting
    labels = [
        "Mastermind",
        "Top 25% by probability ranking",
        "Bottom 75% by probability ranking",
    ]
    x = np.arange(len(labels))  # the label locations

    width = 0.25  # the width of the bars

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Groups", fontsize=label_fontsize)
    ax1.set_ylabel("Effective Size", color=color, fontsize=label_fontsize)
    rects1 = ax1.bar(
        x - width / 2,
        [
            first_ranked_values["eff_size"],
            aggregate_means["head_eff_size"],
            aggregate_means["tail_eff_size"],
        ],
        width,
        label="Eff Size",
        color=color,
    )
    ax1.tick_params(axis="y", labelcolor=color, labelsize=tick_fontsize)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel("Efficiency", color=color, fontsize=label_fontsize)
    rects2 = ax2.bar(
        x + width / 2,
        [
            first_ranked_values["efficiency"],
            aggregate_means["head_efficiency"],
            aggregate_means["tail_efficiency"],
        ],
        width,
        label="Efficiency",
        color=color,
    )
    ax2.tick_params(axis="y", labelcolor=color, labelsize=tick_fontsize)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=tick_fontsize)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(path.join(PROJECT_ROOT, "data", "effective.pdf"))
    plt.show()


# Helper function to aggregate data
def aggregate_data_eff(dfs):
    aggregate_means = {
        "head_eff_size": 0,
        "tail_eff_size": 0,
        "head_efficiency": 0,
        "tail_efficiency": 0,
    }
    first_ranked_values = {"eff_size": 0, "efficiency": 0}
    num_keys = len(dfs)

    for key, df in dfs.items():
        head_mean, tail_mean = calculate_mean(df)
        first_ranked = df.iloc[0]

        aggregate_means["head_eff_size"] += head_mean["eff_size"]
        aggregate_means["tail_eff_size"] += tail_mean["eff_size"]
        aggregate_means["head_efficiency"] += head_mean["efficiency"]
        aggregate_means["tail_efficiency"] += tail_mean["efficiency"]

        first_ranked_values["eff_size"] += first_ranked["eff_size"]
        first_ranked_values["efficiency"] += first_ranked["efficiency"]

    # Averaging the aggregated means
    for key in aggregate_means:
        aggregate_means[key] /= num_keys

    # Averaging the first-ranked values
    for key in first_ranked_values:
        first_ranked_values[key] /= num_keys

    return aggregate_means, first_ranked_values


def plot_heatmap(scores: dict):
    # Your top 10 cryptocurrencies
    top_15 = [
        "BTC",
        "ETH",
        "ADA",
        "XRP",
        "DOGE",
        "DOT",
        "MATIC",
        "LTC",
        "AVAX",
        "BCH",
        "XLM",
        "XMR",
        "ATOM",
        "UNI",
        "ETC",
    ]

    # Extracting the first telegram_chat_id and score for each cryptocurrency in the top 10
    data = {"crypto": [], "telegram_chat_id": [], "score": []}

    for crypto, df in scores.items():
        if crypto in top_15:
            first_row = df.iloc[0]
            data["crypto"].append(crypto)
            data["telegram_chat_id"].append(first_row["telegram_chat_id"])
            data["score"].append(first_row["score"])

    # Creating a DataFrame from the extracted data
    heatmap_data = pd.DataFrame(data)

    # Pivot the data for the heatmap
    heatmap_data = heatmap_data.pivot("crypto", "telegram_chat_id", "score")

    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap of Mastermind Scores")
    plt.xlabel("Mastermind ID")
    plt.ylabel("Cryptocurrency")

    plt.savefig(path.join(PROJECT_ROOT, "data", "heatmap.pdf"))

    plt.show()


def anamoly_detection(graphs_dict: dict, features_dict: dict):
    mastermind_score = {}

    for coin, graph_nx in graphs_dict.items():
        features_all = features_dict[coin]
        features = features_all[features_all["telegram_chat_id"].isin(graph_nx.nodes)]

        # Selecting feature columns for normalization
        feature_columns = [
            "average_increase_percentage",
            # "average_speed",
            "number_of_signals",
            # "latest_chat_crowd_score",
            # "rating",
        ]

        # Extracting features to be normalized
        features_to_normalize = features[feature_columns]

        # Normalization (Min-Max Scaling)
        # normalized_features_min_max = (features_to_normalize - features_to_normalize.min()) / (features_to_normalize.max() - features_to_normalize.min())

        # OR Normalization (Z-score Normalization)
        normalized_features_z_score = (
            features_to_normalize - features_to_normalize.mean()
        ) / features_to_normalize.std()

        # Convert normalized features DataFrame to tensor
        node_attributes = torch.tensor(
            normalized_features_z_score.values, dtype=torch.float
        )  # Use normalized_features_z_score for Z-score Normalization

        # Mapping telegram_chat_id to row index in DataFrame
        id_to_index = {
            telegram_chat_id: index
            for index, telegram_chat_id in enumerate(features["telegram_chat_id"])
        }

        edge_index = []
        # Assuming graph is a NetworkX graph object
        for edge in graph_nx.edges(data=True):
            source = id_to_index[edge[0]]
            target = id_to_index[edge[1]]
            edge_index.append((source, target))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=node_attributes, edge_index=edge_index)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = data.to(device)

        model = DOMINANT(num_layers=2)
        # model = AnomalyDAE()
        # model = OCGNN()
        # model = CONAD()

        model = model.fit(graph)

        scores = model.decision_score_.numpy()

        # Create a DataFrame with telegram_chat_id and their corresponding scores
        score_df = pd.DataFrame(
            {"telegram_chat_id": features["telegram_chat_id"], "score": scores}
        )

        # Sort the DataFrame by scores
        sorted_score_df = score_df.sort_values(by="score", ascending=False)

        mastermind_score[coin] = sorted_score_df

    return mastermind_score


def map_the_id_back_cascade(cascade, id_mapping):
    new_cascades = {}
    for algo, mappings in cascade.items():
        new_cascades[algo] = []
        for mapping in mappings:
            new_mapping = {}
            for key, value in mapping.items():
                if key == "T":
                    new_mapping[key] = value
                else:
                    new_mapping[id_mapping[algo]["new_to_id"][key]] = value
            new_cascades[algo].append(new_mapping)

    return new_cascades


def plot_summary(gs):
    # Aggregating degree data across all graphs
    all_degrees = []
    for graph in gs.values():
        degrees = [graph.degree(n) for n in graph.nodes()]
        all_degrees.extend(degrees)

    # Aggregating edge and node data across all graphs
    all_edges = []
    all_nodes = []
    for graph in gs.values():
        all_edges.append(graph.number_of_edges())
        all_nodes.append(graph.number_of_nodes())

    # Creating subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Histogram for degrees
    axs[0].hist(all_degrees, bins=range(min(all_degrees), max(all_degrees) + 1))
    axs[0].set_title("Aggregated Histogram of Node Degrees", fontsize=18)
    axs[0].set_xlabel("Degree", fontsize=18)
    axs[0].set_ylabel("Frequency", fontsize=18)
    axs[0].tick_params(axis="both", which="major", labelsize=18)

    # Histogram for edges
    axs[1].hist(
        all_edges, bins=range(min(all_edges), max(all_edges) + 1), color="skyblue"
    )
    axs[1].set_title("Histogram of Number of Edges", fontsize=18)
    axs[1].set_xlabel("Number of Edges", fontsize=18)
    axs[1].set_ylabel("Frequency", fontsize=18)
    axs[1].tick_params(axis="both", which="major", labelsize=18)

    # Histogram for nodes
    axs[2].hist(
        all_nodes, bins=range(min(all_nodes), max(all_nodes) + 1), color="salmon"
    )
    axs[2].set_title("Histogram of Number of Nodes", fontsize=18)
    axs[2].set_xlabel("Number of Nodes", fontsize=18)
    axs[2].set_ylabel("Frequency", fontsize=18)
    axs[2].tick_params(axis="both", which="major", labelsize=18)

    # Displaying the plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "summary.pdf"))
    plt.show()


def plot_cascade(cascade_old_id):
    # Initializing lists to hold aggregated data
    total_durations = []
    num_senders = []

    # Aggregating data from all groups
    for group, cascades in cascade_old_id.items():
        for cascade in cascades:
            total_durations.append(cascade["T"])
            num_senders.append(len(cascade) - 1)  # Subtracting 1 to exclude 'T'

    # Creating histograms
    plt.figure(figsize=(12, 6))

    # Histogram for number of senders
    plt.subplot(1, 2, 1)
    plt.hist(
        num_senders,
        bins=range(0, max(num_senders) + 2),
        color="red",
        alpha=0.7,
        edgecolor="black",
    )
    plt.title("Histogram of Number of Senders", fontsize=28)
    plt.xlabel("Number of Senders", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Histogram for total durations
    plt.subplot(1, 2, 2)
    plt.hist(total_durations, bins=10, color="blue", alpha=0.7, edgecolor="black")
    plt.title("Histogram of Total Durations", fontsize=28)
    plt.xlabel("Total Duration", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Displaying the plot
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "cascade.pdf"))
    plt.show()


def count_top_three_senders(cascades, top_three):
    counts = [0, 0, 0]  # Counts for the first, second, and third scorers
    for cascade in cascades:
        first_sender = next(iter(cascade))
        for i, person in enumerate(top_three):
            if first_sender == person:
                counts[i] += 1
    return counts


def plot_comparison(cascade_old_id, scores):
    # Setting aesthetics for the plots
    sns.set(style="whitegrid")

    # Initializing variables
    total_cascades = 0
    top_three_counts = [0, 0, 0]
    total_senders_count = 0

    # Iterate over each group in scores
    for group in scores.keys():
        if group in cascade_old_id:
            cascades = cascade_old_id[group]
            total_cascades += len(cascades)
            top_three = scores[group].iloc[0:3, 0].tolist()
            counts = count_top_three_senders(cascades, top_three)
            top_three_counts = [sum(x) for x in zip(top_three_counts, counts)]
            total_senders_count += len(scores[group])

    # Creating a single figure for both pie charts
    plt.figure(figsize=(14, 7))

    # Creating the pie chart for the aggregated proportion of cascades initiated by the top three senders
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.pie(
        [top_three_counts[0], total_cascades - top_three_counts[0]],
        labels=["Mastermind", "Others"],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 28},
    )
    # plt.title("Proportion of cascades initiated by masterminds", fontsize=28)
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Creating the pie chart for the aggregated proportion of the top three senders compared to total senders
    plt.subplot(1, 2, 2)
    plt.pie(
        [len(scores), total_senders_count - len(scores)],
        labels=["Mastermind", "Others"],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 28},
    )
    # plt.title("Number of masterminds compared to other channels", fontsize=28)
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Displaying the plots
    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "comparison.pdf"))
    plt.show()


if __name__ == "__main__":
    signals = get_scored_signals()
    # with open(path.join(PROJECT_ROOT, "data", "signals.pkl"), "rb") as file:
    # signals = pickle.load(file)
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)

    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    cascade_old_id = map_the_id_back_cascade(cascade, id_mapping)
    # for each key in the cascade_old_id, find the longest length

    features = features_engineer(processed_signals)
    scores = anamoly_detection(gs, features)

    dfs = {}

    # Looping through each key in the scores dictionary
    for key in scores.keys():
        # Creating lists to store node_ids, in-ego ratios, and out-ego ratios
        node_ids = []
        in_ratios = []
        out_ratios = []
        out_nodes = []
        eff_sizes = []
        efficiencies = []
        density = []
        clustering_coeffs = []

        for i in range(len(scores[key])):
            node_id = scores[key]["telegram_chat_id"].iloc[i]
            node_ids.append(node_id)

            # Calculating the in-ego ratio
            inn = len(in_ego_graph(gs[key], node_id).nodes) / len(gs[key])
            in_ratios.append(inn)

            # Calculating the out-ego ratio
            outt = len(out_ego_graph(gs[key], node_id).nodes) / len(gs[key])
            out_ratios.append(outt)

            out_nodes.append(len(out_ego_graph(gs[key], node_id).nodes))

            # Calculate the density of the ego network
            den = nx.density(out_ego_graph(gs[key], node_id))
            density.append(den)

            eff_size, efficiency = calculate_effsize_efficiency(gs[key], node_id)
            eff_sizes.append(eff_size)
            efficiencies.append(efficiency)
            # Calculate clustering coefficient for the node
            clustering_coeff = nx.clustering(gs[key], node_id)
            clustering_coeffs.append(clustering_coeff)

        # Creating a DataFrame for each key and storing it in the dfs dictionary
        dfs[key] = pd.DataFrame(
            {
                "telegram_chat_id": node_ids,
                "in_ratio": in_ratios,
                "out_ratio": out_ratios,
                "out_nodes": out_nodes,
                "eff_size": eff_sizes,
                "efficiency": efficiencies,
                "density": density,
                "clustering_coeff": clustering_coeffs,
            }
        )

        # List to store all data
    all_data = []

    # Iterate through each commodity
    for commodity in dfs.keys():
        # Merge dataframes on telegram_chat_id
        merged_df = pd.merge(features[commodity], dfs[commodity], on="telegram_chat_id")
        # Optionally, you could add a 'commodity' column to keep track of the commodity each row corresponds to:
        # merged_df['commodity'] = commodity
        all_data.append(merged_df)

    # Concatenate all data into a single DataFrame
    all_data_df = pd.concat(all_data)

    # Calculate Pearson correlation and the number of observations
    r = all_data_df["average_increase_percentage"].corr(all_data_df["out_ratio"])
    n = len(all_data_df)

    # Calculate the t statistic
    t_statistic = (r * ((n - 2) ** 0.5)) / ((1 - r**2) ** 0.5)

    # Calculate the degrees of freedom
    df = n - 2

    # Find the p-value for two-tailed test
    p_value = (
        stats.t.sf(abs(t_statistic), df) * 2
    )  # Multiplied by 2 for a two-tailed test

    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the result
    alpha = 0.1  # Assuming a 10% significance level
    if p_value < alpha:
        print("The correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")

    top_15 = [
        "BTC",
        "ETH",
        "ADA",
        "XRP",
        "DOGE",
        "DOT",
        "MATIC",
        "LTC",
        "AVAX",
        "BCH",
        "XLM",
        "XMR",
        "ATOM",
        "UNI",
        "ETC",
    ]

    plot_aggregated_data(dfs)
    plot_heatmap(scores)
    plot_summary(gs)
    plot_cascade(cascade_old_id)
    plot_comparison(cascade_old_id, scores)

    # Assuming each key in scores and features represents a different group, we need to add this information
    # to the DataFrames before merging
    for key in scores:
        scores[key]["group"] = key
        features[key]["group"] = key

    # Merge the scores and features based on telegram_chat_id and the group
    combined_data = pd.DataFrame()
    for key in scores:
        merged = pd.merge(
            scores[key], features[key], on=["telegram_chat_id", "group"], how="inner"
        )
        combined_data = pd.concat([combined_data, merged])

    # Add a column to indicate mastermind status; the first entry of each group is a mastermind
    combined_data["is_mastermind"] = False
    for group in combined_data["group"].unique():
        combined_data.loc[
            (combined_data["group"] == group)
            & (
                combined_data["telegram_chat_id"]
                == combined_data[combined_data["group"] == group][
                    "telegram_chat_id"
                ].iloc[0]
            ),
            "is_mastermind",
        ] = True

    # Perform two-way ANOVA using 'ols' from statsmodels
    # The model includes 'is_mastermind', 'group', and their interaction as factors
    # model = ols(
    #     "number_of_signals	 ~ C(is_mastermind) * C(group)", data=combined_data
    # ).fit()
    model = ols(
        "average_increase_percentage    ~ C(is_mastermind) * C(group)",
        data=combined_data,
    ).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
