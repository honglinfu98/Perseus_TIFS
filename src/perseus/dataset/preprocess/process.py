"""
This script is used to process the scored signals and create summary for the data
"""

from datetime import timedelta
import json
from collections import defaultdict
import pandas as pd
import networkx as nx
from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals
from perseus.dataset.preprocess.DANI import DANI


def calculate_effsize_efficiency(G, ego):
    """
    This function is used to calculate the effective size and efficiency of a node in a graph
    """
    # Get the ego network
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network (excluding ego)
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)
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


def graph_features(gs: dict):
    """
    This function is used to extract graph features from the graph
    """
    dfs = {}

    # Looping through each key in the scores dictionary
    for key in gs.keys():
        # Creating lists to store node_ids, in-ego ratios, and out-ego ratios
        node_ids = []
        in_ratios = []
        out_ratios = []
        out_nodes = []
        eff_sizes = []
        efficiencies = []
        density = []
        clustering_coeffs = []

        for i in gs[key].nodes():
            node_id = i
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

    return dfs


def combine_features(market_features: dict, graph_features: dict):
    """
    This function is used to combine the market and graph features
    """
    combined_data = {}

    for key in market_features.keys():
        # Check if the key exists in dfs and 'telegram_chat_id' exists in both DataFrames
        if (
            key in graph_features
            and "telegram_chat_id" in market_features[key].columns
            and "telegram_chat_id" in graph_features[key].columns
        ):
            # Perform an inner join on 'telegram_chat_id'
            combined_df = pd.merge(
                market_features[key],
                graph_features[key],
                on="telegram_chat_id",
                how="inner",
            )
        else:
            # If key is not in dfs or 'telegram_chat_id' is missing in either DataFrame, use an empty DataFrame
            combined_df = pd.DataFrame()

        combined_data[key] = combined_df

    return combined_data


def relabel_edges(d: dict, mapping: dict):
    """
    Function to relabel edges based on new_to_id mapping
    """
    new_d = defaultdict(float)
    for (src, dst), weight in d.items():
        # Relabel src and dst using the mapping
        new_src = mapping.get(src, src)  # Use original if not in mapping
        new_dst = mapping.get(dst, dst)  # Use original if not in mapping
        new_d[(new_src, new_dst)] = weight
    return new_d


def assign_event_ids(df: pd.DataFrame):
    """
    Assigns event IDs to the pump signals based on the session timeout
    """

    # Initialize the event ID
    event_id = 0

    # Group the data by commodity
    groups = df.groupby(["commodity", "position"])

    # For each group...
    for _, group in groups:
        # Sort the group by source_datetime
        group = group.sort_values("start_date")
        session_timeout = group["time_diff"].quantile(0.95)
        if session_timeout > timedelta(hours=72):
            session_timeout = timedelta(hours=72)

        # Track the previous timestamp for the first row as the first session start time
        prev_time = group.iloc[0]["start_date"]

        # Assign the first event ID to the first row
        df.loc[group.index[0], "event_id"] = event_id

        # For the remaining rows in the group...
        for i, row in group.iloc[1:].iterrows():
            # If the difference between the current row's source_datetime and the previous timestamp exceeds the session_timeout...
            if row["start_date"] - prev_time > session_timeout:
                # Increment the event ID
                event_id += 1

            # Assign the current event ID to the row
            df.loc[i, "event_id"] = event_id

            # Update the previous timestamp
            prev_time = row["start_date"]

        # Increase the event ID for the next commodity (to ensure the next commodity starts with a new event ID)
        event_id += 1

    return df


def process_dataframe(signals_df: pd.DataFrame):
    """
    Process the dataframe to extract and generate the features
    """

    # Parsing 'price_increase' and extracting the values
    df = signals_df[signals_df["price_increase"] != "TRADE DATA NOT AVAILABLE"]
    df["parsed_price_increase"] = df["price_increase"].apply(json.loads)
    df["increase_percentage"] = df["parsed_price_increase"].apply(
        lambda x: x.get("price_increase", None)
    )
    df["start_price"] = df["parsed_price_increase"].apply(lambda x: x.get("from", None))
    df["end_price"] = df["parsed_price_increase"].apply(lambda x: x.get("to", None))
    # create a new column with value "long" if start_price is greater than end_price otherwise "short"
    df["position"] = df.apply(
        lambda x: "long" if x["start_price"] < x["end_price"] else "short", axis=1
    )
    df["parsed_rate"] = df["targets_achieved_rate"].apply(json.loads)
    df["targets_achieved"] = df["parsed_rate"].apply(
        lambda x: x.get("targets_achieved", None)
    )
    df["total_targets"] = df["parsed_rate"].apply(
        lambda x: x.get("total_targets", None)
    )
    df["parsed_duration"] = df["duration"].apply(json.loads)
    df["duration_min"] = df["parsed_duration"].apply(
        lambda x: x.get("duration_min", None)
    )
    df["start_date"] = df["parsed_duration"].apply(lambda x: x.get("start_date", None))
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = df["parsed_duration"].apply(lambda x: x.get("end_date", None))
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["speed"] = df["increase_percentage"] / df["duration_min"]
    df = df.sort_values("start_date", ascending=True)

    # Calculate the time difference between consecutive rows for the same entity_id and signal type
    df["time_diff"] = df.groupby(["position", "commodity"])["start_date"].diff()

    return df


def aggregate_data(df: pd.DataFrame):
    """
    This function aggregates the data based on the commodity and the event_id
    """
    df_grouped = df.sort_values("start_date").groupby(["commodity", "event_id"])

    intermediate_results = {}
    unique_chat_ids_per_commodity = {}
    mappings = {}

    for (commodity, _), group in df_grouped:
        dates = pd.to_datetime(group["start_date"])
        delta_minutes = (dates - dates.iloc[0]).dt.total_seconds() / 60

        # Use drop_duplicates to keep only the first appearing telegram_chat_id
        group = group.drop_duplicates(subset="telegram_chat_id", keep="first")
        tuples = list(
            zip(
                group["telegram_chat_id"],
                delta_minutes,
                group["increase_percentage"],
                group["position"],
                group["targets_achieved"],
                group["total_targets"],
                group["duration_min"],
                group["start_date"],
                group["end_date"],
                group["speed"],
                group["message_text"],
            )
        )

        if len(tuples) >= 2:
            if commodity not in intermediate_results:
                intermediate_results[commodity] = []

            intermediate_results[commodity].append(tuples)

            # Track unique telegram_chat_id per commodity
            if commodity not in unique_chat_ids_per_commodity:
                unique_chat_ids_per_commodity[commodity] = set()

            unique_chat_ids_per_commodity[commodity].update(
                group["telegram_chat_id"].tolist()
            )

    final_results = {}

    for commodity, lists_of_tuples in intermediate_results.items():
        unique_ids = unique_chat_ids_per_commodity[commodity]
        id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids), start=0)
        }

        # Inverse mapping from new_id to old_id
        new_id_to_old = {v: k for k, v in id_mapping.items()}

        # Save the mappings for future use
        if commodity not in mappings:
            mappings[commodity] = {}
        mappings[commodity]["id_to_new"] = id_mapping
        mappings[commodity]["new_to_id"] = new_id_to_old  # Include the inverse mapping

        mapped_list = []
        for tuple_list in lists_of_tuples:
            data_dict = {
                t[0]: t[1] for t in tuple_list
            }  # Convert tuple_list to a dictionary first
            data_dict["T"] = tuple_list[-1][1]  # Include the 'T' key
            mapped_dict = {id_mapping.get(k, k): v for k, v in data_dict.items()}
            mapped_list.append(mapped_dict)

        final_results[commodity] = mapped_list

    unique_counts = {
        commodity: len(chat_ids)
        for commodity, chat_ids in unique_chat_ids_per_commodity.items()
    }

    return final_results, unique_counts, mappings, intermediate_results


def features_engineer(df: pd.DataFrame):
    """
    This function is used to engineer market features for the signals
    """
    grouped_data = (
        df.groupby(["commodity", "telegram_chat_id"])
        .agg(
            {
                "increase_percentage": "mean",
                "speed": "mean",
                "id": "count",
                "chat_crowd_score": "last",
                "targets_achieved": "sum",
                "total_targets": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                "increase_percentage": "average_increase_percentage",
                "speed": "average_speed",
                "id": "number_of_signals",
                "chat_crowd_score": "latest_chat_crowd_score",
                "targets_achieved": "sum_targets_achieved",
                "total_targets": "sum_total_targets",
            }
        )
    )

    # Organize the results into a dictionary
    result_dict = {}
    for commodity in grouped_data["commodity"].unique():
        result_dict[commodity] = grouped_data[grouped_data["commodity"] == commodity][
            [
                "telegram_chat_id",
                "average_increase_percentage",
                "average_speed",
                "number_of_signals",
                "latest_chat_crowd_score",
                "sum_targets_achieved",
                "sum_total_targets",
            ]
        ]

    for commodity, df in result_dict.items():
        result_dict[commodity]["rating"] = (
            result_dict[commodity]["sum_targets_achieved"]
            / result_dict[commodity]["sum_total_targets"]
        )

    return result_dict


def get_graphs(cascade: dict, no_nodes: dict, id_mapping: dict):
    """
    This function creates the graphs for the cascadeX
    :param cascade: the cascade
    :param no_nodes: the number of nodes for each commodity
    :param id_mapping: the mapping between the commodity and the id
    :return: the graphs, the weight ordered edges, the edge list, and relabeled edge list
    """
    ensure_graph_learned = {}
    # Filter the no_nodes that have more than 3 nodes and have more than no nodes cascade
    for key, value in no_nodes.items():
        if value > 3:  #  and value < len(cascade[key])
            ensure_graph_learned[key] = value

    # Create the graphs for each commodity using moer_than_three
    graphs = {}
    result = {}
    A = {}
    P_dict = {}
    for key, value in ensure_graph_learned.items():
        graphs[key], result[key], A[key], P_dict[key] = DANI(
            ensure_graph_learned[key], cascade[key]
        )
        graphs[key] = nx.relabel_nodes(graphs[key], id_mapping[key]["new_to_id"])

    # Loop over each key in P_dict and apply the mapping
    for key in P_dict:
        if key in id_mapping:
            # Extract new_to_id mapping for the current key
            new_to_id = id_mapping[key]["new_to_id"]
            # Relabel edges in the current dictionary using the new_to_id mapping
            P_dict[key] = relabel_edges(P_dict[key], new_to_id)

    return graphs, result, A, P_dict


if __name__ == "__main__":
    signals = get_test_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = (
        aggregate_data(ided_signals)
    )
    gs, results, As, P_dict = get_graphs(
        cascade_buffer, no_nodes_buffer, id_mapping_buffer
    )
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    combine_feature = combine_features(market_feature, graph_feature)
