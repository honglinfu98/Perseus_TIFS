"""
This script is used to process the scored signals and create summary for the data
"""
from datetime import timedelta
import json
import pandas as pd
import networkx as nx
from clotho.dataset.extract.cloudburst_connection import get_scored_signals
from clotho.dataset.preprocess.DANI import DANI
from collections import defaultdict

# Function to relabel edges based on new_to_id mapping
def relabel_edges(d, mapping):
    new_d = defaultdict(float)
    for (src, dst), weight in d.items():
        # Relabel src and dst using the mapping
        new_src = mapping.get(src, src)  # Use original if not in mapping
        new_dst = mapping.get(dst, dst)  # Use original if not in mapping
        new_d[(new_src, new_dst)] = weight
    return new_d


def assign_event_ids(df: pd.DataFrame):
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


def process_dataframe(df: pd.DataFrame):
    # Parsing 'price_increase' and extracting the values
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

    # Parsing 'duration' and extracting the values
    df["parsed_duration"] = df["duration"].apply(json.loads)
    df["duration_min"] = df["parsed_duration"].apply(
        lambda x: x.get("duration_min", None)
    )
    # df["tsv"] = df["ts"] / df["d"]
    df["start_date"] = df["parsed_duration"].apply(lambda x: x.get("start_date", None))
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = df["parsed_duration"].apply(lambda x: x.get("end_date", None))
    df["end_date"] = pd.to_datetime(df["end_date"])
    # Drop temporary parsed columns
    # df = df.drop(columns=['parsed_price_increase', 'parsed_duration'])

    df["speed"] = df["increase_percentage"] / df["duration_min"]

    df = df.sort_values("start_date", ascending=True)

    # Calculate the time difference between consecutive rows for the same entity_id and signal type
    df["time_diff"] = df.groupby(["position", "commodity"])["start_date"].diff()

    return df


def aggregate_data(df: pd.DataFrame):
    df_grouped = df.sort_values("start_date").groupby(["commodity", "event_id"])

    intermediate_results = {}
    unique_chat_ids_per_commodity = {}
    mappings = {}

    for (commodity, event_id), group in df_grouped:
        dates = pd.to_datetime(group["start_date"])
        delta_minutes = (dates - dates.iloc[0]).dt.total_seconds() / 60

        # Use drop_duplicates to keep only the first appearing telegram_chat_id
        group = group.drop_duplicates(subset="telegram_chat_id", keep="first")
        tuples = list(zip(group["telegram_chat_id"], delta_minutes))

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

    return final_results, unique_counts, mappings


def features_engineer(df: pd.DataFrame):
    # Group by 'commodity' and 'telegram_chat_id' and compute the desired metrics
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
    :return: the graphs
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
            new_to_id = id_mapping[key]['new_to_id']
            # Relabel edges in the current dictionary using the new_to_id mapping
            P_dict[key] = relabel_edges(P_dict[key], new_to_id)


    return graphs, result, A, P_dict


if __name__ == "__main__":
    signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
