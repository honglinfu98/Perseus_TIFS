# from os import path
# import pickle
# import pandas as pd

# # import random
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import to_undirected
# from torch_geometric.data import Data
# from perseus.dataset.dataset_preparation import (
#     prepare_cos_data,
#     prepare_data,
#     prepare_ddm_data,
# )
# from perseus.dataset.preprocess.train_test_validate import (
#     get_btc_test_scored_signals,
#     get_btc_train_scored_signals,
#     get_btc_valid_scored_signals,
# )
# from perseus.dataset.preprocess.process import (
#     compute_weighted_graph_features,
#     get_graphs,
#     process_dataframe,
#     assign_event_ids,
#     graph_features,
#     combine_features,
# )
# from perseus.dataset.preprocess.groudtruth_labeling import (
#     read_labeling_csv_back_to_dict,
# )
# from perseus.settings import PROJECT_ROOT
# from perseus.dataset.dataset_preparation import split_data


# def chunked(iterable, n):
#     """
#     Yield successive chunks of size n from iterable.
#     The last chunk may be smaller if len(iterable) % n != 0.
#     """
#     seq = list(iterable)
#     for i in range(0, len(seq), n):
#         yield list(seq[i : i + n])


# def features_engineer_aggregated(df: pd.DataFrame, final_results: dict):
#     aggregated_market_features = {}
#     for k, v in final_results.items():
#         tokens = k.split()
#         filtered_df = df[df["commodity"].isin(tokens)]
#         grouped_data = (
#             filtered_df.groupby(["telegram_chat_id"])
#             .agg(
#                 {
#                     "increase_percentage": "mean",
#                     "btc_base_return": "mean",
#                     "speed": "mean",
#                     "id": "count",
#                     "chat_crowd_score": "last",
#                     "targets_achieved": "sum",
#                     "total_targets": "sum",
#                 }
#             )
#             .reset_index()
#             .rename(
#                 columns={
#                     "increase_percentage": "average_increase_percentage",
#                     "btc_base_return": "average_btc_base_return",
#                     "speed": "average_speed",
#                     "id": "number_of_signals",
#                     "chat_crowd_score": "latest_chat_crowd_score",
#                     "targets_achieved": "sum_targets_achieved",
#                     "total_targets": "sum_total_targets",
#                 }
#             )
#         )
#         grouped_data["rating"] = (
#             grouped_data["sum_targets_achieved"] / grouped_data["sum_total_targets"]
#         )
#         aggregated_market_features[k] = grouped_data

#     return aggregated_market_features


# def aggregate_data_aggregated(df: pd.DataFrame, labelings: dict, N: int = 10):
#     """
#     This function aggregates the data based on the commodity and the event_id
#     """
#     df_grouped = df.sort_values("start_date").groupby(["commodity", "event_id"])

#     intermediate_results = {}
#     # unique_chat_ids_per_commodity = {}
#     aggregating_chat_id = {}
#     aggregating_results = {}
#     new_labelings = {}
#     mappings = {}

#     for (commodity, _), group in df_grouped:
#         dates = pd.to_datetime(group["start_date"])
#         delta_minutes = (dates - dates.iloc[0]).dt.total_seconds() / 60

#         # Use drop_duplicates to keep only the first appearing telegram_chat_id
#         group = group.drop_duplicates(subset="telegram_chat_id", keep="first")
#         tuples = list(
#             zip(
#                 group["telegram_chat_id"],
#                 delta_minutes,
#                 group["increase_percentage"],
#                 group["btc_base_return"],
#                 group["position"],
#                 group["targets_achieved"],
#                 group["total_targets"],
#                 group["duration_min"],
#                 group["start_date"],
#                 group["end_date"],
#                 group["speed"],
#                 group["message_text"],
#             )
#         )

#         if len(tuples) >= 2:
#             if commodity not in intermediate_results:
#                 intermediate_results[commodity] = []

#             intermediate_results[commodity].append(tuples)

#     commodity_set = set(labelings.keys())
#     for c in chunked(commodity_set, N):
#         s = " ".join(c)
#         #  Access the intermediate results and get the key in the key in C to create the aggregating_results
#         aggregating_results[s] = {
#             commodity: intermediate_results[commodity]
#             for commodity in c
#             if commodity in intermediate_results
#         }
#         merged = {}
#         for commodity in c:
#             d = labelings[commodity]
#             for id, val in d.items():
#                 merged[id] = merged.get(id, 0) | val
#         new_labelings[s] = merged

#     for k, v in new_labelings.items():
#         aggregating_chat_id[k] = list(v.keys())

#     i_intermediate_results = {}
#     for k, v in aggregating_results.items():
#         aggregating_list = []
#         for c, t in v.items():
#             aggregating_list = aggregating_list + t
#         i_intermediate_results[k] = aggregating_list
#         # i_intermediate_results[k] = [item for sublist in v.values() for item in sublist]
#     final_results = {}

#     for commodity, lists_of_tuples in i_intermediate_results.items():
#         unique_ids = aggregating_chat_id[commodity]
#         id_mapping = {
#             old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids), start=0)
#         }

#         # Inverse mapping from new_id to old_id
#         new_id_to_old = {v: k for k, v in id_mapping.items()}

#         # Save the mappings for future use
#         if commodity not in mappings:
#             mappings[commodity] = {}
#         mappings[commodity]["id_to_new"] = id_mapping
#         mappings[commodity]["new_to_id"] = new_id_to_old  # Include the inverse mapping

#         mapped_list = []
#         for tuple_list in lists_of_tuples:
#             data_dict = {
#                 t[0]: t[1] for t in tuple_list
#             }  # Convert tuple_list to a dictionary first
#             data_dict["T"] = tuple_list[-1][1]  # Include the 'T' key
#             mapped_dict = {id_mapping.get(k, k): v for k, v in data_dict.items()}
#             mapped_list.append(mapped_dict)

#         final_results[commodity] = mapped_list

#     unique_counts = {
#         commodity: len(chat_ids) for commodity, chat_ids in aggregating_chat_id.items()
#     }

#     return final_results, unique_counts, mappings, intermediate_results, new_labelings


# def split_data_aggregated(options: str, loader: bool = True, N: int = 10):
#     """
#     Split the data into train, test, and validate sets for temporal tasks
#     """
#     # Initial data loading and processing
#     train_signals = get_btc_train_scored_signals()
#     validate_signals = get_btc_valid_scored_signals()
#     test_signals = get_btc_test_scored_signals()

#     labelings = [
#         read_labeling_csv_back_to_dict("train"),
#         read_labeling_csv_back_to_dict("valid"),
#         read_labeling_csv_back_to_dict("test"),
#     ]

#     datasets = [train_signals, validate_signals, test_signals]
#     gs_ls = []
#     features_ls = []
#     label_mapping_ls = []
#     cascade_ls = []
#     P_theta_ls = []

#     for dataset, labeling in zip(datasets, labelings):
#         processed_signals = process_dataframe(dataset)
#         ided_signals = assign_event_ids(processed_signals)
#         cascade, no_nodes, id_mapping, cascade_labeling, new_labeling = (
#             aggregate_data_aggregated(ided_signals, labeling, N)
#         )
#         gs, P_theta = get_graphs(cascade, no_nodes, id_mapping)
#         graph_feature = graph_features(gs)
#         market_feature = features_engineer_aggregated(processed_signals, cascade)
#         weighted_feature = compute_weighted_graph_features(P_theta)
#         combine_feature = combine_features(
#             market_feature, graph_feature, weighted_feature
#         )

#         gs_ls.append(gs)
#         features_ls.append(combine_feature)
#         cascade_ls.append(cascade)
#         P_theta_ls.append(P_theta)
#         label_mapping_ls.append(new_labeling)

#     for i in range(3):
#         common_keys = set(
#             label_mapping_ls[i].keys()
#         )  # Assuming label_mapping_ls[i] is a dict with relevant keys
#         gs_ls[i] = {key: gs_ls[i][key] for key in common_keys if key in gs_ls[i]}
#         features_ls[i] = {
#             key: features_ls[i][key] for key in common_keys if key in features_ls[i]
#         }
#         P_theta_ls[i] = {
#             key: P_theta_ls[i][key] for key in common_keys if key in P_theta_ls[i]
#         }

#     if options == "DDINA":
#         train_data = prepare_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
#         validate_data = prepare_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
#         test_data = prepare_data(gs_ls[2], features_ls[2], label_mapping_ls[2])

#     elif options == "COSS":
#         train_data = prepare_cos_data(gs_ls[0], features_ls[0], label_mapping_ls[0])
#         validate_data = prepare_cos_data(gs_ls[1], features_ls[1], label_mapping_ls[1])
#         test_data = prepare_cos_data(gs_ls[2], features_ls[2], label_mapping_ls[2])
#     elif options == "DDM":
#         train_data = prepare_ddm_data(
#             gs_ls[0], features_ls[0], label_mapping_ls[0], P_theta_ls[0]
#         )
#         validate_data = prepare_ddm_data(
#             gs_ls[1], features_ls[1], label_mapping_ls[1], P_theta_ls[1]
#         )
#         test_data = prepare_ddm_data(
#             gs_ls[2], features_ls[2], label_mapping_ls[2], P_theta_ls[2]
#         )

#     if loader:
#         train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#         validate_loader = DataLoader(validate_data, batch_size=1, shuffle=True)
#         test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

#         return train_loader, validate_loader, test_loader

#     else:

#         return train_data, validate_data, test_data


# def get_split_data_pickle_aa(options: str):
#     """
#     Load the data for temporal tasks using the pickle file
#     """
#     if options == "DDINA":
#         with open(
#             path.join(PROJECT_ROOT, "data", "buffer", "DDINA_data_aa31500.pkl"), "rb"
#         ) as file:
#             data = pickle.load(file)

#     elif options == "COSS":
#         with open(
#             path.join(PROJECT_ROOT, "data", "buffer", "COSS_data_aa31500.pkl"), "rb"
#         ) as file:
#             data = pickle.load(file)

#     elif options == "DDM":
#         with open(
#             path.join(PROJECT_ROOT, "data", "buffer", "DDM_data_aa31500.pkl"), "rb"
#         ) as file:
#             data = pickle.load(file)

#     train_loader = data[0]
#     validate_loader = data[1]
#     test_loader = data[2]

#     return train_loader, validate_loader, test_loader


# if __name__ == "__main__":
#     # a = get_split_data_pickle_t("DDM")

#     a = split_data("DDINA", loader=False)
#     # # save it in pickle
#     # with open(path.join(PROJECT_ROOT, "data", "DDINA_data_a.pkl"), "wb") as file:
#     #     pickle.dump(a, file)
#     b = split_data("COSS", loader=False)
#     # save it in pickle
#     # with open(path.join(PROJECT_ROOT, "data", "COSS_data_a.pkl"), "wb") as file:
#     #     pickle.dump(b, file)
#     c = split_data("DDM", loader=False)
#     # save it in pickle
#     # with open(path.join(PROJECT_ROOT, "data", "DDM_data_a.pkl"), "wb") as file:
#     #     pickle.dump(c, file)

#     # a25 = split_data_aggregated("DDINA", loader=False, N=25)
#     # b25 = split_data_aggregated("COSS", loader=False, N=25)
#     # c25 = split_data_aggregated("DDM", loader=False, N=25)

#     # a = get_split_data_pickle_wot("DDINA")
#     # b = get_split_data_pickle_wot("COSS")
#     # c = get_split_data_pickle_wot("DDM")
#     a5 = split_data_aggregated("DDINA", loader=False, N=3)
#     # with open(
#     #     path.join(PROJECT_ROOT, "data", "buffer", "DDINA_data_a5.pkl"), "wb"
#     # ) as file:
#     #     pickle.dump(a5, file)
#     b5 = split_data_aggregated("COSS", loader=False, N=3)
#     # with open(
#     #     path.join(PROJECT_ROOT, "data", "buffer", "DDINA_data_a5.pkl"), "wb"
#     # ) as file:
#     #     pickle.dump(b5, file)
#     c5 = split_data_aggregated("DDM", loader=False, N=3)
#     # with open(
#     #     path.join(PROJECT_ROOT, "data", "buffer", "DDINA_data_a5.pkl"), "wb"
#     # ) as file:
#     #     pickle.dump(c5, file)

#     a15 = split_data_aggregated("DDINA", loader=False, N=15)
#     b15 = split_data_aggregated("COSS", loader=False, N=15)
#     c15 = split_data_aggregated("DDM", loader=False, N=15)

#     a500 = split_data_aggregated("DDINA", loader=False, N=500)
#     b500 = split_data_aggregated("COSS", loader=False, N=500)
#     c500 = split_data_aggregated("DDM", loader=False, N=500)

#     aa = [
#         (DataLoader(i + j + k + n, batch_size=1, shuffle=True))
#         for i, j, k, n in zip(a, a5, a15, a500)
#     ]
#     with open(
#         path.join(PROJECT_ROOT, "data", "buffer", "DDINA_data_aa31500.pkl"), "wb"
#     ) as file:
#         pickle.dump(aa, file)

#     bb = [
#         (DataLoader(i + j + k + n, batch_size=1, shuffle=True))
#         for i, j, k, n in zip(b, b5, b15, b500)
#     ]
#     with open(
#         path.join(PROJECT_ROOT, "data", "buffer", "COSS_data_aa31500.pkl"), "wb"
#     ) as file:
#         pickle.dump(bb, file)

#     cc = [
#         (DataLoader(i + j + k + n, batch_size=1, shuffle=True))
#         for i, j, k, n in zip(c, c5, c15, c500)
#     ]
#     with open(
#         path.join(PROJECT_ROOT, "data", "buffer", "DDM_data_aa31500.pkl"), "wb"
#     ) as file:
#         pickle.dump(cc, file)


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MultiGraphSAGE(torch.nn.Module):
    def __init__(self, num_graphs, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.num_graphs = num_graphs
        self.sages = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        SAGEConv(in_channels, hidden_channels),
                        SAGEConv(hidden_channels, hidden_channels),
                    ]
                )
                for _ in range(num_graphs)
            ]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_graphs * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, graphs, x):
        embeddings = []
        for idx in range(self.num_graphs):
            edge_index = graphs[idx].edge_index
            x1 = F.relu(self.sages[idx][0](x, edge_index))
            x1 = self.sages[idx][1](x1, edge_index)
            x1 = F.normalize(x1, p=2, dim=-1)  # Normalization step
            embeddings.append(x1)

        fused_embeddings = torch.cat(embeddings, dim=-1)
        return self.classifier(fused_embeddings)


# Training loop:
model = MultiGraphSAGE(
    num_graphs=M, in_channels=d, hidden_channels=128, num_classes=num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(graphs, X)
    loss = F.cross_entropy(output[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
