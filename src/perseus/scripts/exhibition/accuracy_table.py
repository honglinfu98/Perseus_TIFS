from os import path
import pickle
import pandas as pd
from perseus.settings import PROJECT_ROOT


datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1


with open(path.join(PROJECT_ROOT, "data", "results_com.pkl"), "rb") as file:
    results = pickle.load(file)


with open(path.join(PROJECT_ROOT, "data", "results1_com.pkl"), "rb") as file:
    results1 = pickle.load(file)


data = []
metrics_columns = sorted(results1[datasets[0]][models[0]]["metrics"].keys())
for dataset in datasets:
    for model_name in models:
        # Check if the dataset and model_name keys exist in the results dictionary
        if dataset in results1 and model_name in results1[dataset]:
            # Retrieve the stored metrics for the current dataset and model
            metrics = results1[dataset][model_name]["metrics"]
            # Append a new record including the dataset, model, and all metrics
            data.append(
                [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
            )
        else:
            # Handle the missing dataset/model combination by appending NaNs or placeholders
            data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

columns = ["Dataset", "Model"] + metrics_columns
df = pd.DataFrame(data, columns=columns)
filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]
filtered_df["Dataset"] = filtered_df["Dataset"].replace(
    {
        "DDINA": "Directed Diffusion",
        "COSS": "Cossine Similarity",  # Note: There's a typo here, it should be "cosine similarity"
        "DDM": "Weighted Diffusion",
    }
)
filtered_df["Method"] = filtered_df["Dataset"] + " - " + filtered_df["Model"]
filtered_df = filtered_df.drop(columns=["Dataset", "Model"])

filtered_df = filtered_df[["Method", "accuracy", "f1", "precision", "recall"]]

highlighted_df1 = filtered_df.style.highlight_max(
    subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
)
highlighted_df1


data = []
metrics_columns = sorted(results[datasets[0]][models[0]]["metrics"].keys())


# Loop through each dataset and model to gather metrics
for dataset in datasets:
    for model_name in models:
        # Check if the dataset and model_name keys exist in the results dictionary
        if dataset in results and model_name in results[dataset]:
            # Retrieve the stored metrics for the current dataset and model
            metrics = results[dataset][model_name]["metrics"]
            # Append a new record including the dataset, model, and all metrics
            data.append(
                [dataset, model_name] + [metrics[metric] for metric in sorted(metrics)]
            )
        else:
            # Handle the missing dataset/model combination by appending NaNs or placeholders
            data.append([dataset, model_name] + [None for _ in sorted(metrics_columns)])

# Assuming 'metrics_columns' is predefined or you define it based on your known metrics
columns = ["Dataset", "Model"] + metrics_columns
df = pd.DataFrame(data, columns=columns)
filtered_df = df[["Dataset", "Model", "accuracy", "f1", "precision", "recall"]]
filtered_df["Dataset"] = filtered_df["Dataset"].replace(
    {
        "DDINA": "Directed Diffusion",
        "COSS": "Cossine Similarity",  # Note: There's a typo here, it should be "cosine similarity"
        "DDM": "Weighted Diffusion",
    }
)
filtered_df["Method"] = filtered_df["Dataset"] + " - " + filtered_df["Model"]
filtered_df = filtered_df.drop(columns=["Dataset", "Model"])

filtered_df = filtered_df[["Method", "accuracy", "f1", "precision", "recall"]]

highlighted_df = filtered_df.style.highlight_max(
    subset=["accuracy", "f1", "precision", "recall"], color="yellow", axis=0
)
highlighted_df
