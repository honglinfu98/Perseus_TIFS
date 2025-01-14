"""
This function is used to get the global minimum and maximum training times
"""

from os import path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from perseus.settings import PROJECT_ROOT


#  above to prepare a panda dataframe, make sure each row specifies the number of features, model, dataset, accuracy, precision, recall, f1
def get_dataframe(metrics: dict):
    """
    This function is used to get the dataframe for the metrics
    """
    df = pd.DataFrame(metrics).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["dataset", "model"])
    df = df.reset_index()
    df = df.rename(columns={"level_0": "dataset", "level_1": "model"})
    return df


def calculate_metrics_at_threshold(probs, labels, threshold=0.57):
    """
    Calculate accuracy, precision, recall, and F1 score for given probabilities at a specified threshold.

    Args:
    probs (np.array): The probabilities output by the model.
    labels (np.array): The actual labels.
    threshold (float): The threshold to determine positive class predictions.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Convert probabilities to binary predictions based on the threshold
    preds = (probs > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def get_metrics(results: dict):
    """
    This function is used to get the accuracy, precision, recall, and f1 for each model and dataset at a threshold of 0.55
    """
    metrics = {}
    for dataset in datasets:
        for model in models:
            probs = results[dataset][model]["metrics"]["probs"].flatten()
            labels = results[dataset][model]["metrics"]["labels"].flatten()
            metrics_result = calculate_metrics_at_threshold(
                probs, labels, threshold=0.54
            )
            metrics[(dataset, model)] = metrics_result
    return metrics


if __name__ == "__main__":

    datasets = ["DDINA", "COSS", "DDM"]
    dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
    models = ["GCN", "GAT", "GraphSAGE"]
    label = 1

    # open the results files
    with open(path.join(PROJECT_ROOT, "data", "results_l_wc.pkl"), "rb") as file:
        results_tr = pickle.load(file)

    # Assuming results_tr is loaded as shown in your previous example
    three_features = get_metrics(results_tr)
    three_features_df = get_dataframe(three_features)

    # combine the first three columns into one
    three_features_df["model_dataset"] = (
        three_features_df["model"] + " - " + three_features_df["dataset"]
    )

    # change the order of the columns
    three_features_df = three_features_df[
        ["model_dataset", "accuracy", "precision", "recall", "f1"]
    ]

    a = three_features_df.iloc[3]
    a["model_dataset"] = "Cossine similarity GCN (directed features)"
    a_df = a.to_frame().T

    # open the results files
    with open(path.join(PROJECT_ROOT, "data", "results_l.pkl"), "rb") as file:
        results_tr = pickle.load(file)

    # Assuming results_tr is loaded as shown in your previous example
    three_features = get_metrics(results_tr)
    three_features_df = get_dataframe(three_features)

    # combine the first three columns into one
    three_features_df["model_dataset"] = (
        three_features_df["model"] + " - " + three_features_df["dataset"]
    )

    # change the order of the columns
    three_features_df = three_features_df[
        ["model_dataset", "accuracy", "precision", "recall", "f1"]
    ]

    b = three_features_df.iloc[3]
    b["model_dataset"] = "Cossine similarity GCN (weighted features)"
    b_df = b.to_frame().T

    filtered_df = three_features_df[
        ~three_features_df["model_dataset"].str.contains("GCN|COSS")
    ]

    merged_df = pd.concat([filtered_df, a_df, b_df], ignore_index=True)

    merged_df["model_dataset"] = merged_df["model_dataset"].replace(
        {
            "GAT - DDINA": "Directed diffusion GAT",
            "GraphSAGE - DDINA": "Directed diffusion GraphSAGE",
            "GAT - DDM": "Weighted diffusion GAT",
            "GraphSAGE - DDM": "Weighted diffusion GraphSAGE",
        }
    )

    df_reordered = merged_df[["model_dataset", "precision", "f1", "accuracy", "recall"]]
    df_reordered = df_reordered.loc[[1, 3, 0, 2, 4, 5]].reset_index(drop=True)

    # merged_df.loc[merged_df['model_dataset'] == 'GCN - COSS', 'model_dataset'] = 'cossine similarity directed diffusion GCN'
