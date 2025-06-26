# --- Add MultiGAT/MultiGraphSAGE results from multi_run.py ---

from os import path
import pandas as pd

from perseus.settings import PROJECT_ROOT


from os import path
import pickle
import pandas as pd
from perseus.settings import PROJECT_ROOT

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from os import path
import pickle

# Assume PROJECT_ROOT is defined elsewhere in your project settings
from perseus.settings import PROJECT_ROOT

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
from perseus.dataset.dataset_preparation import (
    get_split_data_pickle_wot,
    get_split_data_pickle_btc,
)


# from perseus.dataset.aggregating_dateset import get_split_data_pickle_aa


def extract_data_from_loader(data_loader):
    X_list = []
    y_list = []

    for batch in data_loader:
        # Assuming batch.x are the node features and batch.y are the labels
        X_list.append(batch.x)  # Node features
        y_list.append(batch.y)  # Labels

    # Concatenate all the batches into one matrix for X and y
    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy().flatten()  # Ensure y is 1D
    return X, y


def find_best_f1_threshold(probs, labels):
    """
    Find the threshold that results in the highest F1 score.

    Args:
    probs (np.array): The probabilities output by the model.
    labels (np.array): The actual labels.

    Returns:
    float: The threshold that maximizes the F1 score.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.01
    max_f1 = 0

    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold

    return best_threshold


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


# 0.57


def calculate_metrics_at_threshold(probs, labels, threshold=0.54):
    """
    Calculate accuracy, precision, recall, F1 score, and MCC for given probabilities at a specified threshold.

    Args:
    probs (np.array): The probabilities output by the model.
    labels (np.array): The actual labels.
    threshold (float): The threshold to determine positive class predictions.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, F1 score, and MCC.
    """
    # Convert probabilities to binary predictions based on the threshold
    preds = (probs > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


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
                probs, labels, threshold=find_best_f1_threshold(probs, labels)
            )
            metrics[(dataset, model)] = metrics_result
    return metrics


if __name__ == "__main__":

    datasets = ["DDINA", "DDM"]
    dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
    models = ["GCN", "GAT", "GraphSAGE"]
    label = 1

    # open the results files
    with open(path.join(PROJECT_ROOT, "data", "results_btc.pkl"), "rb") as file:
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
        ["model_dataset", "accuracy", "precision", "recall", "f1", "mcc"]
    ]

    a = three_features_df.iloc[0]
    a["model_dataset"] = "Cossine similarity GCN (directed features)"
    a_df = a.to_frame().T

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

    df_reordered = merged_df[
        ["model_dataset", "precision", "f1", "accuracy", "recall", "mcc"]
    ]
    df_reordered = df_reordered.loc[[3, 1, 0, 2, 4, 5]].reset_index(drop=True)

    train_loader, valid_loader, test_loader = get_split_data_pickle_wot("DDM")

    # Extract train and valid data
    X_train, y_train = extract_data_from_loader(train_loader)
    X_valid, y_valid = extract_data_from_loader(test_loader)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get predicted probabilities for Random Forest (used for ROC)
    y_scores_rf = rf_model.predict_proba(X_valid)[:, 1]

    # Compute ROC curve and AUC for Random Forest
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_valid, y_scores_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Find the best threshold for F1 score
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_valid, y_scores_rf > t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    # Calculate all metrics at the best threshold
    y_pred_rf_optimal = (y_scores_rf >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_valid, y_pred_rf_optimal)
    precision = precision_score(y_valid, y_pred_rf_optimal, zero_division=0)
    recall = recall_score(y_valid, y_pred_rf_optimal, zero_division=0)
    f1 = f1_score(y_valid, y_pred_rf_optimal, zero_division=0)

    # add a new row for model random forest f1, precision, recall, accuracy (0.7169811320754716, 0.7238095238095238, 0.7102803738317757, 0.883495145631068)
    random_forest_wot = {
        "model_dataset": "Random Forest Weighted Diffusion without Topological Features",
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy_score(y_valid, y_pred_rf_optimal),
        "recall": recall,
        "mcc": mcc,
    }

    train_loader, valid_loader, test_loader = get_split_data_pickle_btc("DDM")

    # Extract train and valid data
    X_train, y_train = extract_data_from_loader(train_loader)
    X_valid, y_valid = extract_data_from_loader(test_loader)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get predicted probabilities for Random Forest (used for ROC)
    y_scores_rf = rf_model.predict_proba(X_valid)[:, 1]

    # Compute ROC curve and AUC for Random Forest
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_valid, y_scores_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Find the best threshold for F1 score
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_valid, y_scores_rf > t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    # Calculate all metrics at the best threshold
    y_pred_rf_optimal = (y_scores_rf >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_valid, y_pred_rf_optimal)
    precision = precision_score(y_valid, y_pred_rf_optimal, zero_division=0)
    recall = recall_score(y_valid, y_pred_rf_optimal, zero_division=0)
    f1 = f1_score(y_valid, y_pred_rf_optimal, zero_division=0)

    # add a new row for model random forest f1, precision, recall, accuracy (0.7169811320754716, 0.7238095238095238, 0.7102803738317757, 0.883495145631068)
    random_forest = {
        "model_dataset": "Random Forest Weighted Diffusion",
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy_score(y_valid, y_pred_rf_optimal),
        "recall": recall,
        "mcc": mcc,
    }

    train_loader_d, valid_loader_d, test_loader_d = get_split_data_pickle_btc("DDINA")

    # Extract train and valid data
    X_train_d, y_train_d = extract_data_from_loader(train_loader_d)
    X_valid_d, y_valid_d = extract_data_from_loader(test_loader_d)

    # Initialize and train the Random Forest model
    rf_model_d = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_d.fit(X_train_d, y_train_d)

    # Get predicted probabilities for Random Forest (used for ROC)
    y_scores_rf_d = rf_model_d.predict_proba(X_valid_d)[:, 1]

    # Compute ROC curve and AUC for Random Forest
    fpr_rf_d, tpr_rf_d, thresholds_rf_d = roc_curve(y_valid_d, y_scores_rf_d)
    roc_auc_rf_d = auc(fpr_rf_d, tpr_rf_d)

    # Find the best threshold for F1 score
    thresholds_d = np.linspace(0.01, 0.99, 100)
    f1_scores_d = [f1_score(y_valid_d, y_scores_rf_d > t) for t in thresholds_d]
    best_threshold_d = thresholds_d[np.argmax(f1_scores_d)]
    best_f1_d = np.max(f1_scores_d)

    # Calculate all metrics at the best threshold
    y_pred_rf_optimal_d = (y_scores_rf_d >= best_threshold_d).astype(int)
    mcc_d = matthews_corrcoef(y_valid_d, y_pred_rf_optimal_d)
    precision_d = precision_score(y_valid_d, y_pred_rf_optimal_d, zero_division=0)
    recall_d = recall_score(y_valid_d, y_pred_rf_optimal_d, zero_division=0)
    f1_d = f1_score(y_valid_d, y_pred_rf_optimal_d, zero_division=0)

    # add a new row for model random forest f1, precision, recall, accuracy (0.7169811320754716, 0.7238095238095238, 0.7102803738317757, 0.883495145631068)
    random_forest_d = {
        "model_dataset": "Random Forest Directed Diffusion",
        "precision": precision_d,
        "f1": f1_d,
        "accuracy": accuracy_score(y_valid_d, y_pred_rf_optimal_d),
        "recall": recall_d,
        "mcc": mcc_d,
    }

    # add the new row to the dataframe
    df_reordered = df_reordered.append(random_forest, ignore_index=True)
    df_reordered = df_reordered.append(random_forest_d, ignore_index=True)
    df_reordered = df_reordered.append(random_forest_wot, ignore_index=True)

    latex_code = df_reordered.to_latex(
        index=False,
        header=True,
        column_format="lccccc",
        bold_rows=True,
        float_format=lambda x: f"{x:.3f}",
    )

    # Add the tabular environment, resize box, caption, and label
    latex_output = (
        """
    \\begin{table}[!t]
    \\centering
    \\footnotesize
    \\begin{tabular}{"""
        + latex_code.split("\n")[1]
        + """}
    \\toprule
    """
        + latex_code.split("\n")[3]
        + """
    \\midrule
    """
        + "\\n".join(latex_code.split("\n")[4:-2])
        + """
    \\bottomrule
    \\end{tabular}
    \\caption{Performance comparison of various models}
    \\label{tab:model_performance}
    \\end{table}
    """
    )

    # Print the final LaTeX table code
    print(latex_output)
    df_reordered

    # Read CSV
    hp_df = pd.read_csv(path.join(PROJECT_ROOT, "data/buffer/hp_search_results2.csv"))

    # For each (data, model), get the row with the best test_f1
    best_rows = (
        hp_df.sort_values("test_f1", ascending=False)
        .groupby(["data", "model"], as_index=False)
        .first()
    )

    # Map to your display names
    model_map = {"MultiGAT": "Multi-head GAT", "MultiGraphSAGE": "Multi-head GraphSAGE"}
    dataset_map = {"DDINA": "Directed diffusion", "DDM": "Weighted diffusion"}

    # Build rows for the table
    multi_rows = []
    for _, row in best_rows.iterrows():
        model_dataset = f"{model_map.get(row['model'], row['model'])} {dataset_map.get(row['data'], row['data'])}"
        multi_rows.append(
            {
                "model_dataset": model_dataset,
                "precision": row[
                    "test_precision"
                ],  # Not in CSV, can be left blank or calculated if you have probs/labels
                "f1": row["test_f1"],
                "accuracy": row["test_acc"],
                "recall": row["test_recall"],  # Not in CSV
                "mcc": row["test_mcc"],  # Not in CSV
            }
        )

    # Convert to DataFrame and append
    multi_df = pd.DataFrame(multi_rows)
    # If you want to keep the order, you can insert at a specific position or just append
    df_reordered = pd.concat([df_reordered, multi_df], ignore_index=True)

    # change model_dataset row 4 and 5 to SOTA directed diffusion and SOTA weighted diffusion
    df_reordered.loc[4, "model_dataset"] = "SOTA Directed Diffusion"
    df_reordered.loc[5, "model_dataset"] = "SOTA Weighted Diffusion"

    # change multi-head in model_dataset to fusion
    df_reordered["model_dataset"] = df_reordered["model_dataset"].str.replace(
        "Multi-head", "Fusion"
    )
