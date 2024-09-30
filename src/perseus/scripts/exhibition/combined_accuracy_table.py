"""
This function is used to get the global minimum and maximum training times
"""

from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1


# open the results files
with open(path.join(PROJECT_ROOT, "data", "results_tr.pkl"), "rb") as file:
    results_tr = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_f.pkl"), "rb") as file:
    results_f = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_t.pkl"), "rb") as file:
    results_t = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_fv.pkl"), "rb") as file:
    results_fv = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_e.pkl"), "rb") as file:
    results_e = pickle.load(file)


# eritate over the results and get the accuracy, precision, recall, f1 for each model and dateset, and prepapre a panda dataframe for it
def get_metrics(results: dict):
    """
    This function is used to get the accuracy, precision, recall, and f1 for each model and dataset
    """
    metrics = {}
    for dataset in datasets:
        for model in models:
            accuracy = results[dataset][model]["metrics"]["accuracy"]
            precision = results[dataset][model]["metrics"]["precision"]
            recall = results[dataset][model]["metrics"]["recall"]
            f1 = results[dataset][model]["metrics"]["f1"]
            metrics[(dataset, model)] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return metrics


three_features = get_metrics(results_tr)
four_features = get_metrics(results_f)
two_features = get_metrics(results_t)
five_features = get_metrics(results_fv)
eight_features = get_metrics(results_e)


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


three_features_df = get_dataframe(three_features)
four_features_df = get_dataframe(four_features)
two_features_df = get_dataframe(two_features)
five_features_df = get_dataframe(five_features)
eight_features_df = get_dataframe(eight_features)


# add a column for the number of features
three_features_df["num_features"] = "3 features"
four_features_df["num_features"] = "4 features"
two_features_df["num_features"] = "2 features"
five_features_df["num_features"] = "5 features"
eight_features_df["num_features"] = "8 features"


# combine the first three columns into one
three_features_df["model_dataset"] = (
    three_features_df["model"]
    + " - "
    + three_features_df["dataset"]
    + " - "
    + three_features_df["num_features"]
)
four_features_df["model_dataset"] = (
    four_features_df["model"]
    + " - "
    + four_features_df["dataset"]
    + " - "
    + four_features_df["num_features"]
)
two_features_df["model_dataset"] = (
    two_features_df["model"]
    + " - "
    + two_features_df["dataset"]
    + " - "
    + two_features_df["num_features"]
)
five_features_df["model_dataset"] = (
    five_features_df["model"]
    + " - "
    + five_features_df["dataset"]
    + " - "
    + five_features_df["num_features"]
)
eight_features_df["model_dataset"] = (
    eight_features_df["model"]
    + " - "
    + eight_features_df["dataset"]
    + " - "
    + eight_features_df["num_features"]
)


# change the order of the columns
three_features_df = three_features_df[
    ["model_dataset", "accuracy", "precision", "recall", "f1"]
]
four_features_df = four_features_df[
    ["model_dataset", "accuracy", "precision", "recall", "f1"]
]
two_features_df = two_features_df[
    ["model_dataset", "accuracy", "precision", "recall", "f1"]
]
five_features_df = five_features_df[
    ["model_dataset", "accuracy", "precision", "recall", "f1"]
]
eight_features_df = eight_features_df[
    ["model_dataset", "accuracy", "precision", "recall", "f1"]
]


# combine the dataframes
combined_df = pd.concat(
    [
        three_features_df,
        four_features_df,
        two_features_df,
        five_features_df,
        eight_features_df,
    ]
)
combined_df
