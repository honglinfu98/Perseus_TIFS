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


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, f1_score

from perseus.settings import PROJECT_ROOT

datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]


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


# # Define thresholds
# thresholds = np.linspace(0, 1, 100)

# fig, ax = plt.subplots(figsize=(10, 6))

# # Color and linestyle management for clarity in the plot
# colors = ["r", "g", "b"]
# linestyles = ["-", "--", ":"]

# # Enumerate over all datasets and models
# for i, dataset in enumerate(datasets):
#     for j, model in enumerate(models):
#         # Calculate precision and F1 scores for each dataset-model combination
#         labels = results_t[dataset][model]["metrics"]["labels"]
#         probs = results_t[dataset][model]["metrics"]["probs"]
#         precision_scores = [
#             precision_score(labels, probs > threshold, zero_division=0)
#             for threshold in thresholds
#         ]
#         f1_scores = [
#             f1_score(labels, probs > threshold, zero_division=0)
#             for threshold in thresholds
#         ]

#         # Plot each model as a different line in the plot
#         ax.plot(
#             thresholds,
#             precision_scores,
#             linestyle=linestyles[j],
#             color=colors[i],
#             label=f"{dataset} {model} Precision",
#         )
#         ax.plot(
#             thresholds,
#             f1_scores,
#             linestyle=linestyles[j],
#             color=colors[i],
#             alpha=0.5,
#             label=f"{dataset} {model} F1 Score",
#         )

# ax.set_title("Performance Across Datasets and Models")
# ax.set_xlabel("Threshold")
# ax.set_ylabel("Measure")
# ax.legend(loc="best", fontsize="small")
# plt.show()


# Define thresholds
thresholds = np.linspace(0, 1, 100)

# Color and linestyle management for clarity in the plot
colors = ["r", "g", "b"]  # Different colors for different models
linestyles = ["-", "--", ":"]  # Different linestyles for different datasets

# Set up the plots with shared x-axis
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Assuming 'datasets' and 'models' are defined and 'results_t' contains the necessary data
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        labels = results_t[dataset][model]["metrics"]["labels"]
        probs = results_t[dataset][model]["metrics"]["probs"]

        # Calculate Precision and F1 scores for each threshold
        precision_scores = [
            precision_score(labels, probs > threshold, zero_division=0)
            for threshold in thresholds
        ]
        f1_scores = [
            f1_score(labels, probs > threshold, zero_division=0)
            for threshold in thresholds
        ]

        # Plot Precision
        axs[0].plot(
            thresholds,
            precision_scores,
            color=colors[i],
            linestyle=linestyles[j],
            label=f"{model} {dataset} Precision",
        )

        # Plot F1 Score
        axs[1].plot(
            thresholds,
            f1_scores,
            color=colors[i],
            linestyle=linestyles[j],
            label=f"{model} {dataset} F1 Score",
        )

# Precision subplot settings
axs[0].set_title("Precision Across Models and Datasets")
axs[0].set_ylabel("Precision")
axs[0].legend(loc="best", fontsize="small")
axs[0].grid(True)

# F1 Score subplot settings
axs[1].set_title("F1 Score Across Models and Datasets")
axs[1].set_xlabel("Threshold")
axs[1].set_ylabel("F1 Score")
axs[1].legend(loc="best", fontsize="small")
axs[1].grid(True)

plt.show()
