from os import path
import pickle

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from perseus.settings import PROJECT_ROOT


datasets = ["DDINA", "DDM"]
models = ["GraphSAGE"]
data = []

# open the files and load the results
with open(path.join(PROJECT_ROOT, "data", "results_f.pkl"), "rb") as file:
    results_f = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_fv.pkl"), "rb") as file:
    results_fv = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_tr.pkl"), "rb") as file:
    results_tr = pickle.load(file)

with open(path.join(PROJECT_ROOT, "data", "results_t.pkl"), "rb") as file:
    results_t = pickle.load(file)


def compute_metrics_at_threshold(results_t, models, datasets, threshold=0.55):
    metrics_at_threshold = {}

    for model in models:
        metrics_at_threshold[model] = {}
        for dataset in datasets:
            labels = results_t[dataset][model]["metrics"]["labels"]
            probs = results_t[dataset][model]["metrics"]["probs"]

            # Compute the metrics at the specified threshold
            predicted_labels = probs > threshold
            precision = precision_score(labels, predicted_labels, zero_division=0)
            recall = recall_score(labels, predicted_labels, zero_division=0)
            f1 = f1_score(labels, predicted_labels, zero_division=0)
            accuracy = accuracy_score(labels, predicted_labels)

            # Store the computed metrics
            metrics_at_threshold[model][dataset] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
            }

    return metrics_at_threshold


metrics_at_threshold_t = compute_metrics_at_threshold(results_t, models, datasets)
metrics_at_threshold_tr = compute_metrics_at_threshold(results_tr, models, datasets)
metrics_at_threshold_f = compute_metrics_at_threshold(results_f, models, datasets)
metrics_at_threshold_fv = compute_metrics_at_threshold(results_fv, models, datasets)


# Example usage:
for res in [
    metrics_at_threshold_t,
    metrics_at_threshold_tr,
    metrics_at_threshold_f,
    metrics_at_threshold_fv,
]:
    for model in models:
        for dataset in datasets:
            data.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "Precision": res[model][dataset]["precision"],
                    "Recall": res[model][dataset]["recall"],
                    "F1 Score": res[model][dataset]["f1_score"],
                    "Accuracy": res[model][dataset]["accuracy"],
                }
            )
df = pd.DataFrame(data)
df


# Define the number of features for each row
number_of_features = [
    "2-feature",
    "2-feature",
    "3-feature",
    "3-feature",
    "4-feature",
    "4-feature",
    "5-feature",
    "5-feature",
]

# Add the number of features as a new column to the DataFrame
df["Number of Features"] = number_of_features

df


# Combine 'Model', 'Dataset', and 'Number of Features' into one column named 'model specification'
df["model specification"] = (
    df["Model"] + " - " + df["Dataset"] + " - " + df["Number of Features"]
)

# Drop the original columns as they are now combined into 'model specification'
df.drop(columns=["Model", "Dataset", "Number of Features"], inplace=True)

# Display the updated DataFrame
df


# Creating a function to produce a LaTeX table from the DataFrame


def df_to_latex(df):
    # Defining the LaTeX table structure
    latex_str = r"""\begin{table}[!t]
\centering
\footnotesize
\begin{tabular}{m{1.4cm} m{1.75cm} m{0.7cm} m{0.6cm} m{0.7cm} m{0.6cm}}
\toprule
\textbf{Model Specification} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} & \textbf{Accuracy} \\
\midrule
"""
    # Iterate over the DataFrame rows to populate the table
    for idx, row in df.iterrows():
        latex_str += f"{row['model specification']} & {row['Precision']:.2%} & {row['Recall']:.2%} & {row['F1 Score']:.2%} & {row['Accuracy']:.2%} \\\\\n"

    # Closing table format
    latex_str += r"""\bottomrule
\end{tabular}
\caption{Detection performance comparison of GraphSAGE models for different datasets and feature settings.}
\label{tab:model_performance}
\end{table}"""

    return latex_str


# Generating LaTeX code from the DataFrame
latex_table_code = df_to_latex(df)
print(latex_table_code)
