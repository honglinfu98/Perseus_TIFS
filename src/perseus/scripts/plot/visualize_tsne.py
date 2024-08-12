from os import path
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from perseus.settings import PROJECT_ROOT


datasets = ["DDINA", "COSS", "DDM"]
dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}
models = ["GCN", "GAT", "GraphSAGE"]
label = 1


with open(path.join(PROJECT_ROOT, "data", "results.pkl"), "rb") as file:
    results = pickle.load(file)


with open(path.join(PROJECT_ROOT, "data", "results1.pkl"), "rb") as file:
    results1 = pickle.load(file)


def one_hot_to_indices(labels):
    indices = []
    for label in labels:
        index_array = np.where(label == 1)[0]
        if index_array.size > 0:
            indices.append(index_array[0])
        else:
            # Handle the case where no '1' is found; assign a default class, or handle as error
            indices.append(
                -1
            )  # Using -1 or any specific label to indicate the error/mislabeling
    return np.array(indices)


def visualize_tsne(embeddings, labels, title="t-SNE Visualization of GNN Embeddings"):
    tsne = TSNE(n_components=2, random_state=33)
    transformed_embeddings = tsne.fit_transform(embeddings)
    xs, ys = transformed_embeddings[:, 0], transformed_embeddings[:, 1]

    label_indices = one_hot_to_indices(
        labels
    )  # Convert one-hot encoded labels to indices

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xs, ys, c=label_indices, cmap="viridis")
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


# Call the visualization function
def plot_all_tsne(results, datasets, models):
    for dataset in datasets:
        for model in models:
            # Check if the model and dataset combination exists in results
            if dataset in results and model in results[dataset]:
                # Concatenate tensors for embeddings and labels
                all_embeddings = torch.cat(results[dataset][model]["embs"])
                all_labels = torch.cat(results[dataset][model]["labels"])

                # Convert labels from one-hot to indices if they are one-hot encoded
                all_labels_indices = one_hot_to_indices(all_labels)

                # Call the visualization function
                print(f"Visualizing t-SNE for {dataset} - {model}")
                visualize_tsne(
                    all_embeddings,
                    all_labels_indices,
                    title=f"t-SNE for {dataset} - {model}",
                )
            else:
                print(f"No results available for {dataset} - {model}")


plot_all_tsne(results, datasets, models)
