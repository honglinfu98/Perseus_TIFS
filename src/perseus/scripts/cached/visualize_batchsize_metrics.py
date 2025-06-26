import pickle
import os
import matplotlib.pyplot as plt

from perseus.settings import PROJECT_ROOT


def compute_f1_precision(labels, probs, threshold=0.7):
    from sklearn.metrics import f1_score, precision_score

    preds = (probs > threshold).astype(int)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    return f1, precision


if __name__ == "__main__":
    results_path = os.path.join(PROJECT_ROOT, "data", "buffer", "results_btc.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    for dataset, models in results.items():
        for model, batchsize_results in models.items():
            batch_sizes = []
            f1s = []
            precisions = []
            for batch_size, res in batchsize_results.items():
                labels = res["metrics"]["labels"]
                probs = res["metrics"]["probs"]
                f1, precision = compute_f1_precision(labels, probs)
                batch_sizes.append(batch_size)
                f1s.append(f1)
                precisions.append(precision)

            plt.figure(figsize=(8, 5))
            plt.plot(batch_sizes, f1s, marker="o", label="F1 Score")
            plt.plot(batch_sizes, precisions, marker="s", label="Precision")
            plt.xlabel("Batch Size")
            plt.ylabel("Score")
            plt.title(f"{dataset} - {model}: F1 and Precision vs. Batch Size")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
