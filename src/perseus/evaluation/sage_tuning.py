from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc
from perseus.dataset.dataset_preparation import get_split_data_pickle_wn
from perseus.model.sage_multi import GraphSAGENet, run_experiment
from perseus.settings import PROJECT_ROOT


def plot_single_heatmap(pivot_table, font_size=12, tick_size=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = sns.light_palette("green", as_cmap=True)

    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=font_size)
    plt.rc("axes", labelsize=font_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)

    sns.heatmap(pivot_table, annot=False, cmap=cmap, ax=ax, cbar=True)

    ax.set_xlabel("Hidden Channels - Layers")
    ax.set_ylabel("Learning Rate")

    x_labels = [f"{hc} - {ly}" for hc, ly in pivot_table.columns]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)

    ax.set_yticklabels([f"{lr:.0e}" for lr in pivot_table.index], rotation=0)

    # Coordinates for the red cross (8 hidden channels, 2 layers, 5e-04 learning rate)
    hc_idx = list(pivot_table.columns).index(
        (8, 2)
    )  # find index for 8 hidden channels and 2 layers
    lr_idx = list(pivot_table.index).index(5e-4)  # find index for 5e-04 learning rate

    # Plot the red cross
    ax.plot(hc_idx + 0.5, lr_idx + 0.5, marker="x", markersize=15, color="red", mew=3)

    plt.tight_layout()
    plt.savefig(path.join(PROJECT_ROOT, "data", "ddm_auc_heatmap_benchmark.pdf"))
    plt.show()


def run_experiments():
    dataset_name = "DDM"
    num_features = 13
    train_loader, test_loader, _ = get_split_data_pickle_wn(dataset_name)

    learning_rates = [5e-6, 5e-5, 5e-4, 5e-3, 5e-2]
    hidden_channels_list = [2, 8, 32, 128, 512]
    num_layers_list = [2, 3, 4, 5, 6]
    num_epochs = 100

    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {}
        for lr in learning_rates:
            for hidden_channels in hidden_channels_list:
                for num_layers in num_layers_list:
                    model = GraphSAGENet(num_features, hidden_channels, 1, num_layers)
                    future = executor.submit(
                        run_experiment,
                        model,
                        train_loader,
                        test_loader,
                        lr,
                        num_epochs,
                    )
                    config = {
                        "lr": lr,
                        "hidden_channels": hidden_channels,
                        "num_layers": num_layers,
                    }
                    futures[future] = config

        for future in as_completed(futures):
            config = futures[future]
            experiment_result = future.result()
            result_with_config = {"config": config, "result": experiment_result}
            results.append(result_with_config)
    return results


if __name__ == "__main__":
    # results = run_experiments()
    # with open(path.join(PROJECT_ROOT, "data", "results_tuning_sage_weighted.pkl"), "wb") as file:
    #     pickle.dump(results, file)
    with open(
        path.join(PROJECT_ROOT, "data", "results_tuning_sage_weighted.pkl"), "rb"
    ) as file:
        results = pickle.load(file)

    data = []
    for result in results:
        config = result["config"]
        fpr = result["result"][2][0]
        tpr = result["result"][3][0]
        auc_score = auc(fpr, tpr)

        # Check if AUC is less than 0.5
        if auc_score < 0.5:
            # Reverse the classification by subtracting probabilities from 1
            fpr = 1 - fpr
            tpr = 1 - tpr
            auc_score = auc(fpr, tpr)  # Recalculate AUC with reversed probabilities

        data.append(
            {
                "Learning Rate": config["lr"],
                "Hidden Channels": config["hidden_channels"],
                "Layers": config["num_layers"],
                "AUC Score": auc_score,
            }
        )

    df = pd.DataFrame(data)

    # Create a pivot table for the heatmap
    pivot_table = df.pivot_table(
        index="Learning Rate",
        columns=["Hidden Channels", "Layers"],
        values="AUC Score",
    )

    # Plot the heatmap for DDINA
    plot_single_heatmap(pivot_table, font_size=25, tick_size=25)
