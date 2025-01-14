from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc
from perseus.dataset.dataset_preparation import get_split_data_pickle_l
from perseus.model.sage_multi import GraphSAGENet, run_experiment
from perseus.settings import PROJECT_ROOT


def plot_heatmaps(df, datasets, title_mapping, font_size=12, tick_size=10):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
    cmap = sns.light_palette("green", as_cmap=True)
    norm = plt.Normalize(df["AUC Score"].min(), df["AUC Score"].max())

    # Adjust font sizes globally
    plt.rc("font", size=font_size)  # controls default text size
    plt.rc("axes", titlesize=font_size)  # fontsize of the title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=tick_size)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=tick_size)  # fontsize of the y tick labels

    for i, pivot_table in enumerate(dfs):
        ax = axes[i]
        sns.heatmap(
            pivot_table, annot=False, fmt=".2f", cmap=cmap, norm=norm, ax=ax, cbar=False
        )
        dataset_name = datasets[i]
        ax.set_title(
            title_mapping.get(dataset_name, dataset_name)
        )  # Set title from mapping
        ax.set_xlabel("Learning Rate")
        if i == 0:
            ax.set_ylabel("Hidden Channels - Layers")
        else:
            ax.set_ylabel("")

        # Set x-axis tick labels
        ax.set_xticklabels([f"{lr:.0e}" for lr in pivot_table.columns], rotation=0)

        # Automatically adjust the number of y-axis labels to match tick positions
        yticks = ax.get_yticks()
        labels = [f"{hc} - {ly}" for hc, ly in pivot_table.index]
        ax.set_yticklabels(labels[: len(yticks)], rotation=0)

    # Shared color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
    plt.savefig(path.join(PROJECT_ROOT, "data", "auc_heatmaps_combined.pdf"))
    plt.show()
    plt.close(fig)
    print("Combined plot saved successfully.")


def run_experiments_for_dataset(dataset_name, num_features):
    train_loader, test_loader, _ = get_split_data_pickle_l(dataset_name)
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    hidden_channels_list = [8, 16, 32, 64, 128]
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
            result_with_config = {
                "dataset": dataset_name,
                "config": config,
                "result": experiment_result,
            }
            results.append(result_with_config)
    return results


if __name__ == "__main__":
    datasets = ["DDINA", "DDM"]
    num_features_dict = {
        "DDINA": 14,
        "DDM": 14,
    }  # Adjust as per actual dataset features
    all_results = []

    for dataset in datasets:
        results = run_experiments_for_dataset(dataset, num_features_dict[dataset])
        all_results.extend(results)

    data = []
    for result in all_results:
        config = result["config"]
        dataset_name = result["dataset"]
        fpr = result["result"][2][0]
        tpr = result["result"][3][0]
        auc_score = auc(fpr, tpr)
        data.append(
            {
                "Dataset": dataset_name,
                "Learning Rate": config["lr"],
                "Hidden Channels": config["hidden_channels"],
                "Layers": config["num_layers"],
                "AUC Score": auc_score,
            }
        )

    df = pd.DataFrame(data)

    # Create a pivot table for each dataset
    dfs = []
    for dataset in datasets:
        sub_df = df[df["Dataset"] == dataset]
        pivot_table = sub_df.pivot_table(
            index=["Hidden Channels", "Layers"],
            columns="Learning Rate",
            values="AUC Score",
        )
        dfs.append(pivot_table)

        # Define the title mapping
    title_mapping = {"DDINA": "Directed Diffusion", "DDM": "Weighted Diffusion"}

    # Call the function with custom font and tick sizes and title mapping
    plot_heatmaps(df, datasets, title_mapping, font_size=25, tick_size=25)
