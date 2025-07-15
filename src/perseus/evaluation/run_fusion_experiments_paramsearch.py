import itertools
import torch
import pickle
import os
import random
import numpy as np
from perseus.evaluation.run_fusion_experiments import (
    MultiGAT,
    MultiGraphSAGE,
    get_split_data_pickle_btc_noloader,
    run_experiment,
    collect_full_outputs,
)
from perseus.settings import PROJECT_ROOT

# Hyperparameter search space
HIDDEN_CHANNELS_LIST = [8, 32, 64]  # , 128, 512]
LR_LIST = [0.05, 0.005, 0.0005]  # , 0.00005, 0.000005]
WEIGHT_DECAY = 5e-4
EPOCHS = 10
BATCH_SIZE = 8  # For hyperparam search
BATCH_SIZES_SWEEP = list(range(2, 21, 6))

DATA_NAMES = ["DDM", "DDINA"]
MODEL_NAMES = ["MultiGAT", "MultiGraphSAGE"]
MODEL_MAP = {"MultiGAT": MultiGAT, "MultiGraphSAGE": MultiGraphSAGE}

# Output paths
HYPERPARAM_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "data", "buffer", "paramsearch_hyperparam_results.pkl"
)
BATCHSWEEP_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "data", "buffer", "paramsearch_batchsweep_results.pkl"
)
FLAT_PARAM_RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "data", "buffer", "paramsearch_hyperparam_results_flat.pkl"
)

SEED = 77
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def hyperparam_search():
    """
    Run hyperparameter search for all (data, model) pairs.
    Returns a dict: results[data][model] = list of dicts (one per config)
    Also returns a flat list of all results for heatmap/plotting.
    """
    results = {d: {m: [] for m in MODEL_NAMES} for d in DATA_NAMES}
    flat_results = []
    param_grid = list(
        itertools.product(DATA_NAMES, MODEL_NAMES, HIDDEN_CHANNELS_LIST, LR_LIST)
    )
    for data_name, model_name, hidden_channels, lr in param_grid:
        model_cls = MODEL_MAP[model_name]
        print(f"Running {data_name} {model_name} hidden={hidden_channels} lr={lr}")
        try:
            res = run_experiment(
                data_name,
                model_cls,
                hidden_channels,
                lr,
                WEIGHT_DECAY,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                device_str=None,
                return_roc=False,
                return_timings=False,
                return_embs=False,
                seed=42,
            )
            result_dict = {
                "data": data_name,
                "model": model_name,
                "hidden_channels": hidden_channels,
                "lr": lr,
                "batch_size": BATCH_SIZE,
                "best_val_f1": res.get("best_val_f1", 0),
                "test_f1": res.get("test_f1", 0),
                "test_acc": res.get("test_acc", 0),
                "test_precision": res.get("test_precision", 0),
                "test_recall": res.get("test_recall", 0),
                "test_mcc": res.get("test_mcc", 0),
            }
            results[data_name][model_name].append(result_dict)
            flat_results.append(result_dict)
        except Exception as e:
            print(
                f"Error for {data_name} {model_name} hidden={hidden_channels} lr={lr}: {e}"
            )
    with open(HYPERPARAM_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    with open(FLAT_PARAM_RESULTS_PATH, "wb") as f:
        pickle.dump(flat_results, f)
    print(f"Saved hyperparameter search results to {HYPERPARAM_RESULTS_PATH}")
    print(f"Saved flat hyperparameter results to {FLAT_PARAM_RESULTS_PATH}")
    return results, flat_results


def select_best_configs(hyperparam_results):
    """
    For each (data, model), select the config with the highest best_val_f1.
    Returns dict: best_configs[data][model] = (hidden_channels, lr)
    """
    print("\n" + "=" * 50)
    print("SELECTING BEST HYPERPARAMETER CONFIGS")
    print("=" * 50)

    best_configs = {d: {} for d in DATA_NAMES}
    for data_name in DATA_NAMES:
        for model_name in MODEL_NAMES:
            configs = hyperparam_results[data_name][model_name]
            if not configs:
                best_configs[data_name][model_name] = None
                print(f"❌ No configs found for {data_name} {model_name}")
                continue
            best = max(configs, key=lambda x: x["best_val_f1"])
            best_configs[data_name][model_name] = (best["hidden_channels"], best["lr"])
            print(
                f"✓ Best for {data_name} {model_name}: hidden={best['hidden_channels']} lr={best['lr']} val_f1={best['best_val_f1']:.4f}"
            )

    print("=" * 50)
    return best_configs


def batchsize_sweep(best_configs):
    """
    For each (data, model), run collect_full_outputs with the best config for batch sizes 2-20.
    Save results to BATCHSWEEP_RESULTS_PATH.
    Always produce a dict for every (data, model) pair. If missing, set to a dict mapping batch_size to empty dicts for all batch sizes.
    """
    results = {d: {m: {} for m in MODEL_NAMES} for d in DATA_NAMES}
    param_tuning_results = []

    for data_name in DATA_NAMES:
        for model_name in MODEL_NAMES:
            config = best_configs[data_name][model_name]
            if config is None:
                print(
                    f"Warning: Skipping {data_name} {model_name} (no valid config); setting to empty dicts for all batch sizes."
                )
                results[data_name][model_name] = {bs: {} for bs in BATCH_SIZES_SWEEP}
                continue
            hidden_channels, lr = config
            model_cls = MODEL_MAP[model_name]
            print(
                f"Batch sweep for {data_name} {model_name} hidden={hidden_channels} lr={lr}"
            )
            try:
                res = collect_full_outputs(
                    data_name,
                    model_cls,
                    hidden_channels,
                    lr,
                    WEIGHT_DECAY,
                    batch_sizes=BATCH_SIZES_SWEEP,
                    epochs=EPOCHS,
                    device_str=None,
                )
                results[data_name][model_name] = res

                # Collect param tuning results for each batch size (for f8_parameter_tuning.py)
                for bsz, info in res.items():
                    param_tuning_results.append(
                        {
                            "batch_size": bsz,
                            "lr": info.get("lr", lr),
                            "hidden_channels": info.get(
                                "hidden_channels", hidden_channels
                            ),
                            "model": model_name,
                            "data": data_name,
                            "best_val_f1": info.get("best_val_f1", 0),
                        }
                    )

            except Exception as e:
                print(
                    f"Warning: Error in batch sweep for {data_name} {model_name}: {e}; setting to empty dicts for all batch sizes."
                )
                results[data_name][model_name] = {bs: {} for bs in BATCH_SIZES_SWEEP}

    # Save batch size sweep results
    with open(BATCHSWEEP_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved batch size sweep results to {BATCHSWEEP_RESULTS_PATH}")

    # Save param tuning results compatible with f8_parameter_tuning.py
    param_tuning_path = os.path.join(
        PROJECT_ROOT, "data", "buffer", "param_tuning_results.pkl"
    )
    with open(param_tuning_path, "wb") as f:
        pickle.dump(param_tuning_results, f)
    print(f"Saved param tuning results to {param_tuning_path}")

    return results


def save_results_for_plotting(results):
    """
    Save results in the format expected by the three-step workflow:
    1. f8_parameter_tuning.py - for parameter search visualization
    2. f7_plot_f1_precision.py - for performance analysis with best parameters
    3. t3_multi_results.py - for final results table with best F1 scores
    """
    print("\n" + "=" * 60)
    print("SAVING RESULTS FOR WORKFLOW")
    print("=" * 60)

    # Step 1: Results for f8_parameter_tuning.py (parameter search heatmaps)
    # This was already saved in batchsize_sweep() as param_tuning_results.pkl
    print("✓ Step 1: Parameter search results saved for f8_parameter_tuning.py")
    print("  → Use this to visualize hyperparameter search results")

    # Step 2: Results for f7_plot_f1_precision.py (performance analysis)
    # Save both flattened (for old plots) and full batch structure (for batch analysis)

    # Full batch structure for batch size analysis
    f7_full_results_path = os.path.join(
        PROJECT_ROOT, "data", "buffer", "new_results_btc_seed7_batch_2_20.pkl"
    )
    with open(f7_full_results_path, "wb") as f:
        pickle.dump(results, f)
    print("✓ Step 2a: Full batch results saved for f7_plot_f1_precision.py")
    print("  → Use this for batch size vs performance analysis")

    # Flattened results for traditional plots (batch_size=2 default)
    flattened_results = {d: {m: {} for m in MODEL_NAMES} for d in DATA_NAMES}
    default_batch_size = 2

    for data_name in DATA_NAMES:
        for model_name in MODEL_NAMES:
            batch_results = results[data_name][model_name]
            if default_batch_size in batch_results:
                flattened_results[data_name][model_name] = batch_results[
                    default_batch_size
                ]

    f7_flat_results_path = os.path.join(
        PROJECT_ROOT, "data", "buffer", "new_results_btc.pkl"
    )
    with open(f7_flat_results_path, "wb") as f:
        pickle.dump(flattened_results, f)
    print("✓ Step 2b: Flattened results saved for f7_plot_f1_precision.py")
    print("  → Use this for F1/precision curves and traditional plots")

    # Step 3: Results for t3_multi_results.py (final results table)
    # Uses the same full batch structure file
    print("✓ Step 3: Results ready for t3_multi_results.py")
    print("  → Use this to extract best F1 scores for final comparison tables")

    print("\nWORKFLOW SUMMARY:")
    print("1. Run f8_parameter_tuning.py → Visualize parameter search heatmaps")
    print("2. Run f7_plot_f1_precision.py → Analyze performance with best parameters")
    print("3. Run t3_multi_results.py → Generate final results table")
    print("=" * 60)


def print_workflow_instructions():
    """
    Print clear instructions for the three-step workflow.
    """
    print("\n" + "=" * 70)
    print("NEXT STEPS: THREE-STEP ANALYSIS WORKFLOW")
    print("=" * 70)
    print("Now run the following scripts in order:")
    print()
    print("1. STEP 1: Hyperparameter Analysis")
    print("   cd src/perseus/scripts/plot")
    print("   python f8_parameter_tuning.py")
    print("   → Generates heatmaps showing how hyperparameters affect performance")
    print("   → Visualizes the parameter search results")
    print()
    print("2. STEP 2: Performance Analysis")
    print("   python f7_plot_f1_precision.py")
    print("   → Generates F1/precision curves using best hyperparameters")
    print("   → Shows performance across different batch sizes")
    print("   → Creates ROC curves and timing analysis")
    print()
    print("3. STEP 3: Final Results Table")
    print("   cd ../exhibition")
    print("   python t3_multi_results.py")
    print("   → Generates final comparison table with best F1 scores")
    print("   → Combines with other baseline methods")
    print("   → Creates publication-ready results")
    print()
    print("FILES GENERATED:")
    print("• param_tuning_results.pkl → for f8_parameter_tuning.py")
    print("• new_results_btc.pkl → for f7_plot_f1_precision.py (flattened)")
    print("• new_results_btc_seed7_batch_2_20.pkl → for f7 & t3 (full batch structure)")
    print("=" * 70)


if __name__ == "__main__":
    print("Starting Parameter Search and Batch Size Sweep...")
    print(
        "This will find the best hyperparameters and test them across different batch sizes."
    )
    print()

    # Step 1: Hyperparameter search
    hyperparam_results, flat_results = hyperparam_search()

    # Step 2: Select best configs
    best_configs = select_best_configs(hyperparam_results)

    # Step 3: Batch size sweep with best configs
    results = batchsize_sweep(best_configs)

    # Step 4: Save results for plotting workflow
    save_results_for_plotting(results)

    # Step 5: Print workflow instructions
    print_workflow_instructions()
