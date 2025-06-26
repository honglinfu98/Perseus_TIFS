import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataListLoader
from perseus.dataset.dataset_preparation import get_split_data_pickle_btc_noloader
from perseus.model.fused_models import (
    MultiGAT,
    MultiGraphSAGE,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from perseus.settings import PROJECT_ROOT
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
)
import pickle
from sklearn.metrics import roc_curve
import os
from torch.optim.adam import Adam


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = total_nodes = 0
    for data_list in loader:
        data_list = [d.to(device) for d in data_list]
        logits_list = model(data_list)
        y_list = [d.y.view(-1).float() for d in data_list]
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        optimizer.zero_grad()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = (logits.sigmoid() > 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_nodes += y.size(0)
    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_nodes = 0
    all_preds, all_y = [], []
    for data_list in loader:
        data_list = [d.to(device) for d in data_list]
        logits_list = model(data_list)
        y_list = [d.y.view(-1).float() for d in data_list]
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = (logits.sigmoid() > 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_nodes += y.size(0)
        all_preds.append(preds.cpu())
        all_y.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()
    f1 = f1_score(all_y, all_preds, zero_division=0)
    precision = precision_score(all_y, all_preds, zero_division=0)
    recall = recall_score(all_y, all_preds, zero_division=0)
    acc = accuracy_score(all_y, all_preds)
    mcc = matthews_corrcoef(all_y, all_preds)
    return (total_loss / total_nodes, acc, f1, precision, recall, mcc)


def run_experiment(
    data_name,
    model_cls,
    hidden_channels,
    lr,
    weight_decay,
    epochs=100,
    batch_size=8,
    device_str=None,
    return_roc=False,
    return_timings=False,
    return_embs=False,
):
    import time
    from sklearn.metrics import roc_curve

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_list, test_list, valid_list = get_split_data_pickle_btc_noloader(data_name)
    train_loader = DataListLoader(train_list, batch_size=batch_size, shuffle=True)
    valid_loader = DataListLoader(valid_list, batch_size=batch_size)
    test_loader = DataListLoader(test_list, batch_size=batch_size)
    in_channels = train_list[0].x.size(1)
    model = model_cls(in_channels=in_channels, hidden_channels=hidden_channels).to(
        device
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    best_state = None
    train_times = []
    for epoch in range(1, epochs + 1):
        if return_timings:
            start_time = time.time()
        train_epoch(model, train_loader, optimizer, criterion, device)
        if return_timings:
            train_times.append(time.time() - start_time)
        _, _, val_f1, _, _, _ = eval_epoch(model, valid_loader, criterion, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    # Collect all_probs and all_labels on test set
    model.eval()
    all_probs, all_labels = [], []
    batch_times, num_nodes, embs = [], [], []
    with torch.no_grad():
        for data_list in test_loader:
            data_list = [d.to(device) for d in data_list]
            if return_timings:
                batch_start = time.time()
            logits_list = model(data_list)
            if return_timings:
                batch_times.append(time.time() - batch_start)
            y_list = [d.y.view(-1).float() for d in data_list]
            x_list = [d.x for d in data_list]
            logits = torch.cat(logits_list, dim=0)
            y = torch.cat(y_list, dim=0)
            all_probs.append(logits.cpu())
            all_labels.append(y.cpu())
            if return_embs:
                # Optionally collect embeddings if model supports it
                if hasattr(model, "get_embeddings"):
                    embs.append(model.get_embeddings(data_list))
                else:
                    embs.append([])
            for x in x_list:
                num_nodes.append(x.shape[0])
    all_probs = torch.cat(all_probs, dim=0).sigmoid().numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    test_loss, test_acc, test_f1, test_precision, test_recall, test_mcc = eval_epoch(
        model, test_loader, criterion, device
    )
    results = {
        "data": data_name,
        "model": model_cls.__name__,
        "hidden_channels": hidden_channels,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_mcc": test_mcc,
        "model_instance": model,
        "model_weights": model.state_dict(),
        "all_probs": all_probs,
        "all_labels": all_labels,
        "batch_size": batch_size,
    }
    if return_roc:
        if len(all_labels.shape) == 1 or (
            hasattr(all_labels, "shape") and all_labels.shape[1] == 1
        ):
            fpr, tpr, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
            results["fpr"] = {0: fpr}
            results["tpr"] = {0: tpr}
        else:
            fpr_dict, tpr_dict = {}, {}
            for i in range(all_labels.shape[1]):
                fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
                fpr_dict[i] = fpr
                tpr_dict[i] = tpr
            results["fpr"] = fpr_dict
            results["tpr"] = tpr_dict
    if return_timings:
        results["train_times"] = train_times
        results["batch_times"] = batch_times
        results["num_nodes"] = num_nodes
    if return_embs:
        results["embs"] = embs
    return results


def hyperparameter_search(max_workers=12):
    param_grid = list(
        itertools.product(
            ["DDM", "DDINA"],
            ["MultiGraphSAGE", "MultiGAT"],
            [8, 128, 1025],
            [0.5, 0.005, 0.00005],
            [5e-4],
        )
    )
    results = []
    # For mastermind_detection_from_multirun.py compatibility
    model_save_dict = {}
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_map = {"MultiGAT": MultiGAT, "MultiGraphSAGE": MultiGraphSAGE}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_experiment,
                data_name,
                model_map[model_cls_name],
                hidden,
                lr,
                wd,
                100,
                20,
                device_str,
            ): (data_name, model_cls_name, hidden, lr, wd)
            for data_name, model_cls_name, hidden, lr, wd in param_grid
        }
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                res = future.result()
                print(f"Completed: {cfg}")
                results.append(res)
                # Save for mastermind_detection_from_multirun.py
                data_name = res["data"]
                model_name = res["model"].replace("Multi", "")
                batch_size = res["batch_size"]
                if data_name not in model_save_dict:
                    model_save_dict[data_name] = {}
                if model_name not in model_save_dict[data_name]:
                    model_save_dict[data_name][model_name] = {}
                model_save_dict[data_name][model_name][batch_size] = {
                    "model": res["model_instance"],
                    "model_weights": res["model_weights"],
                    "all_probs": res["all_probs"],
                    "all_labels": res["all_labels"],
                }
            except Exception as exc:
                print(f"Error with config {cfg}: {exc}")
    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(PROJECT_ROOT, "data", "buffer", "hp_search_results_a.csv"),
        index=False,
    )
    print("Search complete. Results saved to hp_search_results.csv")
    # Save models and predictions for mastermind detection
    buffer_dir = os.path.join(PROJECT_ROOT, "data", "buffer")
    os.makedirs(buffer_dir, exist_ok=True)
    with open(os.path.join(buffer_dir, "model_saved_a.pkl"), "wb") as f:
        pickle.dump(model_save_dict, f)
    print(
        f"Model and predictions saved to {os.path.join(buffer_dir, 'model_saved.pkl')}"
    )


def collect_full_outputs(
    data_name,
    model_cls,
    hidden_channels,
    lr,
    weight_decay,
    batch_sizes=[8],
    epochs=100,
    device_str=None,
):
    results_by_batch_size = {}
    for batch_size in batch_sizes:
        result = run_experiment(
            data_name,
            model_cls,
            hidden_channels,
            lr,
            weight_decay,
            epochs,
            batch_size,
            device_str,
            return_roc=True,
            return_timings=True,
            return_embs=True,
        )
        labels = result["all_labels"]
        probs = result["all_probs"]
        results_by_batch_size[batch_size] = {
            "metrics": {
                "labels": labels,
                "probs": probs,
            },
            "fpr": result.get("fpr", {}),
            "tpr": result.get("tpr", {}),
            "train_times": result.get("train_times", []),
            "batch_times": result.get("batch_times", []),
            "num_nodes": result.get("num_nodes", []),
            "embs": result.get("embs", []),
            "labels": [labels],
        }
    return results_by_batch_size


def export_results_for_plot(
    csv_path=os.path.join(PROJECT_ROOT, "data", "buffer", "hp_search_results_a.csv"),
    out_path=os.path.join(PROJECT_ROOT, "data", "buffer", "new_results_btc.pkl"),
    batch_sizes=range(2, 21),
):
    import concurrent.futures

    df = pd.read_csv(csv_path)
    best_configs = (
        df.sort_values("test_f1", ascending=False)
        .groupby(["data", "model"])
        .first()
        .reset_index()
    )
    model_map = {"MultiGAT": MultiGAT, "MultiGraphSAGE": MultiGraphSAGE}
    results = {d: {} for d in df["data"].unique()}
    jobs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_cfg = {}
        for _, row in best_configs.iterrows():
            data_name = row["data"]
            model_name = row["model"]
            model_cls = model_map[model_name]
            print(f"Submitting full output for {data_name} {model_name}...")
            future = executor.submit(
                collect_full_outputs,
                data_name,
                model_cls,
                int(row["hidden_channels"]),
                float(row["lr"]),
                float(row["weight_decay"]),
                list(batch_sizes),
                100,
            )
            future_to_cfg[future] = (data_name, model_name)
        for future in concurrent.futures.as_completed(future_to_cfg):
            data_name, model_name = future_to_cfg[future]
            try:
                batch_size_results = future.result()
                results[data_name][model_name.replace("Multi", "")] = batch_size_results
                print(f"Completed: {data_name} {model_name}")
            except Exception as exc:
                print(f"Error with {data_name} {model_name}: {exc}")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results for plotting to {out_path}")


if __name__ == "__main__":
    hyperparameter_search()

    # Export results for plotting
    export_results_for_plot()
