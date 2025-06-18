import torch
import pickle
from os import path
from perseus.settings import PROJECT_ROOT
from perseus.dataset.dataset_preparation import get_split_data_pickle_btc_noloader
from perseus.model.fused_models import MultiGraphSAGE

# --- Settings ---
DATASET = "DDM"
MODEL_NAME = "GraphSAGE"
BATCH_SIZE = 8
HIDDEN_CHANNELS = 8
LR = 0.05
WEIGHT_DECAY = 5e-4
EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Data ---
train_list, test_list, valid_list = get_split_data_pickle_btc_noloader(DATASET)
from torch_geometric.loader import DataListLoader

train_loader = DataListLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataListLoader(valid_list, batch_size=BATCH_SIZE)
test_loader = DataListLoader(test_list, batch_size=BATCH_SIZE)

# --- Model ---
in_channels = train_list[0].x.size(1)
model = MultiGraphSAGE(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS).to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.BCEWithLogitsLoss()

# --- Training Loop ---
best_val_f1 = 0.0
best_state = None
from sklearn.metrics import f1_score


def eval_epoch(model, loader):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for data_list in loader:
            data_list = [d.to(device) for d in data_list]
            logits_list = model(data_list)
            y_list = [d.y.view(-1).float() for d in data_list]
            logits = torch.cat(logits_list, dim=0)
            y = torch.cat(y_list, dim=0)
            preds = (logits.sigmoid() > 0.5).long()
            all_preds.append(preds.cpu())
            all_y.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()
    f1 = f1_score(all_y, all_preds, average="macro", zero_division=0)
    return f1


for epoch in range(1, EPOCHS + 1):
    model.train()
    for data_list in train_loader:
        data_list = [d.to(device) for d in data_list]
        logits_list = model(data_list)
        y_list = [d.y.view(-1).float() for d in data_list]
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        optimizer.zero_grad()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    val_f1 = eval_epoch(model, valid_loader)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
    print(f"Epoch {epoch}/{EPOCHS}, Val F1: {val_f1:.4f}")

# --- Save the best model ---
model.load_state_dict(best_state)
model.eval()

# Save in the format expected by mastermind_detection_from_multirun.py
save_dict = {
    "DDM": {
        "GraphSAGE": {
            BATCH_SIZE: {
                "model": model,
                "model_weights": best_state,
                # Optionally add more metadata if needed
            }
        }
    }
}

save_path = path.join(PROJECT_ROOT, "data", "buffer", "model_saved.pkl")
with open(save_path, "wb") as f:
    pickle.dump(save_dict, f)

print(f"Model saved to {save_path}")
