import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from perseus.dataset.dataset_preparation import get_split_data_pickle_wn

# Assuming train_loader and valid_loader are already initialized
train_loader, valid_loader, _ = get_split_data_pickle_wn("DDM")


# Prepare function to extract data from the DataLoader
def extract_data_from_loader(data_loader):
    X_list = []
    y_list = []

    for batch in data_loader:
        # Assuming batch.x are the node features and batch.y are the labels
        X_list.append(batch.x)  # Node features
        y_list.append(batch.y)  # Labels

    # Concatenate all the batches into one matrix for X and y
    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy().flatten()  # Ensure y is 1D
    return X, y


# Extract train and valid data
X_train, y_train = extract_data_from_loader(train_loader)
X_valid, y_valid = extract_data_from_loader(valid_loader)

# Initialize and train the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

# Get the predicted scores for ROC curve
y_scores_lin_reg = lin_reg_model.predict(X_valid)

# Compute ROC curve and AUC for Linear Regression
fpr_lin_reg, tpr_lin_reg, thresholds_lin_reg = roc_curve(y_valid, y_scores_lin_reg)
roc_auc_lin_reg = auc(fpr_lin_reg, tpr_lin_reg)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get predicted probabilities for Random Forest (used for ROC)
y_scores_rf = rf_model.predict_proba(X_valid)[:, 1]

# Compute ROC curve and AUC for Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_valid, y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Find the best threshold for F1 score
thresholds = np.linspace(0.01, 0.99, 100)
f1_scores = [f1_score(y_valid, y_scores_rf > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)

# Calculate all metrics at the best threshold
y_pred_rf_optimal = (y_scores_rf >= best_threshold).astype(int)
mcc = matthews_corrcoef(y_valid, y_pred_rf_optimal)
f1 = f1_score(y_valid, y_pred_rf_optimal)
precision = precision_score(y_valid, y_pred_rf_optimal)
recall = recall_score(y_valid, y_pred_rf_optimal)
accuracy = accuracy_score(y_valid, y_pred_rf_optimal)

# Plotting both ROC curves
plt.figure()
plt.plot(
    fpr_lin_reg,
    tpr_lin_reg,
    color="darkorange",
    lw=2,
    label=f"Linear Regression ROC (AUC = {roc_auc_lin_reg:.2f})",
)
plt.plot(
    fpr_rf,
    tpr_rf,
    color="blue",
    lw=2,
    label=f"Random Forest ROC (AUC = {roc_auc_rf:.2f})",
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Linear Regression and Random Forest")
plt.legend(loc="lower right")
plt.show()

# Print the best threshold and metrics
print(
    f"Best Threshold: {best_threshold:.2f}, MCC: {mcc:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}"
)
