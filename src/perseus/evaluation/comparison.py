import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
from torch_geometric.nn import Node2Vec
import pandas as pd
from perseus.dataset.dataset_preparation import get_split_data_pickle_l
import random

# Load dataset (a, b, c)
a, b, c = get_split_data_pickle_l("DDINA")


# Helper function to evaluate the model's performance
def evaluate_performance(y_true, y_pred_probs, threshold=0.54):
    y_pred = (y_pred_probs >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

# Placeholder for results
results = []

# Load the dataset (a, b, c) – assuming `a`, `b`, `c` are data loaders or lists of DataBatches
# Extract features (x) and labels (y) from training and validation sets
x_train = torch.cat([batch.x for batch in a], dim=0).numpy()
y_train = torch.cat([batch.y for batch in a], dim=0).numpy()

x_val = torch.cat([batch.x for batch in b], dim=0).numpy()
y_val = torch.cat([batch.y for batch in b], dim=0).numpy()

#########################################
# 1. Linear Regression
#########################################
linear_model = LogisticRegression()  # Using LogisticRegression for probability output
linear_model.fit(x_train, y_train)

# Predict probabilities
y_pred_probs_linear = linear_model.predict_proba(x_val)[:, 1]
precision, recall, f1, accuracy = evaluate_performance(y_val, y_pred_probs_linear)

# Save results
results.append(["Linear Regression", precision, f1, accuracy, recall])

#########################################
# 2. DeepWalk (Node2Vec)
#########################################
# Use edge_index for graph representation (extracted from your dataset)
edge_index = torch.cat([batch.edge_index for batch in a], dim=1)
node2vec = Node2Vec(edge_index, embedding_dim=64, walk_length=30, context_size=10, walks_per_node=10)
node2vec = node2vec.to('cpu')  # Adjust based on your hardware

# Train Node2Vec
optimizer = torch.optim.Adam(node2vec.parameters(), lr=0.01)
for epoch in range(100):  # Adjust epochs based on your needs
    optimizer.zero_grad()
    loss = node2vec.loss()
    loss.backward()
    optimizer.step()

# Extract embeddings
embeddings = node2vec().detach().numpy()

# Train Logistic Regression on DeepWalk embeddings
deepwalk_model = LogisticRegression()
deepwalk_model.fit(embeddings, y_train)

# Predict probabilities using DeepWalk embeddings
y_pred_probs_deepwalk = deepwalk_model.predict_proba(embeddings)[:, 1]
precision, recall, f1, accuracy = evaluate_performance(y_val, y_pred_probs_deepwalk)

# Save results
results.append(["DeepWalk", precision, f1, accuracy, recall])

#########################################
# 3. DeepWalk + LSTM
#########################################
class DeepWalkLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DeepWalkLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = embeddings.shape[1]
hidden_size = 128
num_layers = 2
output_size = 1

lstm_model = DeepWalkLSTM(input_size, hidden_size, num_layers, output_size).to('cpu')
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

x_lstm_train = torch.tensor(embeddings).view(-1, 1, input_size).float()
y_lstm_train = torch.tensor(y_train).float()

# Train LSTM
for epoch in range(100):
    optimizer.zero_grad()
    outputs = lstm_model(x_lstm_train)
    loss = criterion(outputs.squeeze(), y_lstm_train)
    loss.backward()
    optimizer.step()

# Predict probabilities using LSTM
with torch.no_grad():
    y_pred_probs_lstm = torch.sigmoid(lstm_model(x_lstm_train)).numpy().squeeze()
precision, recall, f1, accuracy = evaluate_performance(y_val, y_pred_probs_lstm)

# Save results
results.append(["DeepWalk + LSTM", precision, f1, accuracy, recall])

#########################################
# 4. Random Forest
#########################################
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Predict probabilities using Random Forest
y_pred_probs_rf = rf_model.predict_proba(x_val)[:, 1]
precision, recall, f1, accuracy = evaluate_performance(y_val, y_pred_probs_rf)

# Save results
results.append(["Random Forest", precision, f1, accuracy, recall])

#########################################
# Output Results
#########################################
results_df = pd.DataFrame(results, columns=["model_dataset", "precision", "f1", "accuracy", "recall"])

# Display results as a table
import ace_tools as tools; tools.display_dataframe_to_user(name="Model Evaluation Results", dataframe=results_df)
