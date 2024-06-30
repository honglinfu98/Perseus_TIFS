import torch
from torch_geometric.data import Data
import networkx as nx
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from perseus.model.magamaga import split_data_noloader

_,_,data_list = split_data_noloader("DDINA")

# Step 1: Calculate closeness centrality and prepare the label list
centrality_list = []
y_true = []

for data in data_list:
    # Convert to NetworkX graph
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    
    # Calculate the closeness centrality for each node
    centrality = nx.closeness_centrality(G)
    
    # Assuming the second column of `y` indicates 'anomaly'
    binary_labels = data.y[:, 0].numpy()
    
    # Append centrality values and corresponding labels to the lists
    centrality_list.extend(list(centrality.values()))
    y_true.extend(binary_labels)

# Step 2: Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_true, centrality_list)
roc_auc = auc(fpr, tpr)

# Step 3: Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Anomaly Detection')
plt.legend(loc="lower right")
plt.show()
