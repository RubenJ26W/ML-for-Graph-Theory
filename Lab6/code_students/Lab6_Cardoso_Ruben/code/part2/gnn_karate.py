"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import models
importlib.reload(models)
import utils
import importlib
importlib.reload(utils)


# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 100
n_hidden_1 = 8
n_hidden_2 = 16
learning_rate = 0.01
dropout_rate = 0.1

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print(G.number_of_nodes())
print(G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)
n_class = 2

adj = nx.adjacency_matrix(G) # Obtains the adjacency matrix
adj = normalize_adjacency(adj) # Normalizes the adjacency matrix

# ---------------------------------------------------------------------
# 1) Features: canonical basis I_n (n x n identity matrix)
# ---------------------------------------------------------------------
X = np.eye(n, dtype=np.float32)          # shape (n, n)
X = torch.from_numpy(X).to(device)       # float32 tensor on the selected device

# Labels as a tensor
y_t = torch.from_numpy(y).long().to(device)

if not isinstance(adj, np.ndarray):
    adj = adj.toarray()

adj_t = torch.from_numpy(adj.astype(np.float32)).to(device)   # shape (n, n)

# ---------------------------------------------------------------------
# 3) Train/Test split identical to the one used for DeepWalk
# ---------------------------------------------------------------------
rng = np.random.RandomState(seed=42)
idx_all = rng.permutation(n)
split = int(0.8 * n)
idx_train = idx_all[:split]
idx_test  = idx_all[split:]

idx_train_t = torch.from_numpy(idx_train).long().to(device)
idx_test_t  = torch.from_numpy(idx_test).long().to(device)

# ---------------------------------------------------------------------
# 4) GNN model, optimizer, and loss function
# ---------------------------------------------------------------------
model = GNN(
    n_feat=X.shape[1],
    n_hidden_1=n_hidden_1,
    n_hidden_2=n_hidden_2,
    n_class=n_class,
    dropout=dropout_rate
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = nn.NLLLoss()   # because the forward pass returns F.log_softmax(...)

# ---------------------------------------------------------------------
# 5) Training loop
# ---------------------------------------------------------------------
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(X, adj_t)                     # shape (n, n_class)
    loss = criterion(out[idx_train_t], y_t[idx_train_t])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# ---------------------------------------------------------------------
# 6) Evaluation on the test set
# ---------------------------------------------------------------------
model.eval()
with torch.no_grad():
    out = model(X, adj_t)
    preds = out[idx_test_t].max(1)[1]        # argmax over log-probabilities
    correct = preds.eq(y_t[idx_test_t]).sum().item()
    acc = correct / len(idx_test)
    print(f"GNN Karate test accuracy: {acc:.4f}")

#The accuracy is exactly 1. Not surprizing since the dataset is very small. 




############## Task 12

# ---------------------------------------------------------------------
# 1) Features: constant features 
# ---------------------------------------------------------------------
# 
X = np.ones((n, n), dtype=np.float32)    # shape (n, n), all ones
X = torch.from_numpy(X).to(device)       # float32 tensor on the selected device
# Labels as a tensor
y_t = torch.from_numpy(y).long().to(device)

if not isinstance(adj, np.ndarray):
    adj = adj.toarray()

adj_t = torch.from_numpy(adj.astype(np.float32)).to(device)   # shape (n, n)

# ---------------------------------------------------------------------
# 3) Train/Test split identical to the one used for DeepWalk
# ---------------------------------------------------------------------
rng = np.random.RandomState(seed=42)
idx_all = rng.permutation(n)
split = int(0.8 * n)
idx_train = idx_all[:split]
idx_test  = idx_all[split:]

idx_train_t = torch.from_numpy(idx_train).long().to(device)
idx_test_t  = torch.from_numpy(idx_test).long().to(device)

# ---------------------------------------------------------------------
# 4) GNN model, optimizer, and loss function
# ---------------------------------------------------------------------
model = GNN(
    n_feat=X.shape[1],
    n_hidden_1=n_hidden_1,
    n_hidden_2=n_hidden_2,
    n_class=n_class,
    dropout=dropout_rate
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = nn.NLLLoss()   # because the forward pass returns F.log_softmax(...)

# ---------------------------------------------------------------------
# 5) Training loop
# ---------------------------------------------------------------------
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(X, adj_t)                     # shape (n, n_class)
    loss = criterion(out[idx_train_t], y_t[idx_train_t])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# ---------------------------------------------------------------------
# 6) Evaluation on the test set
# ---------------------------------------------------------------------
model.eval()
with torch.no_grad():
    out = model(X, adj_t)
    preds = out[idx_test_t].max(1)[1]        # argmax over log-probabilities
    correct = preds.eq(y_t[idx_test_t]).sum().item()
    acc = correct / len(idx_test)
    print(f"GNN Karate test accuracy: {acc:.4f}")


# Task 12 comment:
# All node features are identical, so the GNN cannot distinguish nodes from input data.
# It must rely only on the graph structure, which is insufficient for good classification.
# As expected, accuracy drops sharply (≈ 0.2857).


