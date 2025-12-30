"""
Graph Mining - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7 


#load Mutag dataset
def load_dataset():

    # Load MUTAG from PyTorch Geometric
    dataset = TUDataset(root="./data", name="MUTAG")

    # Convert each PyG graph into a NetworkX graph
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]

    # Extract labels
    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test

K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 8


# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)


    phi_train = np.zeros((len(Gs_train), 4))

    for i, G in enumerate(Gs_train):
        nodes = list(G.nodes())
        if len(nodes) < 3:
            continue

        for _ in range(n_samples):
            # sample 3 random nodes
            sampled = np.random.choice(nodes, size=3, replace=False)
            # induced sub graph
            H = G.subgraph(sampled).copy()

            # isomorphic testing.
            for j, glet in enumerate(graphlets):
                if nx.is_isomorphic(H, glet):
                    phi_train[i, j] += 1
                    break
    
    phi_test = np.zeros((len(Gs_test), 4))

    for i, G in enumerate(Gs_test):
        nodes = list(G.nodes())
        if len(nodes) < 3:
            continue
        
        for _ in range(n_samples):
            sampled = np.random.choice(nodes, size=3, replace=False)
            H = G.subgraph(sampled).copy()

            for j, glet in enumerate(graphlets):
                if nx.is_isomorphic(H, glet):
                    phi_test[i, j] += 1
                    break

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 9

# Compute kernel matrices for the graphlet kernel
K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test, n_samples=200)

#BONUS : Printing graphlet kernels informations.
print("Graphlet kernel (train) shape:", K_train_gk.shape)
print("Graphlet kernel (test)  shape:", K_test_gk.shape)
print("First 5x5 block of K_train_gk:")
print(K_train_gk[:5, :5])


############## Task 10

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- SVM with shortest path kernel ---
svm_sp = SVC(kernel='precomputed')
svm_sp.fit(K_train_sp, y_train)
y_pred_sp = svm_sp.predict(K_test_sp)
acc_sp = accuracy_score(y_test, y_pred_sp)

# --- SVM with graphlet kernel ---
svm_gk = SVC(kernel='precomputed')
svm_gk.fit(K_train_gk, y_train)
y_pred_gk = svm_gk.predict(K_test_gk)
acc_gk = accuracy_score(y_test, y_pred_gk)

# --- Summary ---
print("=== SVM classification with graph kernels ===")
print(f"Shortest path kernel accuracy : {acc_sp:.4f}")
print(f"Graphlet kernel accuracy      : {acc_gk:.4f}")

if acc_sp > acc_gk:
    print(f"\nThe shortest path kernel performs better "
          f"by {acc_sp - acc_gk:.4f} accuracy points.")
elif acc_gk > acc_sp:
    print(f"\nThe graphlet kernel performs better "
          f"by {acc_gk - acc_sp:.4f} accuracy points.")
else:
    print("\nBoth kernels achieve exactly the same accuracy on this split.")





