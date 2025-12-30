"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib 


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

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


############## Task 5
colors=[]
for i in y:
    if i==0:
        colors.append("green") 
    else :
        colors.append("red")
    
pos=nx.spring_layout(G,seed=42)

#Draw the Graph:
plt.figure(figsize=(10,8))
nx.draw_networkx(G,pos=pos,with_labels=True,node_color=colors,node_size=600,font_size=10)

plt.title("Karate Club Network")
plt.axis("off")
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G,n_walks,walk_length,n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

from sklearn.linear_model import LogisticRegression

f=LogisticRegression()

f.fit(X_train,y_train)

y_pred=f.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)


############## Task 8
# Generates spectral embeddings

# 1) Nodes and adjacency matrix (same ordering as in DeepWalk embeddings)
nodes = list(G.nodes())
A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float)

# 2) Degree matrix and L_rw = I - D^{-1}A
D = np.array(A.sum(axis=1)).flatten()
D[D == 0] = 1          # avoid division by zero
D_inv = diags(1 / D)
L_rw = eye(n) - D_inv @ A

# 3) Two eigenvectors corresponding to the smallest eigenvalues
eigvals, eigvecs = eigs(L_rw, k=2, which='SR')
U = np.real(eigvecs)                 # shape (n, 2)

spectral_embeddings = U             # these are our 2D features

# 4) Same train/test split as for DeepWalk
X_train_spec = spectral_embeddings[idx_train, :]
X_test_spec  = spectral_embeddings[idx_test, :]

f_spec = LogisticRegression()
f_spec.fit(X_train_spec, y_train)
y_pred_spec = f_spec.predict(X_test_spec)

acc_deepwalk = accuracy_score(y_test, y_pred)       # from Task 7
acc_spectral = accuracy_score(y_test, y_pred_spec)

print("Accuracy (DeepWalk): ", acc_deepwalk)
print("Accuracy (Spectral):", acc_spectral)





