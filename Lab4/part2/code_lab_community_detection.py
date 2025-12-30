"""
Graph Mining - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    #1 : Defining nodes of the graph G. 
    nodes=list(G.nodes())
    n=len(nodes)

    #2: Creating the adjacency matrix A.
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float)

    #3: Creating D.
    D=np.array(A.sum(axis=1)).flatten() #Sum by row as it is defined. 
    D[D==0]=1 #to prevent from dividing by 0 when inverting D. 
    D_inv=diags(1/D)
    L_rw=eye(n)-D_inv@A  #I_n-D^(-1).A.

    #4: Eigenvalue decomposition to L_rw, finding the d=k smallest eigenvalues and eigenvectors associated.
    eigvals,eigvecs=eigs(L_rw,k=k,which='SR')
    U=np.real(eigvecs) #matrix of size nxk., of the eigen vectors.

    #5 : Kmeans over the lines of U.
    kmeans=KMeans(n_clusters=k,n_init=10,random_state=0)
    labels=kmeans.fit_predict(U)

    #6 : Creating the dictionnary : 
    clustering={nodes[i]:int(labels[i]) for i in range(n)}

    return clustering


############## Task 4

# Path to CA-HepTh dataset
path = "/Users/veroniquemohy-cardoso/Desktop/26 NY /MVA/ALTEGRAD/Lab4/datasets/CA-HepTh.txt"

# Load graph from edge list
G = nx.read_edgelist(
    path,
    delimiter="\t",
    comments="#",
    nodetype=int,
    create_using=nx.Graph()
)

# Find connected components
components = list(nx.connected_components(G))
N_conncomp = len(components)

# Giant connected component (largest one)
largest_concomp = max(components, key=len)

# Induced subgraph on the giant connected component
G_giant = G.subgraph(largest_concomp).copy()

# Apply Spectral Clustering on the giant component with k = 50 clusters
k = 50
clustering_giant = spectral_clustering(G_giant, k=k)

# (Optionnel) petit résumé
print(f"Number of connected components in G: {N_conncomp}")
print(f"Size of giant component: {G_giant.number_of_nodes()} nodes")
print(f"Number of clusters requested: {k}")
print("Example of clustering mapping (first 10 nodes):")
print(list(clustering_giant.items())[:10])



############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    m=G.number_of_edges()
    #extreme case : 
    if m==0:
        return 0
    # group nodes by cluster ID:
    communities = {}
    for node, cluster_id in clustering.items():
        if cluster_id not in communities:
            communities[cluster_id] = []
        communities[cluster_id].append(node)
    modularity=0.0
    for nodes_c in communities.values():
        nodes_c = set(nodes_c)

        # l_c: number of edges fully inside the community
        l_c = G.subgraph(nodes_c).number_of_edges()

        # d_c: total degree of nodes in this community (in the full graph)
        d_c = sum(dict(G.degree(nodes_c)).values())

        # modularity contribution for this community
        modularity += (l_c / m) - (d_c / (2.0 * m))**2

    return modularity

############## Task 6

from random import randint 

def random_clustering(G, k):
    """Assign each node of G to a random cluster in {1, ..., k}."""
    clustering = {}
    for node in G.nodes():
        clustering[node] = randint(1, k)
    return clustering

spec_mod = modularity(G_giant, clustering_giant)
rand_clust = random_clustering(G_giant, k=50)
rand_mod = modularity(G_giant, rand_clust)

print("====== MODULARITY RESULTS ======")
print(f"Spectral clustering:       {spec_mod:.4f}")
print(f"Random clustering (k=50):  {rand_mod:.4f}")
print("================================")
print()

if spec_mod > rand_mod:
    print("→ The spectral clustering is significantly better than a random partition.")
else:
    print("→ The spectral clustering does not improve over randomness.")








