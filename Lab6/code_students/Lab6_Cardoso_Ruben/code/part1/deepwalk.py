"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    
    walk = [node]                 # start from the initial node
    current = node

    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current))
        if len(neighbors) == 0:   # no more neighbors : stop.
            break
        current = random.choice(neighbors)
        walk.append(current)

    # Convert nodes to strings.
    walk = [str(n) for n in walk]

    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes=list(G.nodes())
    for _ in range(walk_length):
        for node in nodes :
            walks.append(random_walk(G,node,walk_length))
    
    random.shuffle(walks)
    permuted_walks=np.array(walks,dtype=object)

    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
