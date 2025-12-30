"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

############## Task 1

path='/Users/veroniquemohy-cardoso/Desktop/26 NY /MVA/ALTEGRAD/Lab4/datasets/CA-HepTh.txt'

#creating the unoriented graph : 

G=nx.read_edgelist(
    path,
    delimiter="\t",
    comments="#",
    nodetype=int,
    create_using=nx.Graph()
)

print("Number of edges:",G.number_of_edges())
print("Number of nodes:",G.number_of_nodes())

############## Task 2


components=list(nx.connected_components(G))

print("Number of connected components :",len(components))

largest_concomp=max(components,key=len)

G_largest=G.subgraph(largest_concomp).copy()

print("Number of edges for the largest connected component:",G_largest.number_of_edges())
print("Their fraction among the whole graph is:",G_largest.number_of_edges()/G.number_of_edges())
print("Number of nodes for the largest connected component:",G_largest.number_of_nodes())
print("Their fraction among the whole graph is:",G_largest.number_of_nodes()/G.number_of_nodes())


#END OF TASK1

