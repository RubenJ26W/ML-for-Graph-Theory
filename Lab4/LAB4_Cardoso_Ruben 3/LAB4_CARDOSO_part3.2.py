"""
ALTEGRAD LAB 4 PART 3.2 

"""

#IMPORTING LIBRARIES 

import numpy as np
import re
from grakel.utils import graph_from_networkx 
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from grakel.kernels import ShortestPath, PyramidMatch, RandomWalk, Propagation
import warnings
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')   

#####################################################################################


def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '/Users/veroniquemohy-cardoso/Desktop/26 NY /MVA/ALTEGRAD/Lab4/datasets/train_5500_coarse.label'
path_to_test_set = '/Users/veroniquemohy-cardoso/Desktop/26 NY /MVA/ALTEGRAD/Lab4/datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()
        
        # define the window
        window_word_id_list = [ None for _ in range(window_size)] 
        for word in doc:
            
            # check if the word is in the vocabulary
            if word in vocab:
                word_id = vocab[word]

                # create node if it does not exist
                if not G.has_node(word_id):
                    G.add_node(word_id,label=str(word_id))
                
                # create edges to words in the sliding window
                for k in range(window_size):
                    neighbor_word_id = window_word_id_list[k]
                    if neighbor_word_id is not None:
                        G.add_edge(word_id, neighbor_word_id, label="0")
            else:
                word_id = None

            # update the sliding window
            window_word_id_list = window_word_id_list[1:] + [word_id]
        
        graphs.append(G)
    
    return graphs



# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


# Task 12

# Transform networkx graphs to grakel representations
G_train = graph_from_networkx(G_train_nx, node_labels_tag='label')
G_test  = graph_from_networkx(G_test_nx,  node_labels_tag='label')

# Initialize a Weisfeiler–Lehman subtree kernel
gk = WeisfeilerLehman(
    normalize=False,
    n_iter=1,
    base_graph_kernel=VertexHistogram
)

# Construct kernel matrices
K_train = gk.fit_transform(G_train)   # (n_train x n_train)
K_test  = gk.transform(G_test)       # (n_test  x n_train)

#Task 13

# Train an SVM classifier and make predictions
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm=SVC(kernel='precomputed')
svm.fit(K_train,y_train)
y_pred=svm.predict(K_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))

# Task 14 : 

# Helper function to rebuild the GraKeL graphs
def rebuild():
    G_train = graph_from_networkx(G_train_nx, node_labels_tag='label')
    G_test  = graph_from_networkx(G_test_nx,  node_labels_tag='label')
    return G_train, G_test

# Helper function to compute accuracy for a given kernel
def run_kernel(kernel_obj, name):
    print(f"\n=== {name} ===")
    G_train, G_test = rebuild()

    K_train = kernel_obj.fit_transform(G_train)
    K_test  = kernel_obj.transform(G_test)

    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    print("accuracy =", acc)
    return acc


accuracies = {}

#1) Pyramid match Kernel

gk_pm = PyramidMatch(L=2, normalize=True)
acc_pm = run_kernel(gk_pm, "PyramidMatch")
accuracies["PyramidMatch"] = acc_pm


# 2) Random Walk kernel
from grakel.kernels import RandomWalk

gk_rw = RandomWalk(lamda=0.01,normalize=True)
acc_rw = run_kernel(gk_rw, "RandomWalk")
accuracies["RandomWalk"] = acc_rw


# 3) Weisfeiler-Lehman with more iterations (n_iter = 3)
gk_wl3 = WeisfeilerLehman(
    n_iter=3,
    normalize=False,
    base_graph_kernel=VertexHistogram
)
accuracies["WL (n_iter=3)"] = run_kernel(gk_wl3, "WL (n_iter = 3)") #Less than the previous accuracy with n_iter=1


# Summary
print("\n=== Summary ===")
for k, v in accuracies.items():
    print(f"{k:20s} : {v:.4f}")

    