"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        # ---- 1st padding message : Z0 = ReLU(Â X W0) ----
        z=torch.mm(adj,x_in)
        z=self.fc1(z)
        z=self.relu(z)

        z=self.dropout(z)

        # ---- 2e message passing : Z1 = ReLU(Â Z0 W1) ----
        z=torch.mm(adj,z)
        z=self.fc2(z)
        z=self.relu(z)

        # ---- fully connected layer. ----
        z=self.fc3(z)
        
        return F.log_softmax(z, dim=1)
