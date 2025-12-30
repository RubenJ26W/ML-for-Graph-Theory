"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        #Build all pairs of nodes connected by an edge. 
        z=self.fc(x)
        adj=adj.coalesce()
        indices=adj.indices()  # size = [2,m]
        z_i=z[indices[0,:],:] #size = [m,n_hidden]
        z_j=z[indices[1,:],:] #size = [m,n_hidden]

        #concatenation and LeakyRelu
        a_input=torch.cat([z_i,z_j],dim=1) # [m, 2*n_hidden]
        h=self.leakyrelu(self.a(a_input)) #[m,1]


        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        
        #Message passing : 
        out=torch.sparse.mm(adj_att,z)

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        
        #First Message passing layer : ReLU + Dropout.
        z1, _=self.mp1(x,adj)
        z1=self.relu(z1)
        z1=self.dropout(z1)

        #Second message passing layer

        z2, alpha=self.mp2(z1,adj)
        z2=self.relu(z2)

        #Fully connected layer : 

        logits=self.fc(z2)

        return F.log_softmax(logits, dim=1), alpha
