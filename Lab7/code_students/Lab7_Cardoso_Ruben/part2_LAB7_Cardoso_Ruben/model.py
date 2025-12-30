"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Dec 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout=0.2):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.fc = nn.ModuleList()
        self.fc.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dim), 
                            nn.Dropout(dropout)
                            ))

        for i in range(1, n_layers):
            self.fc.append(nn.Sequential(nn.Linear(hidden_dim*i, hidden_dim*(i+1)),  
                            nn.ReLU(),
                            nn.LayerNorm(hidden_dim*(i+1)), 
                            nn.Dropout(dropout)
                            ))
        
        self.fc_proj = nn.Linear(hidden_dim*n_layers, n_nodes*n_nodes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    ############## Task 7
    def forward(self, x):
        # Pass latent vector through the MLP
        for layer in self.fc:
            x = layer(x)
        T_1 = x  # Output of the decoder MLP

        # Project to an n^2-dimensional vector
        T_2 = self.fc_proj(T_1)

        # Reshape into an n x n matrix
        T_2_reshape = torch.reshape(T_2, (-1,self.n_nodes, self.n_nodes))

        # Create symmetry for the adjacency matrix
        adj = 0.5 * (T_2_reshape + torch.transpose(T_2_reshape,1,2))
        
        return adj


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super(Encoder, self).__init__()        
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU()
                            ))

        for layer in range(n_layers-1):
            self.mlps.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU()
                            ))

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x, idx):
        n = adj.size(0)
        device = adj.device

        adjx=torch.mm(adj,x) # \bar A*X      

        # Message passing layer 1 : H^(1) = MLP_1( A X )
        h_1 = self.mlps[0](adjx)

        # Message passing layer 2 : H^(2) = MLP_2( A H^(1) )
        h_2 = self.mlps[1](torch.mm(adj, h_1))

        # Readout
        idx = idx.unsqueeze(1).repeat(1, h_2.size(1))
        out = torch.zeros(torch.max(idx)+1, h_2.size(1), device=x.device, requires_grad=False)
        out = out.scatter_add_(0, idx, h_2)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes) 

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu):
        adj = self.decoder(mu)
        adj=torch.sigmoid(adj)
        adj = adj * (1 - torch.eye(adj.size(-2), adj.size(-1), device=adj.device))
        return adj

    def loss_function(self, adj, x, idx, y, beta=0.05):
        x_g  = self.encoder(adj, x, idx)
        
        ############## Task 6
        
        mu=self.fc_mu(x_g)          # mu=x_G*W_mu + b_mu 
        
        logvar = self.fc_logvar(x_g) # log(sigma)=x_G*W_sigma + b_sigma
        
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        triu_idx = torch.triu_indices(self.n_max_nodes, self.n_max_nodes)
        recon = F.binary_cross_entropy_with_logits(adj[:,triu_idx[0,:],triu_idx[1,:]], y[:,triu_idx[0,:],triu_idx[1,:]], reduction='sum', pos_weight=torch.tensor(1./0.4))
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld
        