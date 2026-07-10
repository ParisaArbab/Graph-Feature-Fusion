import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

class GINE_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GINE_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(), nn.Linear(hid_channels, hid_channels))))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        self.embedding_b = BondEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_h(x)
        e = self.embedding_b(edge_attr)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index, e)
            x = self.batch_norms[i](x)
            x = x_h + x
            x = F.relu(x)
            # x = F.dropout(x, self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x