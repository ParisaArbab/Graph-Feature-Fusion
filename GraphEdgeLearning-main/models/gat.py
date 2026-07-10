import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool, BatchNorm
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

class GAT_Net(torch.nn.Module):
    def __init__(self, input_channels, hid_channels, out_channels, num_layers, dropout):
        super(GAT_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.linear = torch.nn.Linear(input_channels, hid_channels)
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GATConv(hid_channels, hid_channels, heads=4, concat=False, dropout=dropout))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        # self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.linear(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = x_h + x
            x = F.relu(x)
            # x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_add_pool(x, batch)
        x = self.mlp(x)

        return x