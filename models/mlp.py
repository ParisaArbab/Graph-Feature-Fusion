import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from ogb.graphproppred.mol_encoder import AtomEncoder
from models.customconv import CustomConv

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        # self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Graph-level output

    def forward(self, x, batch):
        x = self.embedding_h(x)

        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = x_h + x
            x = F.relu(x)
            # x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
    
class MLP_Concat(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.embedding_a = nn.Linear(input_dim, hidden_dim)
        self.embedding_a_power = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Graph-level output

    def forward(self, x, adj_list, batch):
        x = self.embedding_h(x)

        for i in range(len(self.layers)):
            # x = x + self.embedding_a(adj_list[i])
            x_h = x
            # x_h = x + self.embedding_a(adj_list[i])
            x = self.layers[i](x)
            if i == 0:
                x_0 = x
            x = self.batch_norms[i](x)
            x = x_h + x

            x = F.relu(x)
            # x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

class MLP_Custom(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(CustomConv(hidden_dim, hidden_dim, dropout=dropout))
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.embedding_a = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Graph-level output

    def forward(self, x, edge_index, adj_list, batch):
        x = self.embedding_h(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index=edge_index)
            
        # x = torch.relu(x)
            
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
