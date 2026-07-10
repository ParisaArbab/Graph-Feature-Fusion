import torch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)  # Input layer
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)  # Hidden layer
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Graph-level classification

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector mapping nodes to graphs [num_nodes]
            edge_weight: Edge weights [num_edges] (optional)
        """
        # If edge weights are not provided, initialize to 1.0 (float)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=x.device)

        # Pass through GCN layers
        x = self.gcn1(x, edge_index, edge_weight).relu()  # First GCN layer
        x = self.gcn2(x, edge_index, edge_weight).relu()  # Second GCN layer

        # Global Pooling: Aggregate node embeddings to graph embeddings
        graph_embedding = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        # Graph-level classification
        graph_output = self.classifier(graph_embedding)  # [num_graphs, output_dim]
        return graph_output
    
class GCN_Net(torch.nn.Module):
    def __init__(self, input_channels, hid_channels, out_channels, num_layers, dropout):
        super(GCN_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.linear = torch.nn.Linear(input_channels, hid_channels)
        self.batch_norm = BatchNorm(hid_channels)
        for _ in range(self.num_layers):
            self.layers.append(GCNConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        # x = self.embedding_h(x)
        x = self.linear(x)
        # x = self.batch_norm(x)
        
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