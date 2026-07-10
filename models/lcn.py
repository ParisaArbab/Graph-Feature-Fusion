import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.utils import to_dense_adj, add_self_loops, degree
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch_sparse import SparseTensor

# class LaplacianConv(nn.Module):
#     def __init__(self, in_channels, out_channels, alpha, eta, bias=True):
#         super(LaplacianConv, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels, bias=bias)
#         self.alpha = alpha
#         self.eta = eta
#         self.bn = BatchNorm(out_channels)

#     def forward(self, g, edge_index, edge_weight=None):
#         """
#         Forward pass of the simplified LaplacianConv layer using pure PyTorch.

#         Args:
#             g (torch.Tensor): Input node features (G^(l-1)).
#             edge_index (torch.LongTensor): Edge indices.
#             edge_weight (torch.Tensor, optional): Edge weights.  If None,
#                 treat as an unweighted graph.

#         Returns:
#             torch.Tensor: Output node features (X^l).
#         """

#         # 1. Calculate X^(l-1)
#         x = g.relu()

#         # 2. Calculate Laplacian (L)
#         num_nodes = g.size(0)

#         if edge_weight is None:
#             edge_weight = torch.ones(edge_index.size(1), device=g.device)

#         # Add self-loops (important for the Laplacian calculation)
#         edge_index_with_loops, edge_weight_with_loops = add_self_loops(
#             edge_index, edge_weight, num_nodes=num_nodes
#         )

#         # Calculate the degree matrix (D)
#         row, col = edge_index_with_loops
#         deg = degree(col, num_nodes, dtype=g.dtype)

#         # Calculate D^(-1/2)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle 0 degrees

#         # Normalize edge weights: D^(-1/2) * A * D^(-1/2)
#         norm_edge_weight = deg_inv_sqrt[row] * edge_weight_with_loops * deg_inv_sqrt[col]

#         # Convert to a dense adjacency matrix (for easier matrix multiplication)
#         adj_matrix = to_dense_adj(edge_index_with_loops, edge_attr=norm_edge_weight, max_num_nodes=num_nodes).squeeze(0)

#         # L = I - D^(-1/2) * A * D^(-1/2)
#         laplacian = torch.eye(num_nodes, device=g.device) - adj_matrix

#         # 3. Calculate (I - eta * L) * G^(l-1)
#         laplacian_term = torch.matmul(laplacian, g)  # L * G^(l-1)
#         g = g - self.eta * laplacian_term

#         # 4. Calculate alpha * X^(l-1) * W^l
#         x_lin = self.alpha * self.linear(x)

#         # 5. Combine: G^l = (I - eta * L) * G^(l-1) + alpha * X^(l-1) * W^l
#         g = g + x_lin

#         # 6. Apply batch normalization
#         g = self.bn(g)

#         return g

#     def reset_parameters(self):
#         self.linear.reset_parameters()

class LaplacianConvSparse(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, eta, bias=True):
        super(LaplacianConvSparse, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.alpha = alpha
        self.eta = eta
        self.bn = BatchNorm(out_channels)

    def forward(self, g, edge_index, adj_emb, edge_weight=None):
        """
        Forward pass of the simplified LaplacianConv layer using torch_sparse.

        Args:
            g (torch.Tensor): Input node features (G^(l-1)).
            edge_index (torch.LongTensor): Edge indices.
            edge_weight (torch.Tensor, optional): Edge weights.  If None,
                treat as an unweighted graph.

        Returns:
            torch.Tensor: Output node features (X^l).
        """

        # 1. Calculate X^(l-1)
        x = g.relu()
        x = x + adj_emb

        # 2. Calculate Laplacian (L) - using SparseTensor
        num_nodes = g.size(0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=g.device)

        # Add self-loops
        edge_index_with_loops, edge_weight_with_loops = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )

        # Calculate the degree matrix (D)
        row, col = edge_index_with_loops
        deg = degree(col, num_nodes, dtype=g.dtype)

        # Calculate D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle 0 degrees

        # Normalize edge weights: D^(-1/2) * A * D^(-1/2)
        norm_edge_weight = deg_inv_sqrt[row] * edge_weight_with_loops * deg_inv_sqrt[col]


        # Create a SparseTensor
        adj_t = SparseTensor(row=col, col=row, value=norm_edge_weight, sparse_sizes=(num_nodes, num_nodes))

        # L = I - D^(-1/2) * A * D^(-1/2)
        # Efficiently calculate (I - eta * L) * G  directly.  This avoids
        # explicitly forming the full Laplacian matrix.  We use the property:
        # (I - eta * L) * G = (I - eta * (I - D^-1/2 * A * D^-1/2)) * G
        #                   = (I - eta * I + eta * D^-1/2 * A * D^-1/2) * G
        #                   = ((1 - eta) * I + eta * D^-1/2 * A * D^-1/2) * G
        #                   = (1 - eta) * G + eta * (D^-1/2 * A * D^-1/2) * G
        # The adj_t represents D^-1/2 * A * D^-1/2
        laplacian_term = self.eta * adj_t.matmul(g) # eta * (D^-1/2 * A * D^-1/2) * G
        g = (1 - self.eta) * g + laplacian_term

        # 4. Calculate alpha * X^(l-1) * W^l
        x_lin = self.alpha * self.linear(x)

        # 5. Combine: G^l = (I - eta * L) * G^(l-1) + alpha * X^(l-1) * W^l
        g = g + x_lin

        # 6. Apply batch normalization
        g = self.bn(g)

        return g

class LCN_Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(LaplacianConvSparse(hidden_dim, hidden_dim, alpha=1, eta=1, bias=True))
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.embedding_a = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Graph-level output

    def forward(self, x, edge_index, adj_list, batch):
        x = self.embedding_h(x)

        for i in range(len(self.layers)):
            adj_emb = self.embedding_a(adj_list[i])
            x = self.layers[i](x, edge_index=edge_index, adj_emb=adj_emb)
            
        # x = torch.relu(x)
            
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
