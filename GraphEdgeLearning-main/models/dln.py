import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops # add_self_loops might not be needed directly here
from torch_scatter import scatter_add # Used internally by MessagePassing with aggr='add'
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, BatchNorm, global_add_pool

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class CustomGNNLayer(MessagePassing):
    """
    A custom GNN layer implementing the formula:
    H_out = ((1 + epsilon) * I + A') * H
    where A' = D^(-1/2) * A * D^(-1/2) is the symmetrically normalized adjacency matrix,
    I is the identity matrix, H is the node feature matrix, and epsilon is a small scalar.
    This layer has no learnable parameters.
    """
    def __init__(self, epsilon: float, hidden_dim: int = 300, linear: bool = False):
        """
        Args:
            epsilon (float): The epsilon value in the layer formula.
        """
        # Initialize the MessagePassing base class with 'add' aggregation
        super().__init__(aggr='add')
        if not isinstance(epsilon, (float, int)):
             raise TypeError("epsilon must be a float or int")
        self.epsilon = float(epsilon) # Store epsilon as a float
        self.linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.batch_norm = BatchNorm(hidden_dim)
        self.lin = linear
        # Freeze the parameters of the linear layer and batch norm
        # if self.linear:
        # for param in self.linear.parameters():
        #     param.requires_grad = False
        # for param in self.batch_norm.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the custom GNN layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights corresponding to edge_index,
                                                 shape [num_edges]. If None, edges are assumed
                                                 to have weight 1. Defaults to None.

        Returns:
            torch.Tensor: Output node features of shape [num_nodes, num_node_features].
        """
        num_nodes = x.size(0)

        pre_h = x

        # --- Compute A' * H ---
        # This part calculates the contribution from the normalized adjacency matrix A'

        # 1. Get edge indices and potentially edge weights
        row, col = edge_index
        if edge_weight is None:
            # If no edge weights are provided, use uniform weights (1.0)
            edge_weight = torch.ones(edge_index.size(1), dtype=x.dtype, device=edge_index.device)
        else:
            # Ensure edge_weight has the correct shape and type
             edge_weight = edge_weight.view(-1)
             assert edge_weight.size(0) == edge_index.size(1)

        # 2. Calculate node degrees (based on incoming edges, column index `col`)
        # Use edge_weight for weighted degrees if available
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)

        # 3. Compute normalization: D^(-1/2)
        # Add a small epsilon to prevent division by zero for isolated nodes
        deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv = deg.pow(-1)
        # Replace 'inf' resulting from 0^(-0.5) with 0
        # deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # 4. Calculate normalization coefficients for each edge (i, j)
        # norm_ij = deg_inv_sqrt[i] * edge_weight_ij * deg_inv_sqrt[j]
        # We use row for i and col for j based on edge_index definition
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # norm = deg_inv[row] * edge_weight

        # 5. Propagate messages using the calculated normalization factors
        # self.propagate computes sum_{j \in N(i)} norm_ij * x_j
        # This effectively computes (D^(-1/2) * A * D^(-1/2)) * H
        # Note: We pass `x=x` which becomes `x_i` and `x_j` in the message/update steps
        # We pass `norm=norm` which will be used in the message function.
        # `size=num_nodes` ensures the output tensor has the correct size, important for graphs
        # where the max node index isn't the last node.
        # print(f"edge_index: {edge_index}")
        aggr_out = self.propagate(edge_index, x=x, norm=norm)

        # --- Combine the terms: (1 + epsilon) * H + A' * H ---
        # Add the scaled identity term (1 + epsilon) * H
        out = (1.0 + self.epsilon) * x + aggr_out

        if not self.lin:
            out = self.linear(out)
            out = self.batch_norm(out)

            # eps = 1e-6
            # norms = torch.linalg.norm(out, ord=2, dim=1, keepdim=True)
            # norms = norms + eps
            # out = out / norms

            # out = out + pre_h

            out = F.relu(out)


        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages from source nodes j to target nodes i.
        This function is called by `propagate`.

        Args:
            x_j (torch.Tensor): Features of source nodes j of shape [num_edges, num_node_features].
                                These are the features corresponding to the `col` indices in edge_index.
            norm (torch.Tensor): The normalization coefficient computed in `forward` for each edge,
                                 shape [num_edges].

        Returns:
            torch.Tensor: Messages to be aggregated, shape [num_edges, num_node_features].
                          Represents norm_ij * H_j.
        """
        # Apply the normalization to the features of the source node (j)
        # norm has shape [num_edges], x_j has shape [num_edges, num_features]
        # We need to reshape norm to [num_edges, 1] for broadcasting
        return norm.view(-1, 1) * x_j

class DLN_Net(nn.Module):
    """
    A Graph Neural Network model stacking multiple CustomGNNLayer layers.

    Args:
        num_layers (int): The number of CustomGNNLayer layers to stack.
        epsilon (float): The epsilon value to use for all layers.
        activation (nn.Module, optional): The activation function to apply between
                                         layers (e.g., nn.ReLU()). If None, no activation
                                         is applied. Defaults to None.
    """
    def __init__(self, num_layers: int, num_linear_layers: int, input_dim: int, mp_hidden_dim: int, fl_hidden_dim: int, output_dim:int, epsilon: float, dropout: float = 0.2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        self.num_layers = num_layers
        self.epsilon = epsilon
        self.dropout = dropout  

        self.linear = nn.Linear(input_dim, mp_hidden_dim)
        self.batch_norm = BatchNorm(mp_hidden_dim)
        self.layer_norm = nn.LayerNorm(mp_hidden_dim)
        self.linear2 = nn.Linear(mp_hidden_dim, fl_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CustomGNNLayer(epsilon=self.epsilon, hidden_dim=mp_hidden_dim))

        

        params = list(self.layers.parameters())
        print(f"Number of parameters in GNN layers: {sum(p.numel() for p in params)}")

        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_linear_layers):
            self.linear_layers.append(nn.Linear(fl_hidden_dim, fl_hidden_dim))
            self.batch_norms.append(BatchNorm(fl_hidden_dim))
            self.layer_norms.append(nn.LayerNorm(fl_hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout))
            self.activations.append(nn.ReLU())
        
        self.embedding_h = AtomEncoder(emb_dim=fl_hidden_dim)
        self.embedding_b = BondEncoder(emb_dim=fl_hidden_dim)
        self.classifier = nn.Linear(fl_hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the stacked GNN layers.

        Args:
            x (torch.Tensor): Initial node features [num_nodes, num_features].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights [num_edges]. Defaults to None.

        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        h = x # Start with initial features

        # Apply the embedding layers
        # h = self.embedding_h(h)

        h = self.linear(h)
        # h = self.batch_norm(h)
        # h = self.layer_norm(h)
        # h = torch.relu(h)

        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight=edge_weight) + h
        h = self.linear2(h)
        for i, linear_layer in enumerate(self.linear_layers):
            pre_h = h
            h = linear_layer(h)
            # h = self.batch_norms[i](h)
            # h = self.layer_norms[i](h)
            h = h + pre_h
            h = self.activations[i](h)
            h = self.dropouts[i](h)

        h = global_add_pool(h, batch) if batch is not None else h

        h = self.classifier(h)

        return h