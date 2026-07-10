import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree, add_self_loops

class CustomConv(MessagePassing):
    r"""Custom graph convolutional layer that updates node features using the formula:
    :math:`\mathbf{x}^{\prime} = (1 - \alpha) (\mathbf{D}^{-1} \mathbf{A}) \mathbf{x} + \alpha \mathbf{W} \phi(\mathbf{x})`

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        alpha (float): Balancing factor between the convolved and transformed features.
        dropout (float): Dropout probability for regularization. (default: :obj:`0.0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lin = torch.nn.Linear(out_channels, out_channels, bias=bias)
        self.input_transform = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        # self.bn = BatchNorm(out_channels)
        # self.alpha = torch.nn.Parameter(torch.randn(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the learnable parameters of the layer."""
        self.lin.reset_parameters()
        self.input_transform.reset_parameters()
        self.bn.reset_parameters()
        # torch.nn.init.uniform_(self.alpha, 0, 1)
        # Initial self.alpha to 0.1
        # torch.nn.init.constant_(self.alpha, 0.1)

    def forward(self, g: Tensor, edge_index: Adj) -> Tensor:
        x = torch.relu(g)
        num_nodes = x.size(0)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=norm, sparse_sizes=(num_nodes, num_nodes))

        # alpha = torch.sigmoid(self.alpha)
        alpha = 0.1

        # Perform the convolution operation: (1 - alpha) * (D^-1/2 * A * D^-1/2) * x
        x_conv = matmul(adj, g)
        x_conv = (1 - alpha) * x_conv

        # Perform the linear transformation and activation function: alpha * W * x
        x_lin = self.lin(x)
        x_lin = alpha * x_lin
        

        # Combine the convolved and transformed features
        out = x_conv + x_lin
        out = self.bn(out)
        # out = self.dropout(out)

        return out
    

# import torch
# import torch.nn.functional as F
# from torch.nn import Linear, Parameter
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import degree, add_self_loops
# from torch_sparse import SparseTensor

# class CustomConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, alpha=0.1, dropout: float = 0.0, bias: bool=True, **kwargs):
#         super().__init__(aggr='add', **kwargs)  # Aggregation method (summation)
#         self.lin = Linear(in_channels, out_channels, bias=bias) # Learnable weight matrix W
#         # self.alpha = torch.nn.Parameter(torch.randn(1)) # Learnable parameter alpha
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         # torch.nn.init.constant_(self.alpha, 0.1) # Initialize alpha (you can change this)
#         # torch.nn.init.uniform_(self.alpha, 0, 1)

#     def forward(self, g, edge_index, adj_list: list):
#         # x: [N, in_channels] - Node features
#         # edge_index: [2, E] - Graph connectivity

#         x = torch.relu(g)
#         # alpha = torch.sigmoid(self.alpha)
#         alpha = 0.1

#         # 1. Transform node features with W (XkW part will be done later)
#         x_lin = self.lin(x) # This will be used for alpha XkW later

#         # 2. Calculate propagation matrix P = D^-1/2 (A+I) D^-1/2
#         edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=g.size(0))
#         row, col = edge_index_with_loops

#         deg = degree(col, x.size(0)) # Degree based on edge_index_with_loops
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # 3. Message Passing (PGk part) - using propagate
#         # print(edge_index.shape, norm.size(), g.size(), x.size()) # Debug print
#         out = self.propagate(edge_index_with_loops, x=g, norm=norm) # Pass norm to message function

#         # 4. Combine message passing and feature transformation: G(k+1) = (1-a)PGk + alpha XkW
#         out = (1 - alpha) * out + alpha * x_lin # (1-alpha) * PGk + alpha * XkW

#         # 5. Apply ReLU activation: Xk = relu(Gk) - where here 'out' is G(k+1) and becomes input for next layer, so we apply ReLU to get Xk (layer output)
#         # return F.relu(out)
#         return out


#     def message(self, x_j, norm):
#         # x_j: Features of neighboring nodes [E, in_channels]
#         # norm: Normalization factor for edges [E]
#         # print("Shape of norm in message:", norm.shape) # Debug print
#         # print("Shape of x_j in message:", x_j.shape) # Debug print
#         return norm.view(-1, 1) * x_j # Multiply neighbor features by normalization factor

#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         # 'add' aggregation is set in __init__, so this is handled automatically by MessagePassing
#         return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size) # Summation aggregation