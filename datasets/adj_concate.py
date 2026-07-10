import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

class AdjacencyConcateDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        self.max_num_nodes = max([graph.num_nodes for graph in graphs])
        self.max_num_features = max([graph.num_node_features for graph in graphs])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        # Get dense adjacency matrix and pad it to [max_num_nodes, max_num_nodes]
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(0))[0]  # Shape: [max_num_nodes, max_num_nodes]
        adj_power_2 = adj * adj
        adj_power_3 = adj * adj * adj
        adj_power_4 = adj * adj * adj * adj
        adj_power_5 = adj * adj * adj * adj * adj

        # Compute degree matrix D
        degree = adj.sum(dim=1)  # Sum over columns to get degree of each node
        # D_inv = 1.0 / (degree + 1e-6)  # Avoid division by zero
        # D_inv_sqrt = 1.0 / torch.sqrt(degree + 1e-6) # Avoid division by zero
        D_inv_sqrt = torch.pow(degree, -0.5)
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)

        degree_power_2 = adj_power_2.sum(dim=1)
        # D_power_inv_2 = 1.0 / (degree_power_2 + 1e-6)
        # D_power_inv_sqrt_2 = 1.0 / torch.sqrt(degree_power_2 + 1e-6)
        D_power_inv_sqrt_2 = torch.pow(degree_power_2, -0.5)
        D_mat_power_inv_sqrt_2 = torch.diag(D_power_inv_sqrt_2)


        degree_power_3 = adj_power_3.sum(dim=1)
        # D_power_inv_3 = 1.0 / (degree_power_3 + 1e-6)
        # D_power_inv_sqrt_3 = 1.0 / torch.sqrt(degree_power_3 + 1e-6)
        D_power_inv_sqrt_3 = torch.pow(degree_power_3, -0.5)
        D_mat_power_inv_sqrt_3 = torch.diag(D_power_inv_sqrt_3)

        degree_power_4 = adj_power_4.sum(dim=1)
        # D_power_inv_4 = 1.0 / (degree_power_4 + 1e-6)
        # D_power_inv_sqrt_4 = 1.0 / torch.sqrt(degree_power_4 + 1e-6)
        D_power_inv_sqrt_4 = torch.pow(degree_power_4, -0.5)
        D_mat_power_inv_sqrt_4 = torch.diag(D_power_inv_sqrt_4)

        degree_power_5 = adj_power_5.sum(dim=1)
        # D_power_inv_5 = 1.0 / (degree_power_5 + 1e-6)
        # D_power_inv_sqrt_5 = 1.0 / torch.sqrt(degree_power_5 + 1e-6)
        D_power_inv_sqrt_5 = torch.pow(degree_power_5, -0.5)
        D_mat_power_inv_sqrt_5 = torch.diag(D_power_inv_sqrt_5)

        # Compute normalized_adj
        # adj = adj * D_inv.view(1, -1)  # AD^{-1}, broadcast across cols
        # adj = adj * D_inv.view(-1, 1)  #  D^{-1}A, broadcast across rows
        # adj = adj * D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(1, -1) # D^{-1/2} A D^{-1/2}\

        
        # adj = D_inv_sqrt.view(-1, 1) * adj * D_inv_sqrt.view(-1, 1)

        # adj_power_2 = D_power_inv_sqrt_2.view(-1, 1) * adj_power_2 * D_power_inv_sqrt_2.view(-1, 1)

        # adj_power_3 = D_power_inv_sqrt_3.view(-1, 1) * adj_power_3 * D_power_inv_sqrt_3.view(-1, 1)

        # adj_power_4 = D_power_inv_sqrt_4.view(-1, 1) * adj_power_4 * D_power_inv_sqrt_4.view(-1, 1)

        # adj_power_5 = D_power_inv_sqrt_5.view(-1, 1) * adj_power_5 * D_power_inv_sqrt_5.view(-1, 1)

        adj = torch.mm(torch.mm(D_mat_inv_sqrt, adj), D_mat_inv_sqrt)
        adj_power_2 = torch.mm(torch.mm(D_mat_power_inv_sqrt_2, adj_power_2), D_mat_power_inv_sqrt_2)
        adj_power_3 = torch.mm(torch.mm(D_mat_power_inv_sqrt_3, adj_power_3), D_mat_power_inv_sqrt_3)
        adj_power_4 = torch.mm(torch.mm(D_mat_power_inv_sqrt_4, adj_power_4), D_mat_power_inv_sqrt_4)
        adj_power_5 = torch.mm(torch.mm(D_mat_power_inv_sqrt_5, adj_power_5), D_mat_power_inv_sqrt_5)



        # Pad node features to [max_num_nodes, max_num_features]
        num_nodes, num_features = graph.x.size()
        padded_adj = torch.zeros(num_nodes, self.max_num_nodes)  # Shape: [max_num_nodes, max_num_features]
        padded_adj[:, :num_nodes] = adj  # Copy node features into padded tensor
        
        padded_adj_power_2 = torch.zeros(num_nodes, self.max_num_nodes)
        padded_adj_power_2[:, :num_nodes] = adj_power_2

        padded_adj_power_3 = torch.zeros(num_nodes, self.max_num_nodes)
        padded_adj_power_3[:, :num_nodes] = adj_power_3

        padded_adj_power_4 = torch.zeros(num_nodes, self.max_num_nodes)
        padded_adj_power_4[:, :num_nodes] = adj_power_4

        padded_adj_power_5 = torch.zeros(num_nodes, self.max_num_nodes)
        padded_adj_power_5[:, :num_nodes] = adj_power_5

        adj_list = [padded_adj, padded_adj_power_2, padded_adj_power_3, padded_adj_power_4, padded_adj_power_5]

        # return concated_x, graph.y
        return Data(x=graph.x, edge_index=graph.edge_index, adj_list=adj_list, y=graph.y)