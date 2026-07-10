import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GraphAsBagDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        self.max_num_nodes = max([graph.num_nodes for graph in graphs])
        self.max_num_features = max([graph.num_node_features for graph in graphs])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        num_nodes, num_features = graph.x.size()
        padded_x = torch.zeros(self.max_num_nodes, self.max_num_features) # [max_num_nodes, max_num_features]
        padded_x[:num_nodes, :num_features] = graph.x

        # return padded_x, graph.y 
        # return Data(x=padded_x, y=graph.y)
        return self.graphs[idx]