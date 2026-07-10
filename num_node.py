from ogb.graphproppred import PygGraphPropPredDataset

data_name = 'ogbg-ppa'

dataset = PygGraphPropPredDataset(name=data_name, root='dataset')

max_num_nodes = 0
min_num_nodes = float('inf')
num_nodes = []
for data in dataset:
    max_num_nodes = max(max_num_nodes, data.num_nodes)
    min_num_nodes = min(min_num_nodes, data.num_nodes)
    num_nodes.append(data.num_nodes)
print(f"Average number of nodes in the dataset: {sum(num_nodes) / len(num_nodes)}")
print(f"Median number of nodes in the dataset: {sorted(num_nodes)[len(num_nodes) // 2]}")
print(f"Maximum number of nodes in the dataset: {max_num_nodes}")
print(f"Minimum number of nodes in the dataset: {min_num_nodes}")