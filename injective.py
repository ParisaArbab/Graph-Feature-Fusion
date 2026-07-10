import torch
from torch_geometric.datasets import TUDataset
import math

data_name = 'REDDIT-BINARY'  # Change this to your dataset name
dataset = TUDataset(root='data/TUDataset', name=data_name)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


data = dataset[0].to(device)

print(f"Data: {data}")

from models.dln import CustomGNNLayer

dim_list = [30, 75, 300]
# dim_list = [300]

linear_res = []
non_linear_res = []
labels = ["Non_linear_mp/dim="+str(dim) for dim in dim_list]
# labels.append("Linear_mp/dim="+str(300))

for dim in dim_list:
    x = torch.ones((data.num_nodes, dim), dtype=torch.float32).to(device)
    h = torch.ones((data.num_nodes, dim), dtype=torch.float32).to(device)

    distinct_rows_matrix = torch.unique(x, dim=0).float()

    rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=1e-5)
    print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")

    rank_linear = [(0, rank_of_distinct_matrix.item())]
    rank_non_linear = [(0, rank_of_distinct_matrix.item())]
    distinct_node_feature = [distinct_rows_matrix.size(0)]
    distinct_node_feature_x = [0]

    for i in range(1, 7):
        rank_linear.append((i, rank_linear[-1][1]))
        rank_non_linear.append((i, rank_non_linear[-1][1]))
        # x_matrix = torch.unique(x, dim=0).float()
        # h_matrix = torch.unique(h, dim=0).float()
        
        distinct_node_feature.append(distinct_node_feature[-1])
        # distinct_node_feature.append(x_matrix.size(0))
        distinct_node_feature_x.append(i)
        # distinct_node_feature_x.append(i)
        gnn = CustomGNNLayer(epsilon=math.pi - 3, hidden_dim=dim, linear=True)
        gnn = gnn.to(device)
        x = gnn(x, data.edge_index, data.edge_attr)
        
        dln = CustomGNNLayer(epsilon=math.pi - 3, hidden_dim=dim, linear=False)
        dln = dln.to(device)
        h = dln(h, data.edge_index, data.edge_attr)
        # print(torch.linalg.svdvals(h))
        # print(torch.linalg.eigvals(h@h.T))
        x_matrix = torch.unique(x, dim=0).float()
        h_matrix = torch.unique(h, dim=0).float()
        
        # x_matrix = x.float()
        # h_matrix = h.float()
        print(f"H matrix: {h_matrix.size()}")
        rank_of_distinct_matrix_x = torch.linalg.matrix_rank(x_matrix, tol=1e-5)
        rank_of_distinct_matrix_h = torch.linalg.matrix_rank(h_matrix, tol=1e-5)
        print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
        rank_linear.append((i, min(rank_of_distinct_matrix_x.item(), h_matrix.size(0))))
        rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), h_matrix.size(0))))
        distinct_node_feature.append(h_matrix.size(0))
        distinct_node_feature_x.append(i)
    linear_res.append(rank_linear)
    non_linear_res.append(rank_non_linear)

print(f"Linear rank: {rank_linear}")
print(f"Non-linear rank: {rank_non_linear}")
print(f"Distinct node feature: {distinct_node_feature}")

res = non_linear_res
# res = []
# res.append(linear_res[-1])

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
# Get a nice color palette from Seaborn
palette = sns.color_palette("tab10", n_colors=len(res)+1)

fig, ax = plt.subplots(figsize=(10, 6))

data_rows = []
sns.set_theme(style="whitegrid") # Apply Seaborn styling
plt.figure(figsize=(10, 6)) # Set the figure size
plt.plot(distinct_node_feature_x, distinct_node_feature, linestyle='-', linewidth=1.5, color=palette[0], alpha=0.8, label="Num_distinct_node_feature")
for i, data in enumerate(res):
    series_label = labels[i]
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    # for x, y in dataset:
    plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+1], alpha=0.8, label=series_label)

    
        



# sns.lineplot(data=df, x='num_mp_layers', y='rank of distinct node feature matrix', hue='label', marker='o', linewidth=2.5, markersize=6, alpha=0.5, palette="viridis")

# --- 4. Customize ---
plt.title('')
plt.xlabel('Number of Message Passing Layers')
plt.ylabel('Distinct Features and Their Ranks')
plt.xlim(0, 6) # Set x-axis limits

plt.legend() # Add legend (Seaborn usually adds one automatically)
plt.tight_layout()
plt.show()
# Save the figure to pdf
plt.savefig('injective.pdf', format='pdf', bbox_inches='tight')
