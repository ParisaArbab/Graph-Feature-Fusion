import torch
from torch_geometric.datasets import TUDataset
import math
import numpy as np
from tqdm import tqdm

data_name = 'REDDIT-BINARY'  # Change this to your dataset name
dataset = TUDataset(root='data/TUDataset', name=data_name)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def cal_mu(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / row_norms  # Avoid division by zero (if a row is all zeros)
    
    # Compute mean of normalized rows
    xbar_normalized = np.mean(X_normalized, axis=0)
    
    # Center the normalized matrix
    X_centered = X_normalized - xbar_normalized
    
    # Compute MSE
    mse = np.mean(X_centered ** 2)

    if mse == 0:
        return 1e-5
    
    return mse.item()

data = dataset[0].to(device)

print(f"Data: {data}")

from models.dln import CustomGNNLayer

# dim_list = [30, 75, 300]
layer_list = [i for i in range(1, 100)]
dim_list = [30, 75, 300]
n_sample = 50

all_linear_res = []
all_non_linear_res = []
labels_x = ["Linear_mp_layer="+str(layer) for layer in layer_list]
labels_h = ["Non_linear_mp_layer="+str(layer) for layer in layer_list]

for dim in tqdm(dim_list):
    linear_res = [0]*101
    non_linear_res = [0]*101
    linear_res[0] = 1e-5
    non_linear_res[0] = 1e-5
    for data in dataset[:n_sample]:

        x = torch.ones((data.num_nodes, dim), dtype=torch.float32).to(device)
        h = torch.ones((data.num_nodes, dim), dtype=torch.float32).to(device)

        # def cal_mu(input):
        #     if not isinstance(input, np.ndarray):
        #         input = np.array(input)
        #     input_bar = np.mean(input, axis=0)
        #     # print("======================================")
        #     # print(input_bar)
        #     # print(input)
        #     # print(stop)
        #     input_norm = np.linalg.norm(input_bar)
        #     return (np.mean((input - input_bar)**2) / input_norm).item()


        for layer in layer_list:
            gnn = CustomGNNLayer(epsilon=math.pi - 3, hidden_dim=dim, linear=True)
            gnn = gnn.to(device)
            x = gnn(x, data.edge_index)
            
            dln = CustomGNNLayer(epsilon=math.pi - 3, hidden_dim=dim, linear=False)
            dln = dln.to(device)
            h = dln(h, data.edge_index)
            linear_res[layer+1] += math.log10(cal_mu(x))
            non_linear_res[layer+1] += math.log10(cal_mu(h))
    linear_res = [x/n_sample for x in linear_res]
    non_linear_res = [x/n_sample for x in non_linear_res]
    all_linear_res.append(linear_res)
    all_non_linear_res.append(non_linear_res)

# print(mu_linear)
# print(mu_non_linear)
# print(stop)

# mu_linear.sort()
# mu_non_linear.sort()

import matplotlib.pyplot as plt
import seaborn as sns

palette = [
    '#FF00FF', '#00FFFF', 
    '#FFFF00', 
    '#0000FF', 
    '#00FF00', 
    # '#FF0000',
    # "#FFA500",
    # "#FF4500",
    "#ADFF2F", 
    "#7FFF00",
    "#00BFFF",    
    "#1E90FF",
    "#FF69B4", 
    "#FF1493", 
    "#DA70D6", 
    "#BA55D3", 
    "#32CD32", 
    "#FFD700" 
]

# Set the seaborn style to "whitegrid"
sns.set(style="whitegrid", font_scale=1.2, rc={"lines.linewidth": 1.5})
# Plot the figure
plt.figure(figsize=(10, 6))
# palette = sns.color_palette("Blues", 10)

for i in range(len(all_linear_res)):
    lin = all_linear_res[i]
    non_lin = all_non_linear_res[i]
    labels_ = ["Linear_mp_layer="+str(layer)+"/Width="+str(dim_list[i]) for layer in layer_list]

    sns.lineplot(x=[j for j in range(101)], y=lin, label="Linear_mp/Width="+str(dim_list[i]), color=palette[i*2])
    sns.lineplot(x=[j for j in range(101)], y=non_lin, label="Non_linear_mp/Width="+str(dim_list[i]), color=palette[i*2], linestyle='--')
plt.xlabel("Number of layers")
plt.ylabel("$\\mu$(X)")

plt.title("Oversmoothing")
plt.legend()
plt.savefig('oversmoothing_measure.pdf', format='pdf', bbox_inches='tight')

sns.set(style="whitegrid", font_scale=1.2, rc={"lines.linewidth": 1.5})
# # Plot the figure
# plt.figure(figsize=(10, 6))
# palette = sns.color_palette("Blues", 10)

# # sns.lineplot(x=[x[0] for x in mu_linear], y=[x[1] for x in mu_linear], label="Linear", color=palette[0])

# sns.lineplot(x=[x[0] for x in mu_non_linear], y=[x[1] for x in mu_non_linear], label="Non-linear", color=palette[1])
# plt.xlabel("Number of layers")
# plt.ylabel("$\\mu$(X)")

# plt.title("Oversmoothing")
# plt.legend()
# plt.savefig('oversmoothing_non_linear_mp.pdf', format='pdf', bbox_inches='tight')