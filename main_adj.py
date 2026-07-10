import argparse
import torch
import torch.nn as nn
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from utils.split import split_dataset
from datasets.graph_as_bag import GraphAsBagDataset
from datasets.adj_concate import AdjacencyConcateDataset
from models.mlp import MLP, MLP_Concat, MLP_Custom
from models.lcn import LCN_Net
from models.gcn import SimpleGCN, GCN_Net
from models.gin import GIN_Net
from models.gine import GINE_Net
from torch import optim
from train import train_model
from evaluate import evaluate_model
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser(description='GNN baselines on ogbg-mol* datasets')
parser.add_argument('--data_name', type=str, default='ogbg-molbace')
parser.add_argument('--data_path', type=str, default='datasets')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--save_path', type=str, default='saved_models')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

# Load the dataset
dataset = PygGraphPropPredDataset(name=args.data_name, root=args.data_path)

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of node features: {dataset.num_node_features}')
print(f'Number of edge features: {dataset.num_edge_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of tasks: {dataset.num_tasks}')
dataset.get_summary()

# Split dataset
train_indices, valid_indices, test_indices = split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1)

# Get dataloader for normal GNN
train_dataloader_gnn = DataLoader(dataset[train_indices], batch_size=args.batch_size, shuffle=True)
valid_dataloader_gnn = DataLoader(dataset[valid_indices], batch_size=args.batch_size, shuffle=False)
test_dataloader_gnn = DataLoader(dataset[test_indices], batch_size=args.batch_size, shuffle=False)

# Get dataloader for Graph As Bag
graph_as_bag_dataset = GraphAsBagDataset(dataset)
print(f'num of graphs: {len(graph_as_bag_dataset)}')
graph_as_bag_train_dataloader = DataLoader(Subset(graph_as_bag_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
graph_as_bag_valid_dataloader = DataLoader(Subset(graph_as_bag_dataset, valid_indices), batch_size=args.batch_size, shuffle=False)
graph_as_bag_test_dataloader = DataLoader(Subset(graph_as_bag_dataset, test_indices), batch_size=args.batch_size, shuffle=False)

# Get dataloader for Adjacency Concate
adj_concate_dataset = AdjacencyConcateDataset(dataset)
train_dataloader_with_adj = DataLoader(Subset(adj_concate_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
valid_dataloader_with_adj = DataLoader(Subset(adj_concate_dataset, valid_indices), batch_size=args.batch_size, shuffle=False)
test_dataloader_with_adj = DataLoader(Subset(adj_concate_dataset, test_indices), batch_size=args.batch_size, shuffle=False)

# Run multiple times
test_results_mlp = []
test_results_gcn = []
test_results_gin = []
test_results_gine = []
test_results_concate = []

for i in range(args.runs):
    print(f'Run {i+1}/{args.runs}')

    

    # Define models
    input_channels = dataset.num_node_features
    output_channels = dataset.num_tasks
    model_mlp = MLP(input_channels, args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    # model_gcn = SimpleGCN(input_channels, args.hidden_channels, output_channels).to(device)
    model_gcn = GCN_Net(args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    model_gin = GIN_Net(args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    model_gine = GINE_Net(args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    # model_concate = MLP(adj_concate_dataset[0].x.size(1), args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    # model_concate = MLP_Concat(adj_concate_dataset[0].adj_list[0].size(1), args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    # model_concate = MLP_Custom(adj_concate_dataset[0].adj_list[0].size(1), args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)
    model_concate = LCN_Net(adj_concate_dataset[0].adj_list[0].size(1), args.hidden_channels, output_channels, args.num_layers, args.dropout).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=args.lr)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=args.lr)
    optimizer_gin = optim.Adam(model_gin.parameters(), lr=args.lr)
    optimizer_gine = optim.Adam(model_gine.parameters(), lr=args.lr)
    optimizer_concate = optim.Adam(model_concate.parameters(), lr=args.lr)

    # Train models
    evaluator = Evaluator(name=args.data_name)

    # Train MLP
    print('Training MLP...')
    train_losses_mlp, val_losses_mlp = train_model(
        model_mlp, graph_as_bag_train_dataloader, graph_as_bag_valid_dataloader, criterion, optimizer_mlp, evaluator, device, args.epochs, args.save_path+'/mlp_'+args.data_name+'.pt'
    )

    # Train GCN
    print('Training GCN...')
    train_losses_gcn, val_losses_gcn = train_model(
        model_gcn, train_dataloader_gnn, valid_dataloader_gnn, criterion, optimizer_gcn, evaluator, device, args.epochs, args.save_path+'/gcn_'+args.data_name+'.pt'
    )

    # Train GIN
    print('Training GIN...')
    train_losses_gin, val_losses_gin = train_model(
        model_gin, train_dataloader_gnn, valid_dataloader_gnn, criterion, optimizer_gin, evaluator, device, args.epochs, args.save_path+'/gin_'+args.data_name+'.pt'
    )

    # Train GINE
    print('Training GINE...')
    train_losses_gine, val_losses_gine = train_model(
        model_gine, train_dataloader_gnn, valid_dataloader_gnn, criterion, optimizer_gine, evaluator, device, args.epochs, args.save_path+'/gine_'+args.data_name+'.pt'
    )

    # Train Concatenated
    print('Training Concatenated...')
    train_losses_concate, val_losses_concate = train_model(
        model_concate, train_dataloader_with_adj, valid_dataloader_with_adj, criterion, optimizer_concate, evaluator, device, args.epochs, args.save_path+'/concate_'+args.data_name+'.pt'
    )

    ### Plot losses
    plt.figure(figsize=(12, 8))

    # MLP Losses
    plt.plot(train_losses_mlp, label='MLP (Train)', linestyle='-', linewidth=2, color='blue')  # Solid blue line for training
    plt.plot(val_losses_mlp, label='MLP (Validation)', linestyle='--', linewidth=2, color='blue')  # Dashed blue line for validation

    # # GCN Losses
    plt.plot(train_losses_gcn, label='GCN (Train)', linestyle='-', linewidth=2, color='orange')  # Solid orange line for training
    plt.plot(val_losses_gcn, label='GCN (Validation)', linestyle='--', linewidth=2, color='orange')  # Dashed orange line for validation

    # GIN Losses
    plt.plot(train_losses_gin, label='GIN (Train)', linestyle='-', linewidth=2, color='green')  # Solid green line for training
    plt.plot(val_losses_gin, label='GIN (Validation)', linestyle='--', linewidth=2, color='green')  # Dashed green line for validation

    # GINE Losses
    plt.plot(train_losses_gine, label='GINE (Train)', linestyle='-', linewidth=2, color='purple')  # Solid purple line for training
    plt.plot(val_losses_gine, label='GINE (Validation)', linestyle='--', linewidth=2, color='purple')  # Dashed purple line for validation

    # Concatenated Losses
    plt.plot(train_losses_concate, label='Concatenated (Train)', linestyle='-', linewidth=2, color='red')  # Solid red line for training
    plt.plot(val_losses_concate, label='Concatenated (Validation)', linestyle='--', linewidth=2, color='red')  # Dashed red line for validation

    # # Adding labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves for MLP, GCN, GIN, GINE and Concatenated Models')
    plt.legend()
    plt.grid(True)

    plt.ylim(0, 2)
    # Display the plot
    plt.show()
    # Save the plot
    plt.savefig('loss_curves_'+args.data_name+'.pdf', format='pdf')

    # Evaluate models
    # Load the best model
    model_mlp.load_state_dict(torch.load(args.save_path+'/mlp_'+args.data_name+'.pt', weights_only=False))
    model_gcn.load_state_dict(torch.load(args.save_path+'/gcn_'+args.data_name+'.pt', weights_only=False))
    model_gin.load_state_dict(torch.load(args.save_path+'/gin_'+args.data_name+'.pt', weights_only=False))
    model_gine.load_state_dict(torch.load(args.save_path+'/gine_'+args.data_name+'.pt', weights_only=False))
    model_concate.load_state_dict(torch.load(args.save_path+'/concate_'+args.data_name+'.pt', weights_only=False))

    # Evaluate MLP
    print('Evaluating MLP...')
    results_mlp = evaluate_model(model_mlp, graph_as_bag_test_dataloader, device, evaluator)
    print(f'MLP Test results: {results_mlp}')

    # Evaluate GCN
    print('Evaluating GCN...')
    results_gcn = evaluate_model(model_gcn, test_dataloader_gnn, device, evaluator)
    print(f'GCN Test results: {results_gcn}')

    # Evaluate GIN
    print('Evaluating GIN...')
    results_gin = evaluate_model(model_gin, test_dataloader_gnn, device, evaluator)
    print(f'GIN Test results: {results_gin}')

    # Evaluate GINE
    print('Evaluating GINE...')
    results_gine = evaluate_model(model_gine, test_dataloader_gnn, device, evaluator)
    print(f'GINE Test results: {results_gine}')

    # Evaluate Concatenated
    print('Evaluating Concatenated...')
    results_concate = evaluate_model(model_concate, test_dataloader_with_adj, device, evaluator)
    print(f'Concatenated Test results: {results_concate}')

    test_results_mlp.append(results_mlp['rocauc'])
    test_results_gcn.append(results_gcn['rocauc'])
    test_results_gin.append(results_gin['rocauc'])
    test_results_gine.append(results_gine['rocauc'])
    test_results_concate.append(results_concate['rocauc'])

# Print results
print(f"The mean test results for MLP are: {sum(test_results_mlp)/args.runs}, The standard deviation is: {torch.std(torch.tensor(test_results_mlp))}")
print(f"The mean test results for GCN are: {sum(test_results_gcn)/args.runs}, The standard deviation is: {torch.std(torch.tensor(test_results_gcn))}")
print(f"The mean test results for GIN are: {sum(test_results_gin)/args.runs}, The standard deviation is: {torch.std(torch.tensor(test_results_gin))}")
print(f"The mean test results for GINE are: {sum(test_results_gine)/args.runs}, The standard deviation is: {torch.std(torch.tensor(test_results_gine))}")
print(f"The mean test results for Concatenated are: {sum(test_results_concate)/args.runs}, The standard deviation is: {torch.std(torch.tensor(test_results_concate))}")