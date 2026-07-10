import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from models.dln import DLN_Net
from tqdm import tqdm
from utils.split import split_dataset
import json
import matplotlib.pyplot as plt
import math

def train():
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data.to(device)
        # out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        out = model(data.x, data.edge_index, batch=data.batch)
        is_labeled = data.y == data.y
        loss = criterion(out[is_labeled], data.y.float()[is_labeled])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss


def test(loder):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loder:
            data.to(device)
            # out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            out = model(data.x, data.edge_index, batch=data.batch)
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)

data_name = 'ogbg-molbace'

dataset = PygGraphPropPredDataset(name=data_name, root='dataset')

evaluator = Evaluator(name=data_name)

loop_n = 1
list_train_rocauc = [[] for i in range(loop_n)]
list_valid_rocauc = [[] for i in range(loop_n)]
list_test_rocauc = [[] for i in range(loop_n)]
max_train_rocauc = []
max_valid_rocauc = []
max_test_rocauc = []

path = 'split/'+data_name+'_split_indices.json'
with open(path, 'r') as f:
    split_idx = json.load(f)

for i in range(loop_n):

    # split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=512, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

    # train_idx, valid_idx, test_idx = split_dataset(dataset, 0.8, 0.1)
    # train_loader = DataLoader(dataset[train_idx], batch_size=32, shuffle=True)
    # valid_loader = DataLoader(dataset[valid_idx], batch_size=32, shuffle=False)
    # test_loader = DataLoader(dataset[test_idx], batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DLN_Net(
        num_layers=3,
        num_linear_layers=6,
        hidden_dim=300,
        epsilon=math.pi - 3,
    )
    model = model.to(device)
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate number of trainable parameters (parameters requiring gradients)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the results
    print(f"Total Parameters: {total_params:,}") # Using :, adds commas for readability
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {total_params - trainable_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    max_valid = 0.0
    loss_list = []
    for epoch in tqdm(range(1,2501)):
        loss = train()
        loss_list.append(loss)
        train_acc = test(train_loader)
        valid_acc = test(valid_loader)
        test_acc = test(test_loader)
        list_train_rocauc[i].append(list(train_acc.values()))
        list_valid_rocauc[i].append(list(valid_acc.values()))
        list_test_rocauc[i].append(list(test_acc.values()))
        # if list(valid_acc.values())[0] >= max_valid:
        #     max_test = list(test_acc.values())[0]
        #     torch.save(model.state_dict(), 'models/Transformer.pth')
        print(
            f'Epoch: {epoch:03d}, Train AUC: {train_acc}, Valid AUC :{valid_acc}, Test AUC: {test_acc}')

    max_valid_rocauc_result = max(list_valid_rocauc[i])
    index_test = list_valid_rocauc[i].index(max_valid_rocauc_result)
    max_train_rocauc.append(list_train_rocauc[i][index_test])
    max_valid_rocauc.append(list_valid_rocauc[i][index_test])
    max_test_rocauc.append(list_test_rocauc[i][index_test])

    # Plotting the loss, min cell is 0.1, large figure
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()  # Clear the current figure for the next plot
    # Plotting the ROC AUC in three figures
    plt.figure(figsize=(10, 5))
    plt.plot(list_train_rocauc[i], label='Train ROC AUC', color='blue')
    plt.plot(list_valid_rocauc[i], label='Valid ROC AUC', color='orange')
    plt.plot(list_test_rocauc[i], label='Test ROC AUC', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Epochs')
    plt.legend()
    plt.savefig('roc_auc.png')
    plt.clf()  # Clear the current figure for the next plot

    # save loss and roc_auc
    with open('loss.json', 'w') as f:
        json.dump(loss_list, f)
    with open('roc_auc.json', 'w') as f:
        json.dump([list_train_rocauc[i], list_valid_rocauc[i], list_test_rocauc[i]], f)

for i in range(loop_n):
    print('max_train_rocauc -> max_valid_rocauc -> max_test_rocauc:', max_train_rocauc[i], ' -> ', max_valid_rocauc[i],
          ' -> ', max_test_rocauc[i])

print("Mean", '->', torch.mean(torch.tensor(max_train_rocauc), dim=0).item(),
      ' -> ', torch.mean(torch.tensor(max_valid_rocauc), dim=0).item(),
      ' -> ', torch.mean(torch.tensor(max_test_rocauc), dim=0).item())
print("std:", ' -> ', torch.std(torch.tensor(max_train_rocauc), dim=0).item(),
      ' -> ', torch.std(torch.tensor(max_valid_rocauc), dim=0).item(),
      ' -> ', torch.std(torch.tensor(max_test_rocauc), dim=0).item())