import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score  # Import metrics
import numpy as np
import math
from models.dln import DLN_Net
from models.gcn import GCN_Net
from tqdm import tqdm
import os
import json
from utils.split import split_dataset
from torch.optim.lr_scheduler import ExponentialLR

dataset = MNISTSuperpixels(root='data/MNISTSuperpixels')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
num_features = dataset.num_features
num_classes = 4
data_name = 'MNISTSuperpixels'
print(dataset[0])
print(dataset[0].y)

num_run = 10

all_train_our_losses = []
all_val_our_losses = []
all_test_our_losses = []
all_val_our_metrics = [] # Changed to all_val_metrics
all_val_our_accs = []
all_test_our_accs = []
batch_size = 128

gcn_dim = 65
dln_mp_dim = 100
dln_fl_dim=300
# gcn_layers = 1
dln_layers = 1
dln_linear_layers = 1

split_path = 'split/MNISTSuperpixels_split_indices.json'
if not os.path.exists(split_path):
    # Create the split indices if they don't exist
    # filter dataset
    select_indices = [i for i in range(len(dataset)) if dataset[i].y.item() == 0 or dataset[i].y.item() == 1 or dataset[i].y.item() == 2 or dataset[i].y.item() == 3]
    import random
    # select_indices = [i for i in range(len(dataset))]
    random.seed(42)
    random.shuffle(select_indices)
    select_indices = select_indices[:4000]
    train_idx, valid_idx, test_idx = select_indices[:3200], select_indices[3200:3600], select_indices[3600:]
    split_indices = {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx
    }
    with open(split_path, 'w') as f:
        json.dump(split_indices, f)

with open(split_path, 'r') as f:
    split_indices = json.load(f)
train_indices = split_indices['train']
val_indices = split_indices['valid']
test_indices = split_indices['test']
train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

for i in range(num_run):
    dln_model = DLN_Net(
        num_layers=dln_layers,
        num_linear_layers=dln_linear_layers,
        input_dim=num_features,
        mp_hidden_dim=dln_mp_dim,
        fl_hidden_dim=dln_fl_dim,
        output_dim=num_classes,
        epsilon=math.pi - 3,
        dropout=0.0,
    ).to(device)

    # Print model parameters
    print(f"Number of parameters in DLN model: {sum(p.numel() for p in dln_model.parameters())}")
    # print(stop)

    criterion = torch.nn.CrossEntropyLoss()
    # gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    dln_optimizer = torch.optim.Adam(dln_model.parameters(), lr=0.001)
    # dln_optimizer = torch.optim.SGD(dln_model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = ExponentialLR(dln_optimizer, gamma=0.95)
    # 3.3. Training loop    
    dln_train_losses = []
    dln_val_losses = []
    dln_test_losses = []
    dln_val_accs = []
    dln_test_accs = []

    for epoch in tqdm(range(1, 1001)):
        # Validation
        dln_model.eval()
        dln_val_loss = 0.0
        dln_val_y_true = []
        dln_val_y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                dln_out = dln_model(batch.x, batch.edge_index, batch=batch.batch)
                dln_loss = criterion(dln_out.squeeze(), batch.y)
                dln_val_loss += dln_loss.item()
                dln_val_y_true.append(batch.y.cpu())
                dln_val_y_pred.append(dln_out.argmax(dim=1).cpu())
        dln_val_losses.append(dln_val_loss / len(val_loader))
        dln_val_y_true = torch.cat(dln_val_y_true)
        dln_val_y_pred = torch.cat(dln_val_y_pred)
        dln_val_y_true = dln_val_y_true.numpy()
        dln_val_y_pred = dln_val_y_pred.numpy()
        dln_val_acc = accuracy_score(dln_val_y_true, dln_val_y_pred)
        dln_val_accs.append(dln_val_acc)

        # Test
        dln_model.eval()
        dln_test_loss = 0.0
        dln_test_y_true = []
        dln_test_y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                dln_out = dln_model(batch.x, batch.edge_index, batch=batch.batch)
                dln_loss = criterion(dln_out.squeeze(), batch.y)
                dln_test_loss += dln_loss.item()
                dln_test_y_true.append(batch.y.cpu())
                dln_test_y_pred.append(dln_out.argmax(dim=1).cpu())
        dln_test_losses.append(dln_test_loss / len(test_loader))
        dln_test_y_true = torch.cat(dln_test_y_true)
        dln_test_y_pred = torch.cat(dln_test_y_pred)
        dln_test_y_true = dln_test_y_true.numpy()
        dln_test_y_pred = dln_test_y_pred.numpy()
        dln_test_acc = accuracy_score(dln_test_y_true, dln_test_y_pred)
        dln_test_accs.append(dln_test_acc)

        # gcn_model.train()
        dln_model.train()
        gcn_epoch_loss = 0.0
        dln_epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            dln_optimizer.zero_grad()

            # DLN model
            dln_out = dln_model(batch.x, batch.edge_index, batch=batch.batch)
            is_labeled = batch.y == batch.y
            dln_loss = criterion(dln_out.squeeze(), batch.y)
            dln_loss.backward()
            dln_optimizer.step()
            dln_epoch_loss += dln_loss.item()
        dln_train_losses.append(dln_epoch_loss / len(train_loader))
        # scheduler.step()
        
        print(f"Epoch {epoch}: DLN Loss: {dln_train_losses[-1]:.4f}, "
              f"DLN Val Loss: {dln_val_losses[-1]:.4f}, "
              f"DLN Test Loss: {dln_test_losses[-1]:.4f}, "
              f"DLN Val Acc: {dln_val_acc}"
              f"DLN Test Acc: {dln_test_acc}")

    all_train_our_losses.append(dln_train_losses)
    all_val_our_losses.append(dln_val_losses)
    all_test_our_losses.append(dln_test_losses)
    all_val_our_metrics.append(dln_val_accs[dln_val_accs.index(max(dln_val_accs))])
    all_val_our_accs.append(dln_val_accs)
    all_test_our_accs.append(dln_test_accs)

print("Performance across Runs:")
print(f"DLN Val Acc: {np.mean(all_val_our_metrics):.4f} ± {np.std(all_val_our_metrics):.4f}")


for i, train_losses in enumerate(all_train_our_losses):

    path = 'plots/'+data_name+'/4000_no_lr_decay/'+'dln_mp_dim_'+str(dln_mp_dim)+'/trained_mp_'+str(dln_layers)+'_dln_fl_dim_'+str(dln_fl_dim)+'/fl_'+str(dln_linear_layers)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + f'/train_loss_fold_{i+1}.txt', 'w') as f:
        for loss in all_train_our_losses[i]:
            f.write(f"{loss}\n")
    with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
        for loss in all_val_our_losses[i]:
            f.write(f"{loss}\n")
    with open(path + f'/test_loss_fold_{i+1}.txt', 'w') as f:
        for loss in all_test_our_losses[i]:
            f.write(f"{loss}\n")

# for i, val_losses in enumerate(all_val_our_losses):
#     path = 'plots/'+data_name+'/'+'dln'+str(dln_dim)+'/mp_'+str(dln_layers)+'/fl_'+str(dln_linear_layers)

#     with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
#         for loss in all_val_our_losses[i]:
#             f.write(f"{loss}\n")
#     with open(path + f'/val_acc_fold_{i+1}.txt', 'w') as f:
#         for acc in all_val_our_accs[i]:
#             f.write(f"{acc}\n")