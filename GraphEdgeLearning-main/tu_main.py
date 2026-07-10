import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score  # Import metrics
import numpy as np
import math
from models.dln import DLN_Net
from models.gcn import GCN_Net
from models.gat import GAT_Net
from models.sage import SAGE_Net
from tqdm import tqdm
import os

data_name = 'REDDIT-BINARY'  # Change this to your dataset name
dataset = TUDataset(root='data/TUDataset', name=data_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print(dataset[0].y)

processed_dataset = []

if data_name in ['REDDIT-MULTI-5K', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
    num_features = 1
    num_classes = dataset.num_classes - 1
    for i in range(len(dataset)):
        data = dataset[i]
        data.x = torch.ones((data.num_nodes, num_features), dtype=torch.float32)  # Set all node features to zero
        processed_dataset.append(data)
else:
    processed_dataset = dataset
    num_features = dataset.num_features
    num_classes = dataset.num_classes - 1

dataset = processed_dataset

print(dataset[0])
print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Number of graphs: {len(dataset)}")

folds = 10
kf = KFold(n_splits=folds, shuffle=True, random_state=42)  # Consistent split
all_train_gcn_losses = []
all_val_gcn_losses = []
all_val_gcn_metrics = [] # Changed to all_val_metrics
all_train_our_losses = []
all_val_our_losses = []
all_val_our_metrics = [] # Changed to all_val_metrics
all_val_gcm_accs = []
all_val_our_accs = []
batch_size = 128

gcn_dim = 64
dln_dim = 256
gcn_layers = 5
dln_layers = 3
dln_linear_layers = 32

for fold, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
    print(f"Fold {fold + 1}/{folds}")

    # 3.1. Create Data Loaders for each fold
    train_dataset = [dataset[i] for i in train_index]
    val_dataset = [dataset[i] for i in val_index]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    gcn_model = SAGE_Net(
        input_channels=num_features,
        hid_channels=gcn_dim,
        out_channels=num_classes,
        num_layers=gcn_layers,
        dropout=0.0,
    ).to(device)

    # dln_model = DLN_Net(
    #     num_layers=dln_layers,
    #     num_linear_layers=dln_linear_layers,
    #     input_dim=num_features,
    #     hidden_dim=dln_dim,
    #     output_dim=num_classes,
    #     epsilon=math.pi - 3,
    #     dropout=0.0,
    # ).to(device)

    # Print model parameters
    print(f"Number of parameters in GCN model: {sum(p.numel() for p in gcn_model.parameters())}")
    # print(f"Number of parameters in DLN model: {sum(p.numel() for p in dln_model.parameters())}")
    # print(stop)

    # 3.2. Define loss function and optimizer
    if data_name in ['REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    # dln_optimizer = torch.optim.Adam(dln_model.parameters(), lr=0.001)
    # dln_optimizer = torch.optim.SGD(dln_model.parameters(), lr=0.001, momentum=0.9)
    # 3.3. Training loop    
    gcn_train_losses = []
    gcn_val_losses = []
    dln_train_losses = []
    dln_val_losses = []
    gcn_val_accs = []
    dln_val_accs = []

    for epoch in tqdm(range(1, 1001)):
        # Validation
        gcn_model.eval()
        # dln_model.eval()
        gcn_val_loss = 0.0
        dln_val_loss = 0.0
        gcn_val_y_true = []
        gcn_val_y_pred = []
        dln_val_y_true = []
        dln_val_y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # GCN model
                gcn_out = gcn_model(batch.x, batch.edge_index, batch.batch)
                if data_name in ['REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
                    gcn_loss = criterion(gcn_out.squeeze(), batch.y)
                else:
                    gcn_loss = criterion(gcn_out.squeeze(), batch.y.float())
                gcn_val_loss += gcn_loss.item() / batch.y.size(0)
                gcn_val_y_true.append(batch.y.cpu())
                # print(batch.y)
                if data_name in ['REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
                    gcn_val_y_pred.append(gcn_out.argmax(dim=1).cpu())
                else:
                    gcn_val_y_pred.append(gcn_out.cpu())

                # DLN model
                # dln_out = dln_model(batch.x, batch.edge_index, batch=batch.batch)
                # if data_name in ['REDDIT-MULTI-5K', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
                #     dln_loss = criterion(dln_out.squeeze(), batch.y)
                # else:
                #     dln_loss = criterion(dln_out.squeeze(), batch.y.float())
                # dln_val_loss += dln_loss.item()
                # dln_val_y_true.append(batch.y.cpu())
                # if data_name in ['REDDIT-MULTI-5K', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
                #     dln_val_y_pred.append(dln_out.argmax(dim=1).cpu())
                # else:
                #     dln_val_y_pred.append(dln_out.cpu())
        gcn_val_losses.append(gcn_val_loss / len(val_loader))
        # dln_val_losses.append(dln_val_loss / len(val_loader))
        gcn_val_y_true = torch.cat(gcn_val_y_true)
        gcn_val_y_pred = torch.cat(gcn_val_y_pred)
        # dln_val_y_true = torch.cat(dln_val_y_true)
        # dln_val_y_pred = torch.cat(dln_val_y_pred)
        gcn_val_y_true = gcn_val_y_true.numpy()
        # dln_val_y_true = dln_val_y_true.numpy()
        if data_name in ['REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
            gcn_val_y_pred = gcn_val_y_pred.numpy()
            # dln_val_y_pred = dln_val_y_pred.numpy()
            # print(gcn_val_y_true)
            # print(gcn_val_y_pred)
            
            gcn_val_acc = accuracy_score(gcn_val_y_true, gcn_val_y_pred)
            
            # dln_val_acc = accuracy_score(dln_val_y_true, dln_val_y_pred)
            # print(gcn_val_acc)
            # print(stop)
        else:
            gcn_val_y_pred = torch.sigmoid(gcn_val_y_pred).numpy()
            # dln_val_y_pred = torch.sigmoid(dln_val_y_pred).numpy()
            gcn_val_acc = accuracy_score(gcn_val_y_true, (gcn_val_y_pred > 0.5).astype(int))
            # dln_val_acc = accuracy_score(dln_val_y_true, (dln_val_y_pred > 0.5).astype(int))
        gcn_val_accs.append(gcn_val_acc)
        # dln_val_accs.append(dln_val_acc)

        gcn_model.train()
        # dln_model.train()
        gcn_epoch_loss = 0.0
        dln_epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            gcn_optimizer.zero_grad()
            # dln_optimizer.zero_grad()

            # GCN model
            gcn_out = gcn_model(batch.x, batch.edge_index, batch.batch)
            is_labeled = batch.y == batch.y
            if data_name in ['REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
                gcn_loss = criterion(gcn_out.squeeze(), batch.y)
            else:
                gcn_loss = criterion(gcn_out.squeeze(), batch.y.float())
            gcn_loss.backward()
            gcn_optimizer.step()
            gcn_epoch_loss += gcn_loss.item() / batch.y.size(0)

            # DLN model
            # dln_out = dln_model(batch.x, batch.edge_index, batch=batch.batch)
            # is_labeled = batch.y == batch.y
            # if data_name in ['REDDIT-MULTI-5K', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
            #     dln_loss = criterion(dln_out.squeeze(), batch.y)
            # else:
            #     dln_loss = criterion(dln_out.squeeze(), batch.y.float())
            # dln_loss.backward()
            # dln_optimizer.step()
            # dln_epoch_loss += dln_loss.item()
        gcn_train_losses.append(gcn_epoch_loss / len(train_loader))
        # dln_train_losses.append(dln_epoch_loss / len(train_loader))
        
        print(f"Epoch {epoch}: GCN Loss: {gcn_train_losses[-1]:.4f}, "
              f"GCN Val Loss: {gcn_val_losses[-1]:.4f},"
              f"GCN Val Acc: {gcn_val_acc},")
        # print(f"Epoch {epoch}: DLN Loss: {dln_train_losses[-1]:.4f}, "
        #       f"DLN Val Loss: {dln_val_losses[-1]:.4f}, "
        #       f"DLN Val Acc: {dln_val_acc}")
        # if epoch == 10:
        #     print(stop)
        
    all_train_gcn_losses.append(gcn_train_losses)
    all_val_gcn_losses.append(gcn_val_losses)
    # all_train_our_losses.append(dln_train_losses)
    # all_val_our_losses.append(dln_val_losses)
    all_val_gcn_metrics.append(max(gcn_val_accs))
    # all_val_our_metrics.append(max(dln_val_accs))
    all_val_gcm_accs.append(gcn_val_accs)
    # all_val_our_accs.append(dln_val_accs)

print("Performance across folds:")
print(f"GCN Val Acc: {np.mean(all_val_gcn_metrics):.4f} ± {np.std(all_val_gcn_metrics):.4f}")   
# print(f"DLN Val Acc: {np.mean(all_val_our_metrics):.4f} ± {np.std(all_val_our_metrics):.4f}")

# Plotting the losses
# import matplotlib.pyplot as plt
# all_path = 'plots/'+data_name+'/'+str(gcn_dim)+'_'+str(gcn_layers)+'_'+str(dln_dim)+'_'+str(dln_layers)+'_'+str(dln_linear_layers)+'_'+str(batch_size)
# if not os.path.exists(all_path):
#     os.makedirs(all_path)

# for i, train_losses in enumerate(all_train_our_losses):
#     # save losses and test acc
#     path = 'plots/'+data_name+'/'+'gat_'+str(gcn_dim)+'/'+str(gcn_layers)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     with open(path + f'/train_loss_fold_{i+1}.txt', 'w') as f:
#         for loss in train_losses:
#             f.write(f"{loss}\n")
#     with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
#         for loss in all_val_our_losses[i]:
#             f.write(f"{loss}\n")
    
    # path = 'plots/'+data_name+'/'+'dln'+str(dln_dim)+'/mp_'+str(dln_layers)+'/fl_'+str(dln_linear_layers)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # with open(path + f'/train_loss_fold_{i+1}.txt', 'w') as f:
    #     for loss in all_train_our_losses[i]:
    #         f.write(f"{loss}\n")
    # with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
    #     for loss in all_val_our_losses[i]:
    #         f.write(f"{loss}\n")
    # plt.figure(figsize=(12, 6))
    # plt.plot(train_losses, label=f'Train GCN Fold {i+1}')
    # plt.plot(all_train_our_losses[i], label=f'Train DLN Fold {i+1}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Train Losses for Fold {i+1}')
    # plt.legend()
    # plt.grid()
    # plt.savefig(all_path + f'/train_loss_fold_{i+1}.png')
    # plt.close()

# for i, val_losses in enumerate(all_val_our_losses):
    # path = 'plots/'+data_name+'/'+'gcn'+str(gcn_dim)+'/'+str(gcn_layers)
    # with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
    #     for loss in all_val_gcn_losses[i]:
    #         f.write(f"{loss}\n")
    # with open(path + f'/val_acc_fold_{i+1}.txt', 'w') as f:
    #     for acc in all_val_gcm_accs[i]:
    #         f.write(f"{acc}\n")
    # path = 'plots/'+data_name+'/'+'dln'+str(dln_dim)+'/mp_'+str(dln_layers)+'/fl_'+str(dln_linear_layers)

    # with open(path + f'/val_loss_fold_{i+1}.txt', 'w') as f:
    #     for loss in all_val_our_losses[i]:
    #         f.write(f"{loss}\n")
    # with open(path + f'/val_acc_fold_{i+1}.txt', 'w') as f:
    #     for acc in all_val_our_accs[i]:
    #         f.write(f"{acc}\n")
    # plt.figure(figsize=(12, 6))
    # plt.plot(val_losses, label=f'Val GCN Fold {i+1}')
    # plt.plot(all_val_our_losses[i], label=f'Val DLN Fold {i+1}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Validation Losses for Fold {i+1}')
    # plt.legend()
    # plt.grid()
    
    # plt.savefig(all_path+f'/val_loss_fold_{i+1}.png')
    # plt.close()
# Plotting the metrics  

# for i, val_metrics in enumerate(all_val_gcn_metrics):
#     plt.figure(figsize=(12, 6))
#     plt.plot(val_metrics, label=f'Val GCN Fold {i+1}')
#     plt.plot(all_val_our_metrics[i], label=f'Val DLN Fold {i+1}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Validation Metrics for Fold {i+1}')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'plots/val_metrics_fold_{i+1}.png')
#     plt.close()