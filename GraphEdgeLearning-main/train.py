import torch
from models.mlp import MLP, MLP_Concat, MLP_Custom
from models.gine import GINE_Net
from models.lcn import LCN_Net

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, evaluator, device, epochs=100, save_path=None):
    train_losses = []
    val_losses = []

    max_val_roc_auc = -float('inf')

    for epoch in range(epochs):
        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():  # Disable gradient computation
            for data in val_dataloader:
                data = data.to(device)
                if isinstance(model, MLP):
                    outputs = model(data.x, data.batch)
                elif isinstance(model, MLP_Custom):
                    outputs = model(data.x, data.edge_index, data.adj_list, data.batch)
                elif isinstance(model, GINE_Net) and hasattr(data, 'edge_attr'):
                    outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
                elif isinstance(model, LCN_Net):
                    outputs = model(data.x, data.edge_index, data.adj_list, data.batch)
                else:
                    outputs = model(data.x, data.edge_index, data.batch)
                
                is_label = (data.y == data.y).squeeze()
                loss = criterion(outputs[is_label], data.y[is_label].float())  # Match shapes
                running_val_loss += loss.item()
                y_true_list.append(data.y)
                y_pred_list.append(outputs)

        val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(val_loss)
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

        # Reshape to 2D arrays: [num_samples, -1]
        y_true = y_true.view(y_true.size(0), -1).cpu().numpy()
        y_pred = y_pred.view(y_pred.size(0), -1).cpu().detach().numpy()

        input_dict = {'y_true': y_true, 'y_pred': y_pred}

        roc_auc = evaluator.eval(input_dict)['rocauc']

        if roc_auc > max_val_roc_auc:
            max_val_roc_auc = roc_auc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        # Training phase
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            if isinstance(model, MLP):
                outputs = model(data.x, data.batch)
            elif isinstance(model, MLP_Custom):
                outputs = model(data.x, data.edge_index, data.adj_list, data.batch)
            elif isinstance(model, GINE_Net) and hasattr(data, 'edge_attr'):
                outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
            elif isinstance(model, LCN_Net):
                    outputs = model(data.x, data.edge_index, data.adj_list, data.batch)
            else:
                outputs = model(data.x, data.edge_index, data.batch)
            is_label = (data.y == data.y).squeeze()
            loss = criterion(outputs[is_label], data.y[is_label].float())  # Match shapes
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # Print epoch summary
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val ROC-AUC: {roc_auc:.4f}")

    return train_losses, val_losses