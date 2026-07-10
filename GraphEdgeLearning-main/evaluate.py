import torch
from models.mlp import MLP, MLP_Concat, MLP_Custom
from models.gine import GINE_Net
from models.lcn import LCN_Net

def evaluate_model(model, dataloader, device, evaluator):
    """
    Evaluate the model and return AUC-ROC (using the OGB Evaluator).

    Parameters:
      model           : The model to evaluate.
      dataloader      : A DataLoader yielding batches.
      device          : Torch device (cpu or cuda).
      evaluator       : OGB evaluator.
      
    Returns:
      A dictionary containing the AUC-ROC score.
    """
    model.to(device)
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if isinstance(model, MLP):
                outputs = model(batch.x, batch.batch)
            elif isinstance(model, MLP_Custom):
                outputs = model(batch.x, batch.edge_index, batch.adj_list, batch.batch)
            elif isinstance(model, GINE_Net) and hasattr(batch, 'edge_attr'):
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            elif isinstance(model, LCN_Net):
                outputs = model(batch.x, batch.edge_index, batch.adj_list, batch.batch)
            else:
                outputs = model(batch.x, batch.edge_index, batch.batch)
            y_true_list.append(batch.y)
            y_pred_list.append(outputs)

    # Concatenate results
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    # Reshape to 2D arrays: [num_samples, -1]
    y_true = y_true.view(y_true.size(0), -1).cpu().numpy()
    y_pred = y_pred.view(y_pred.size(0), -1).cpu().detach().numpy()

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    return evaluator.eval(input_dict)