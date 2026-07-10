from ogb.graphproppred import PygGraphPropPredDataset
from utils.split import split_dataset


data_name = 'ogbg-molbace'

dataset = PygGraphPropPredDataset(name=data_name, root='dataset')

train_idx, valid_idx, test_idx = split_dataset(dataset, 0.8, 0.1)
print(test_idx)

# Store the indices in a dictionary
split_indices = {
    'train': train_idx.tolist(),
    'valid': valid_idx.tolist(),
    'test': test_idx.tolist()
}
# Save the indices to a file
import json
with open('split/'+data_name+'_split_indices.json', 'w') as f:
    json.dump(split_indices, f)