import torch

def split_dataset(dataset, train_ratio, valid_ratio):
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)

    # Compute split sizes
    train_size = int(train_ratio * num_samples)
    valid_size = int(valid_ratio * num_samples)

    # Slice indices
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    print(f"Train size: {len(train_indices)}, Val size: {len(valid_indices)}, Test size: {len(test_indices)}")

    return train_indices, valid_indices, test_indices