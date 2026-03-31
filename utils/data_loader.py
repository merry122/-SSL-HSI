import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(batch_size=256, use_patches=False, patch_size=9):
    data = sio.loadmat("data/salinas.mat")["salinas_corrected"]
    labels = sio.loadmat("data/salinas_gt.mat")["salinas_gt"]

    if use_patches:
        return load_patches(data, labels, batch_size, patch_size)
    else:
        X = data.reshape(-1, data.shape[2])
        y = labels.reshape(-1)

        mask = y > 0
        X = X[mask]
        y = y[mask] - 1

        # normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return X, y, loader


def load_patches(data, labels, batch_size=256, patch_size=9):
    H, W, C = data.shape
    patches = []
    patch_labels = []

    for i in range(patch_size//2, H - patch_size//2):
        for j in range(patch_size//2, W - patch_size//2):
            if labels[i, j] > 0:
                patch = data[i-patch_size//2:i+patch_size//2+1, j-patch_size//2:j+patch_size//2+1, :]
                patches.append(patch.flatten())
                patch_labels.append(labels[i, j] - 1)

    X = np.array(patches)
    y = np.array(patch_labels)

    # normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return X, y, loader