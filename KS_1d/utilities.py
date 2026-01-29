import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MLP_AR(nn.Module):
    def __init__(self, input_dim=1024, hidden1_dim=512, hidden2_dim=512, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)

        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        
        self.fc3 = nn.Linear(hidden2_dim, output_dim)


    def forward(self, x, return_features=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)

        if return_features:
            return out, h2  # return last hidden layer activations
        return out


def compute_and_save_singular_values(model, data_loader, device, model_name, epoch, output_dir):
    # Compute and save singular values of hidden layer activations (penultimate features).

    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_feats = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader, 1):
            data = data.to(device)
            _, feats = model(data, return_features=True)  # directly get features
            all_feats.append(feats.cpu())

            if batch_idx % 100 == 0 or batch_idx == len(data_loader):
                print(f'Processed batch {batch_idx}/{len(data_loader)}')

    # Concatenate features
    Phi = torch.cat(all_feats, dim=0)
    print(f"Collected hidden activations shape: {Phi.shape}")

    # Center and normalize
    Phi = Phi - Phi.mean(dim=0, keepdim=True)
    Phi = Phi / Phi.shape[0]**0.5

    # Compute SVD
    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)

    # Print summary
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")

    # Save singular values
    sv_dir = os.path.join(output_dir, 'singular_values')
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, f'{model_name}_e{epoch}.pt')
    torch.save(S, sv_path)
    print(f"Singular values saved to {sv_path}")

    return S


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_ks_data(train_data_path, val_data_path, batch_size=256, num_workers=4):
    """
    Load KS training and validation data and create DataLoaders.
    
    Args:
        train_data_path: Path to training data .npy file (1024, 200000)
        val_data_path: Path to validation data .npy file (1024, 50000)
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Spatial dimension (1024)
        n_train: Number of training samples
        n_val: Number of validation samples
    """
    # Load data
    print(f"Loading training data from {train_data_path}...")
    train_data = np.load(train_data_path)  # Shape: (1024, 200000)
    print(f"Training data shape: {train_data.shape}")

    print(f"Loading validation data from {val_data_path}...")
    val_data = np.load(val_data_path)  # Shape: (1024, 50000)
    print(f"Validation data shape: {val_data.shape}")

    # Transpose to (time, spatial)
    train_data = train_data.T  # (200000, 1024)
    val_data = val_data.T      # (50000, 1024)

    # Create autoregressive dataset: X_t -> X_{t+1}
    X_train = train_data[:-1]  # Input: all timesteps except last
    y_train = train_data[1:]   # Target: all timesteps except first

    X_val = val_data[:-1]
    y_val = val_data[1:]

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    input_dim = X_train.shape[1]  # 1024
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    return train_loader, val_loader, input_dim, n_train, n_val