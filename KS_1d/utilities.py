import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Models
# ============================================================================

class AR_MLP_1_layer(nn.Module):
    """Single hidden layer MLP for autoregressive time stepping."""
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x, return_features=False):
        features = F.relu(self.fc1(x))
        out = self.fc2(features)

        if return_features:
            return out, features  # return last hidden layer activations
        return out


class AR_MLP_deep(nn.Module):
    """Deep MLP (5 hidden layers) for autoregressive time stepping."""
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        features = F.relu(self.fc5(x))
        out = self.fc6(features) 

        if return_features:
            return out, features  # return last hidden layer activations
        return out


# ============================================================================
# Training Functions
# ============================================================================

def train_ks(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, dt=1e-3, scaler=None):
    """
    Training function for Kuramoto-Sivashinsky model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        dt: Time step for KS integration
        scaler: GradScaler for mixed precision (None to disable)
    
    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast(device_type="cuda"):
                output = model(data)
                predicted_next = data + output * dt
                loss = loss_fn(predicted_next, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            output = model(data)
            predicted_next = data + output * dt
            loss = loss_fn(predicted_next, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    # Step scheduler after epoch
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.6g}, LR: {current_lr:.6e}')
    
    return avg_loss


def train_ks_noEuler(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, dt=1e-3, scaler=None):
    """
    Training function for Kuramoto-Sivashinsky model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        dt: Time step for KS integration
        scaler: GradScaler for mixed precision (None to disable)
    
    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast(device_type="cuda"):
                output = model(data)
                loss = loss_fn(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    # Step scheduler after epoch
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.6g}, LR: {current_lr:.6e}')
    
    return avg_loss


def test_ks(model, test_loader, loss_fn, device, dt=1e-3):
    """
    Validation/test function for Kuramoto-Sivashinsky model.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test/validation data
        loss_fn: Loss function
        device: Device to evaluate on
        dt: Time step for KS integration
    
    Returns:
        avg_loss: Average test loss
    """
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            predicted_next = data + output * dt
            loss = loss_fn(predicted_next, target)
            
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    print(f'  Val Loss: {avg_loss:.6g}')
    
    return avg_loss


# ============================================================================
# Singular Value Computation
# ============================================================================

def compute_and_save_singular_values(model, data_loader, device, model_name, epoch, output_dir):
    """
    Compute and save singular values of hidden layer activations (penultimate features).
    
    Args:
        model: Neural network model with return_features capability
        data_loader: DataLoader for data
        device: Device to compute on
        model_name: Name for saving files
        epoch: Current epoch number
        output_dir: Directory to save singular values
    
    Returns:
        S: Singular values tensor
        sv_path: Path where singular values were saved
    """
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

    # Compute SVD
    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)

    # Print summary
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")

    # Save singular values
    sv_dir = os.path.join(output_dir, 'singular_values')
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, f'{model_name}_e{epoch}.pt')
    torch.save(S.cpu(), sv_path)
    print(f"Singular values saved to {sv_path}")

    return S, sv_path


# ============================================================================
# Data Loading
# ============================================================================
def load_ks_data(train_data_path, val_data_path, batch_size=256, num_workers=4, n_samples=None):
    """
    Load KS training and validation data and create DataLoaders.
    
    Args:
        train_data_path: Path to training data .npy file (timesteps, spatial_dim)
        val_data_path: Path to validation data .npy file (timesteps, spatial_dim)
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        n_samples: Number of training samples to use (None = use all). 
                   Samples are drawn randomly without replacement.
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Spatial dimension (1024)
        n_train: Number of training samples
        n_val: Number of validation samples
    """
    # Load data
    print(f"Loading training data from {train_data_path}...")
    train_data = np.load(train_data_path) 
    print(f"Training data shape: {train_data.shape}")

    print(f"Loading validation data from {val_data_path}...")
    val_data = np.load(val_data_path) 
    print(f"Validation data shape: {val_data.shape}")

    # Create autoregressive dataset: X_t -> X_{t+1}
    X_train = train_data[:-1]  # Input: all timesteps except last
    y_train = train_data[1:]   # Target: all timesteps except first

    X_val = val_data[:-1]
    y_val = val_data[1:]

    # Subsample training data if n_samples is specified
    if n_samples is not None:
        if n_samples > len(X_train):
            print(f"n_samples ({n_samples}) exceeds available training samples ({len(X_train)}) ... Reducing to {len(X_train)}")
            n_samples = len(X_train)

        indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Subsampled to {n_samples} training samples (from {train_data.shape[0]-1} available)")

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