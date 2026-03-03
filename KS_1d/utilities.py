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
            return out, features
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
            return out, features
        return out


# ============================================================================
# Training Functions
# ============================================================================

def train_ks(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, dt=1e-3, scaler=None):
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                output = model(data)
                predicted_next = data + output * dt
                loss = loss_fn(predicted_next, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            predicted_next = data + output * dt
            loss = loss_fn(predicted_next, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.6g}, LR: {current_lr:.6e}')

    return avg_loss


def test_ks(model, test_loader, loss_fn, device, dt=1e-3):
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
    print(f'  Test Loss: {avg_loss:.6g}')

    return avg_loss


# ============================================================================
# Activation & Singular Value Computation
# ============================================================================

def save_hidden_activations(model, train_loader, test_loader, device, model_name, epoch, output_dir):
    print("\n" + "="*50)
    print(f"Saving hidden activations at epoch {epoch}...")
    print("="*50)

    model.eval()

    def collect_features(data_loader, split):
        all_feats = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader, 1):
                data = data.to(device)
                _, feats = model(data, return_features=True)
                all_feats.append(feats.cpu())

                if batch_idx % 100 == 0 or batch_idx == len(data_loader):
                    print(f'[{split}] Processed batch {batch_idx}/{len(data_loader)}')

        Phi = torch.cat(all_feats, dim=0)
        print(f"[{split}] Collected hidden activations shape: {Phi.shape}")
        return Phi

    train_feats = collect_features(train_loader, "train")
    test_feats   = collect_features(test_loader,   "test")

    os.makedirs(os.path.join(output_dir, 'activations'),     exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'singular_values'), exist_ok=True)

    acts_path = os.path.join(output_dir, 'activations', f'{model_name}_e{epoch}.pt')
    torch.save({'train': train_feats, 'test': test_feats}, acts_path)
    print(f"Hidden activations saved to {acts_path}")

    print("\nComputing singular values...")
    sv_dict = {}
    sv_mean = {}
    for split, feats in [("train", train_feats), ("test", test_feats)]:
        S = torch.linalg.svdvals(feats)
        sv_dict[split] = S
        sv_mean[split] = S.mean()
        print(f"[{split}] Singular values shape: {S.shape}  |  top-5: {S[:5].tolist()}")

    sv_path = os.path.join(output_dir, 'singular_values', f'{model_name}_e{epoch}.pt')
    torch.save(sv_dict, sv_path)
    print(f"Singular values saved to {sv_path}")

    return acts_path, sv_path, float(sv_mean['train']), float(sv_mean['test'])



# ============================================================================
# Data Loading
# ============================================================================

def load_ks_data(train_data_path, test_data_path, batch_size=256, num_workers=4, n_samples=None):
    print(f"Loading training data from {train_data_path}...")
    train_data = np.load(train_data_path)
    print(f"Training data shape: {train_data.shape}")

    print(f"Loading testing data from {test_data_path}...")
    test_data = np.load(test_data_path)
    print(f"Testing data shape: {test_data.shape}")

    X_train = train_data[:-1]
    y_train = train_data[1:]

    X_test = test_data[:-1]
    y_test = test_data[1:]

    if n_samples is not None:
        if n_samples > len(X_train):
            print(f"n_samples ({n_samples}) exceeds available training samples ({len(X_train)}) ... Reducing to {len(X_train)}")
            n_samples = len(X_train)

        indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Subsampled to {n_samples} training samples (from {train_data.shape[0]-1} available)")

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor   = torch.FloatTensor(X_test)
    y_test_tensor   = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset   = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    input_dim = X_train.shape[1]
    n_train   = X_train.shape[0]
    n_test    = X_test.shape[0]

    return train_loader, test_loader, input_dim, n_train, n_test