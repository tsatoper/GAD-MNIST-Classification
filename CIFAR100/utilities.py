
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class CIFAR100Loader:
    def __init__(
        self,
        root="./data",
        n_samples=50000,
        batch_size=128,
        num_workers=4,
        seed=0
    ):
        assert n_samples % 100 == 0, "n_samples must be divisible by 100"

        self.root = root
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.samples_per_class = n_samples // 100

        self._build_transforms()
        self._build_datasets()

    def _build_transforms(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])

    def _build_datasets(self):
        # Load full training set WITHOUT transforms to select indices
        full_train = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=None
        )

        targets = np.array(full_train.targets)
        rng = np.random.default_rng(self.seed)

        balanced_indices = []
        for c in range(100):
            class_idx = np.where(targets == c)[0]
            chosen = rng.choice(
                class_idx,
                size=self.samples_per_class,
                replace=False
            )
            balanced_indices.extend(chosen)

        rng.shuffle(balanced_indices)

        # Reload dataset with transforms
        train_dataset_full = datasets.CIFAR100(
            root=self.root,
            train=True,
            download=False,
            transform=self.transform_train
        )

        self.train_dataset = Subset(train_dataset_full, balanced_indices)

        self.test_dataset = datasets.CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform_test
        )

    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader


def train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, scaler=None, n_classes=100):
    """
    Training function with mixed precision support.
    
    Args:
        scaler: GradScaler for mixed precision training (None to disable)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast(device_type='cuda'):
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
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    # Step scheduler after epoch
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%), LR: {current_lr:.6f}')
    return avg_loss, accuracy


def test(model, test_loader, loss_fn, device, n_classes=100):
    """Test/validation function."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += loss_fn(output, target).item()
                
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy

    
def compute_and_save_singular_values(model, data_loader, device, filename, epoch, output_dir):
    """
    Compute and save singular values of penultimate layer activations.
    For WideResNet, we extract features before the final FC layer.
    """

    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_feats = []

    # Create a hook to extract penultimate features
    features = []
    def hook_fn(module, input, output):
        # Extract features before FC layer (after global pooling and flatten)
        features.append(input[0].detach())
    
    # Register hook on the final linear layer
    hook = model.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader, 1):
            data = data.to(device)
            features = []
            _ = model(data)  # Forward pass triggers the hook
            all_feats.append(features[0].cpu())

            if batch_idx % 50 == 0 or batch_idx == len(data_loader):
                print(f'Processed batch {batch_idx}/{len(data_loader)}')

    # Remove hook
    hook.remove()

    Phi = torch.cat(all_feats, dim=0)
    print(f"Collected hidden activations shape: {Phi.shape}")

    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")
        
    sv_path = os.path.join(output_dir, 'singular_values', f'{filename}_e{epoch}.pt')

    torch.save(S.cpu(), sv_path)
    print(f"Singular values saved to {sv_path}")
    
    return S, sv_path