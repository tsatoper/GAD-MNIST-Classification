import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.amp import autocast, GradScaler

from utilities import CIFAR100Loader, train, test, compute_and_save_singular_values
from pytorch_ood.model import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument('--array-idx', type=int, default=0)
parser.add_argument('--job-num', type=str, default='__nojob__')
parser.add_argument('--output-dir', type=str, default='./default')
parser.add_argument('--depth', type=int, default=28, help='WRN depth (must be 6n+4)')
parser.add_argument('--widen-factor', type=int, default=10, help='WRN width multiplier')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--samples', type=int, default=50000, help='Number of CIFAR100 samples')
parser.add_argument('--use-mixed-precision', action='store_true', help='Enable mixed precision training')
args = parser.parse_args()


widen_factor = args.widen_factor
depth = args.depth
filename = f"wrn{depth}_{widen_factor}_job{args.job_num[:7]}"
num_epochs = 200
save_at_this_epoch = [1, 50, 100, 150, 200]
samples = args.samples  # Full CIFAR-100 training set


batch_size = 128
learning_rate = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()

# Initialize mixed precision scaler (optional, not in original paper)
scaler = GradScaler() if (torch.cuda.is_available() and args.use_mixed_precision) else None

model = WideResNet(
    num_classes=100,
    depth=depth,
    widen_factor=widen_factor,
    drop_rate=args.dropout
).to(device)

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

# Scheduler - MultiStepLR as per WRN paper (no warmup in original)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[60, 120, 160],
    gamma=0.2
)

print("="*60)
print("Wide Residual Network Training - CIFAR-100")
print("="*60)
print(f"Model: WRN-{depth}-{widen_factor}")
print(f"Dataset: CIFAR-100 ({samples} training samples)")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Dropout: {args.dropout}")
print(f"Mixed precision: {'Enabled' if scaler else 'Disabled (Original Paper)'}")
print(f"Device: {device}")
print(f"Loss function: {loss_fn}")
print(f"Optimizer: SGD (momentum=0.9, weight_decay=5e-4, nesterov=True)")
print(f"Scheduler: MultiStepLR (milestones=[60, 120, 160], gamma=0.2)")
print(f"Total epochs: {num_epochs}")
print(f"Checkpoint epochs: {save_at_this_epoch}")

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_parameters:,} ({num_parameters / 1e6:.2f}M)")

param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size
print(f"Model size: {total_size / 1e6:.2f} MB ({total_size / (1024**3):.4f} GB)")
print("="*60)

json_input = {
    'num_epochs': num_epochs,
    'depth': depth,
    'widen_factor': widen_factor,
    'dropout': args.dropout,
    'batch_size': batch_size,
    'samples': samples,
    'num_parameters': num_parameters,
    'loss_function': str(loss_fn),
    'learning_rate': learning_rate,
    'optimizer': 'SGD_Nesterov',
    'scheduler': 'MultiStepLR',
    'lr_milestones': [60, 120, 160],
    'lr_gamma': 0.2,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'mixed_precision': scaler is not None
}

loader = CIFAR100Loader(
    n_samples=args.samples,      # e.g. 5-shot per class
    batch_size=batch_size,
)

train_loader, test_loader = loader.get_loaders()

# SAVING
args.output_dir = os.path.join(args.output_dir, f'depth{depth}')
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)

# TRAINING
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(
        model, train_loader, loss_fn, optimizer, scheduler, 
        device, epoch, scaler=scaler, n_classes=100
    )

    if epoch in save_at_this_epoch:
        test_loss, test_acc = test(model, test_loader, loss_fn, device, n_classes=100)

        json_input[f'epoch{epoch}_train_loss'] = train_loss
        json_input[f'epoch{epoch}_test_loss'] = test_loss
        json_input[f'epoch{epoch}_train_acc'] = train_acc
        json_input[f'epoch{epoch}_test_acc'] = test_acc
        json_input[f'epoch{epoch}_learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Save model weights
        torch.save(model.state_dict(), f'{args.output_dir}/weights/{filename}_e{epoch}.pth')
        print(f"Model weights saved at epoch {epoch} to {args.output_dir}/weights/{filename}_e{epoch}.pth")
        
        # Compute and save singular values
        S, sv_path = compute_and_save_singular_values(model, test_loader, device, filename+'test', epoch, args.output_dir)
        S, sv_path = compute_and_save_singular_values(model, train_loader, device, filename, epoch, args.output_dir)
        print(f"Singular Values saved at epoch {epoch} to {sv_path}")

        json_input[f'epoch{epoch}_sv_max'] = float(S[0].cpu())
        json_input[f'epoch{epoch}_sv_min'] = float(S[-1].cpu())
        json_input[f'epoch{epoch}_sv_path'] = sv_path
    

with open(f'{args.output_dir}/metrics/{filename}.json', 'w') as f:
    json.dump(json_input, f, indent=4)
print(f"\nConfig and Metrics saved to '{args.output_dir}/metrics/{filename}.json'")

print("\n" + "="*60)
print("Training and singular value computation complete!")
print("="*60)