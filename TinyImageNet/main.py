import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

# Import your new models and utilities
from utilities import SimpleWideCNN, compute_and_save_singular_values

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--output-dir', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/models/default')
parser.add_argument('--data-dir', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/tinyimagenet/tiny-imagenet-200')
parser.add_argument('--train-suffix', type=str, default='train_N1')
parser.add_argument('--width', type=int, default=64, help='Width of the CNN')
args = parser.parse_args()


# Configuration
num_epochs = 1000
batch_size = 16
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = SimpleWideCNN(width=args.width).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_parameters:,} ({num_parameters/1e6:.2f}M)")


# -----------------------------
# Data loading
# -----------------------------
traindir = os.path.join(args.data_dir, args.train_suffix)
valdir = os.path.join(args.data_dir, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
# subset_size = 1000
# train_subset = Subset(train_dataset, range(subset_size))
# train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


# -----------------------------
# Output directories
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)


# -----------------------------
# Training loop
# -----------------------------
json_input = {
    'num_epochs': num_epochs,
    'width': args.width,
    'batch_size': batch_size,
    'num_parameters': num_parameters,
    'learning_rate': learning_rate,
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'job_idx': args.job_idx
}

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, target)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = logits.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total

    print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    json_input[f'epoch{epoch}_train_loss'] = train_loss
    json_input[f'epoch{epoch}_train_acc'] = train_acc

    # Validate and compute singular values every 10 epochs
    if epoch % 100 == 0:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, feats = model(data, return_features=True)
                loss = loss_fn(logits, target)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        json_input[f'epoch{epoch}_val_loss'] = val_loss
        json_input[f'epoch{epoch}_val_acc'] = val_acc

        # Save weights
        torch.save(model.state_dict(), f'{args.output_dir}/weights/w{args.width}_epoch{epoch}.pth')

        # Compute and save singular values
        model_name = f'w{args.width}'
        S = compute_and_save_singular_values(model, train_loader, device, model_name, epoch, args.output_dir)

# Save metrics
with open(f'{args.output_dir}/metrics/w{args.width}.json', 'w') as f:
    json.dump(json_input, f, indent=4)

print(f"\nMetrics saved to '{args.output_dir}/metrics/w{args.width}.json'")
