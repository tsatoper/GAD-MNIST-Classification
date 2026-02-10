import os
import sys
import argparse
import json
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

from utilities import FCNN, train, test, compute_and_save_singular_values

parser = argparse.ArgumentParser()
parser.add_argument('--array-idx', type=int, default=0)
parser.add_argument('--job-num', type=str, default='__nojob__')
parser.add_argument('--output-dir', type=str, default='./default')
parser.add_argument('--learning-rate', type=float, default=1e-3)
args = parser.parse_args()


width = args.array_idx % 30 +1 #128 = 2^7 2^13 #2**args.array_idx 
filename = f"w{width}_job{args.job_num[:7]}"
num_epochs = 2000
save_at_this_epoch = [1, 500, 2000]
samples = 4000


batch_size = 2048
learning_rate = args.learning_rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()

model = FCNN(hidden_dim=width).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

print(f"Running with width = {width}")
print(f"Running with {samples} training samples")
print(f"Running with batch size = {batch_size}")
print(f"Running with learning rate = {learning_rate}")
print(f"Saving SV and validating at epochs: {save_at_this_epoch}")
print(f"Training on device: {device}")
print(f"Running with loss function = {loss_fn}")


num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Running with parameters = {num_parameters}")

param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size
print(f"Model size: {total_size / 1e6:.2f} MB "
        f"({total_size / (1024**3):.2f} GB)")

json_input = {
    'num_epochs': num_epochs,
    'width': width,
    'batch_size': batch_size,
    'samples': samples,
    'num_parameters': num_parameters,
    'loss_function': str(loss_fn),
    'learning_rate': learning_rate
}

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))   # standard MNIST normalization
])
train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = Subset(train_dataset_full, range(samples))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=False)

# SAVING
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)


# TRAINING
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch)

    if epoch in save_at_this_epoch:
        test_loss, test_acc = test(model, test_loader, loss_fn, device)

        json_input[f'epoch{epoch}_train_loss'] = train_loss
        json_input[f'epoch{epoch}_test_loss'] = test_loss
        json_input[f'epoch{epoch}_train_acc'] = train_acc
        json_input[f'epoch{epoch}_test_acc'] = test_acc
        
        # Save model weights
        torch.save(model.state_dict(), f'{args.output_dir}/weights/{filename}_e{epoch}.pth')
        print(f"Model weights saved at epoch {epoch} to {args.output_dir}/weights/{filename}_e{epoch}.pth")
        
        # Compute and save singular values
        S, sv_path = compute_and_save_singular_values(model, train_loader, device, filename, epoch, args.output_dir)
        print(f"Singular Values saved at epoch {epoch} to {sv_path}")

        json_input[f'epoch{epoch}_sv_max'] = float(S[0].cpu())
        json_input[f'epoch{epoch}_sv_min'] = float(S[-1].cpu())
        json_input[f'epoch{epoch}_sv_path'] = sv_path
    

with open(f'{args.output_dir}/metrics/{filename}.json', 'w') as f:
    json.dump(json_input, f, indent=4)
print(f"\nConfig and Metrics saved to '{args.output_dir}/metrics/{filename}.json'")

print("\n" + "="*50)
print("Training and singular value computation complete!")
print("="*50)