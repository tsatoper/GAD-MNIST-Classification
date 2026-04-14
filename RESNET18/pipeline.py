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
import sys
sys.path.append('/glade/derecho/scratch/tsatoperry/GAD/MNIST')
from utilities import train, test, save_hidden_activations

from ResNet import make_resnet18k, cifar10_loader
parser = argparse.ArgumentParser()
parser.add_argument('--array-idx', type=int, default=0)
parser.add_argument('--job-num', type=str, default='__nojob__')
parser.add_argument('--output-dir', type=str, default='./default')
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--n-samples', type=int, default=None, help='Number of training samples to use (None = use all)')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--noise', type=float, default=0.0)


args = parser.parse_args()
width = [4*i for i in range(1, 16+1)][args.array_idx]
filename = f"w{width}_job{args.job_num[:7]}"
num_epochs = args.epochs
samples = args.n_samples if args.n_samples is not None else 50000
learning_rate = args.learning_rate


save_at_this_epoch = [1]+list(range(50, num_epochs + 1, 50))
batch_size = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()

model = make_resnet18k(k=width, num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

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
    'learning_rate': learning_rate,
    'gamma': args.gamma,
    'noise': args.noise
}

train_loader = cifar10_loader(
    train=True,
    n_samples=samples,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    noise = args.noise,

)
test_loader = cifar10_loader(
    train=False,
    batch_size=batch_size,
    shuffle=False,
    noise = args.noise,
)

json_input['train_samples'] = len(train_loader.dataset)
json_input['test_samples'] = len(test_loader.dataset)

# SAVING
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'activations'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)


# TRAINING
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch)

    if epoch in save_at_this_epoch:
        test_loss, test_acc = test(model, test_loader, loss_fn, device)

        json_input[f'epoch{epoch}_train_loss'] = float(train_loss)
        json_input[f'epoch{epoch}_test_loss']  = float(test_loss)
        json_input[f'epoch{epoch}_train_acc']  = float(train_acc)
        json_input[f'epoch{epoch}_test_acc']   = float(test_acc)
        
        # Save model weights
        torch.save(model.state_dict(), f'{args.output_dir}/weights/{filename}_e{epoch}.pth')
        print(f"Model weights saved at epoch {epoch} to {args.output_dir}/weights/{filename}_e{epoch}.pth")
        
        acts_path, sv_path, sv_mean = save_hidden_activations(model, train_loader, test_loader, device, filename, epoch, args.output_dir)

        json_input[f'epoch{epoch}_acts_path'] = acts_path
        json_input[f'epoch{epoch}_sv_path'] = sv_path
        json_input[f'epoch{epoch}_train_sv_mean'] = float(sv_mean['train'])
        json_input[f'epoch{epoch}_test_sv_mean'] = float(sv_mean['test'])

    

with open(f'{args.output_dir}/metrics/{filename}.json', 'w') as f:
    json.dump(json_input, f, indent=4)
print(f"\nConfig and Metrics saved to '{args.output_dir}/metrics/{filename}.json'")

print("\n" + "="*50)
print("Training and singular value computation complete!")
print("="*50)