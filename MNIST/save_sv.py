import argparse
import os
import torch
import torch.nn as nn
from utilities import FCNN, compute_and_save_singular_values
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-idx",
    type=int,
    required=True,
    help="Index into hidden_dim_list to choose hidden dimension"
)
parser.add_argument(
    "--max-samples",
    type=int,
    default=10000,
    help="Maximum number of training samples to use for SVD"
)
parser.add_argument(
    "--model-dir",
    type=str,
    default='/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/mse',
    help="Which model to compute svd on"
)
args = parser.parse_args()

# Construct hidden dimension list
hidden_dim_list = [2**i for i in range(0, 19)]

hidden_dim = int(hidden_dim_list[args.job_idx])
print(f"Selected hidden_dim = {hidden_dim} (from job_idx {args.job_idx})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Prepare TRAINING data with subsampling
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Subsample training data to reduce memory
subset_size = min(args.max_samples, len(train_dataset_full))
indices = np.random.RandomState(42).choice(len(train_dataset_full), subset_size, replace=False)
train_dataset = Subset(train_dataset_full, indices)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

print(f"Using {subset_size} training samples (subsampled from {len(train_dataset_full)})")

# Base directory for saving singular values
os.makedirs(os.path.join(args.model_dir, 'singular_values'), exist_ok=True)

# Process epoch 2000 (based on your file naming)
epoch = 2000

print(f"\n{'='*60}")
print(f"Processing epoch {epoch}")
print(f"{'='*60}")

weights_path = (
    f"/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/mse/weights/"
    f"mnist_hidden_dim{hidden_dim}_epochs{epoch}.pth"
)

# Load model
model = FCNN(input_dim=784, hidden_dim=hidden_dim, output_dim=10)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

print(f"Model loaded from {weights_path}")
print(f"Model architecture: input_dim=784, hidden_dim={hidden_dim}, output_dim=10")

# Estimate memory usage
matrix_size_gb = (subset_size * hidden_dim * 4) / (1024**3)
print(f"Estimated SVD matrix size: {matrix_size_gb:.2f} GB")

# Compute and save singular values using the TRAINING dataset
S = compute_and_save_singular_values(model, train_loader, device, hidden_dim, epoch, args.model_dir)

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")