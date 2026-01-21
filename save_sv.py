import argparse
import os
import torch
import torch.nn as nn
from utilities import FCNN, compute_and_save_singular_values
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-idx",
    type=int,
    required=True,
    help="Index into hidden_dim_list to choose hidden dimension"
)
args = parser.parse_args()

# Construct hidden dimension list
hidden_dim_list = (
    [i for i in range(1, 30 + 1)] +  # 30
    [2**i for i in range(5, 18+1)]   # 44
)

hidden_dim = int(hidden_dim_list[args.job_idx])
print(f"Selected hidden_dim = {hidden_dim} (from job_idx {args.job_idx})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Prepare test data (only need to do this once)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset_full, batch_size=256, shuffle=False)

# Base directory for saving singular values
output_dir = "/glade/derecho/scratch/tsatoperry/GAD/models/omni"
os.makedirs(os.path.join(output_dir, 'singular_values'), exist_ok=True)

# Process both epochs
epochs = [500, 1000]

for epoch in epochs:
    print(f"\n{'='*60}")
    print(f"Processing epoch {epoch}")
    print(f"{'='*60}")
    
    weights_path = (
        f"/glade/derecho/scratch/tsatoperry/GAD/models/omni/weights/"
        f"hidden_dim{hidden_dim}_epoch{epoch}.pth"
    )
    
    # Load model
    model = FCNN(input_dim=784, hidden_dim=hidden_dim, output_dim=10)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {weights_path}")
    print(f"Model architecture: input_dim=784, hidden_dim={hidden_dim}, output_dim=10")
    
    # Compute and save singular values using the utility function
    S = compute_and_save_singular_values(model, test_loader, device, hidden_dim, epoch, output_dir)

print(f"\n{'='*60}")
print("All epochs processed successfully!")
print(f"{'='*60}")