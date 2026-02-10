import os
import re
import glob
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utilities import FCNN, compute_and_save_singular_values

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
                    help='Directory containing FCNN checkpoints')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Directory to save singular values')
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--epoch', type=int, default=2000,
                    help='Epoch number to process (e.g., 2000)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Find all matching checkpoints
# -----------------------------
pattern = os.path.join(
    args.weights,
    f"*_e{args.epoch}.pth"
)

matches = glob.glob(pattern)
if len(matches) == 0:
    raise RuntimeError(f"No checkpoints found matching pattern: {pattern}")

print(f"Found {len(matches)} checkpoints to process")

# -----------------------------
# Data (load once, use for all models)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # standard MNIST normalization
])

test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False)

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Process each checkpoint
# -----------------------------
for ckpt_path in sorted(matches):
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    
    # Extract width from filename
    # Expecting pattern like: w{width}_job{jobid}_e{epoch}.pth
    match = re.search(r'w(\d+)_job', ckpt_name)
    if not match:
        print(f"Skipping {ckpt_name}: couldn't extract width")
        continue
    
    width = int(match.group(1))
    
    print(f"\n{'='*60}")
    print(f"Processing: {ckpt_name}")
    print(f"Width: {width}")
    print(f"{'='*60}")
    
    # -----------------------------
    # Model
    # -----------------------------
    model = FCNN(hidden_dim=width).to(device)
    
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_name}: {e}")
        continue
    
    # -----------------------------
    # Compute + save SVs
    # -----------------------------
    with torch.no_grad():
        try:
            S_test, test_path = compute_and_save_singular_values(
                model,
                test_loader,
                device,
                ckpt_name + "_test",
                args.epoch,
                args.output_dir
            )
            print(f"SV(test) saved to {test_path}")
        except Exception as e:
            print(f"Error computing SVs for {ckpt_name}: {e}")
            continue
    
    # Free up memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*60}")
print(f"Completed processing {len(matches)} checkpoints")
print(f"{'='*60}")