import os
import re
import glob
import argparse
import torch

from pytorch_ood.model import WideResNet
from utilities import CIFAR100Loader, compute_and_save_singular_values

# -----------------------------
# Args (MATCHES YOUR CALL)
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
                    help='Directory containing WRN checkpoints')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Directory to save singular values')
parser.add_argument('--widen-factor', type=int, required=True)
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--samples', type=int, default=50000)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Find matching checkpoint
# -----------------------------
pattern = os.path.join(
    args.weights,
    f"wrn{args.depth}_{args.widen_factor}_job*_e{args.epoch}.pth"
)

matches = glob.glob(pattern)
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one match, found {len(matches)}:\n{matches}")

ckpt_path = matches[0]
ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]

print(f"Using checkpoint: {ckpt_path}")

# -----------------------------
# Model
# -----------------------------
model = WideResNet(
    num_classes=100,
    depth=args.depth,
    widen_factor=args.widen_factor,
    drop_rate=args.dropout
).to(device)

state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

# -----------------------------
# Data
# -----------------------------
loader = CIFAR100Loader(
    n_samples=args.samples,
    batch_size=args.batch_size,
)
train_loader, test_loader = loader.get_loaders()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Compute + save SVs
# -----------------------------
with torch.no_grad():
    S_test, test_path = compute_and_save_singular_values(
        model,
        test_loader,
        device,
        ckpt_name + "_test",
        args.epoch,
        args.output_dir
    )


print(f"SV(test)  saved to {test_path}")