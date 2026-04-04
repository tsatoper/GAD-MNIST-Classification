"""
Train width-200, depth-3 ReLU MLP on MNIST with MSE loss.

Configuration:
  - Architecture : 3-hidden-layer MLP, each layer width 200, ReLU activations
  - Loss         : MSE (targets are one-hot encoded)
  - Optimizer    : AdamW, lr=0.001, weight_decay=0.5
  - Batch size   : 200
  - Epochs       : 3000
  - Init scale α : 9  (weights scaled so ||w|| = α · ||w0||)

Imports data loading, training, and evaluation utilities from mnist_utils.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.append('/glade/derecho/scratch/tsatoperry/GAD/MNIST')
from utilities import mnist_loader, train, test


# ── Hyper-parameters ─────────────────────────────────────────────────────────
WIDTH        = 200
DEPTH        = 3          # number of hidden layers
ALPHA        = 9.0        # initialization scaling factor α ≡ w / w0
LR           = 1e-3
WEIGHT_DECAY = 0.5
BATCH_SIZE   = 200
EPOCHS       = 5000
SAMPLES      = 1000
NUM_CLASSES  = 10
INPUT_DIM    = 784        # 28×28 flattened
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Fully-connected ReLU MLP with `depth` hidden layers of size `width`."""

    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten (N,1,28,28) → (N,784)
        return self.net(x)


# ── Alpha initialization ──────────────────────────────────────────────────────
def get_weight_norm(model: nn.Module) -> float:
    """Global L2 norm over all weight matrices (biases excluded)."""
    total = sum(
        p.data.norm(2).item() ** 2
        for name, p in model.named_parameters()
        if "weight" in name
    )
    return total ** 0.5


def scale_weights_by_alpha(model: nn.Module, alpha: float) -> None:
    """
    Multiply every weight matrix by α so that the global weight norm
    becomes α · w0, where w0 is the norm after standard initialization.
    Biases are left unchanged.
    """
    w0 = get_weight_norm(model)
    if w0 == 0:
        raise ValueError("Weight norm before scaling is 0.")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                p.mul_(alpha)
    w_after = get_weight_norm(model)
    print(f"Weight norm before scaling : {w0:.4f}")
    print(f"Weight norm after  scaling : {w_after:.4f}  (α = {w_after / w0:.4f})")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(f"Device      : {DEVICE}")
    print(f"Architecture: MLP  input={INPUT_DIM}  hidden={WIDTH}×{DEPTH}  output={NUM_CLASSES}")
    print(f"Alpha       : {ALPHA}")
    print(f"Optimizer   : AdamW  lr={LR}  weight_decay={WEIGHT_DECAY}")
    print(f"Batch size  : {BATCH_SIZE}   Epochs : {EPOCHS}\n")

    # ── Model & alpha-scaled initialization ───────────────────────────────────
    model = MLP(INPUT_DIM, WIDTH, DEPTH, NUM_CLASSES).to(DEVICE)
    scale_weights_by_alpha(model, ALPHA)
    print()

    # ── Data loaders (from mnist_utils) ───────────────────────────────────────
    train_loader = mnist_loader(train=True,  batch_size=BATCH_SIZE, shuffle=True, n_samples=SAMPLES)
    test_loader  = mnist_loader(train=False, batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimizer & constant-LR scheduler ─────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)  # constant LR

    # ── Training loop (train / test from mnist_utils) ─────────────────────────
    log_interval = max(1, EPOCHS // 20)   # log ~20 times

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch, n_classes=NUM_CLASSES)
        total_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
        print(f"Total weight L2 norm: {total_norm:.4f}")
        if epoch % log_interval == 0 or epoch == EPOCHS:
            test(model, test_loader, criterion, DEVICE, n_classes=NUM_CLASSES)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt = "mnist_mlp_alpha9.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\nModel saved to {ckpt}")


if __name__ == "__main__":
    main()