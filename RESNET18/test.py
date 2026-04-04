"""
Compute mean singular value of activations (hidden layer) for each
ResNet-18k checkpoint at epoch 2000, using 1000 training samples.
Then plot mean SV vs model width.

Usage:
  python plot_activation_sv_vs_width.py
"""

import os, re, sys, glob
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Adjust these imports to match your paths ─────────────────────────────────
sys.path.insert(0, '/glade/derecho/scratch/tsatoperry/GAD/RESNET18')
from ResNet import make_resnet18k, cifar10_loader
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_DIR = '/glade/derecho/scratch/tsatoperry/GAD/RESNET18/models/s10k_n15/weights'
PATTERN     = 'w*_job5269273_e2000.pth'
N_SAMPLES   = 400
BATCH_SIZE  = 256
NUM_CLASSES = 10
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH   = f'activation_sv_vs_width{N_SAMPLES}.png'


def extract_width(path):
    m = re.search(r'w(\d+)_', os.path.basename(path))
    return int(m.group(1))


@torch.no_grad()
def get_activations(model, loader, device):
    model.eval()
    hidden = []
    for x, _ in loader:
        _, h = model(x.to(device), return_hidden=True)
        hidden.append(h.cpu().float().numpy())
    return np.concatenate(hidden, axis=0)   # (N, D)


def main():
    print(f"Device: {DEVICE}")

    loader = cifar10_loader(train=True, n_samples=N_SAMPLES,
                            batch_size=BATCH_SIZE, shuffle=False, seed=0)

    weight_files = sorted(glob.glob(os.path.join(WEIGHTS_DIR, PATTERN)),
                          key=extract_width)
    if not weight_files:
        raise FileNotFoundError(f"No files found: {WEIGHTS_DIR}/{PATTERN}")
    print(f"Found {len(weight_files)} checkpoints.\n")

    widths, sv_means = [], []

    for wf in weight_files:
        width = extract_width(wf)
        print(f"  width={width:4d} ...", end=' ', flush=True)

        model = make_resnet18k(k=width, num_classes=NUM_CLASSES).to(DEVICE)
        state = torch.load(wf, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)

        A = get_activations(model, loader, DEVICE)          # (1000, D)
        sv = np.linalg.svdvals(A)             # descending
        mean_sv = float(np.mean(sv))
        print(f"shape={A.shape}, mean_sv={mean_sv:.4f}")

        widths.append(width)
        sv_means.append(mean_sv)

    widths   = np.array(widths)
    sv_means = np.array(sv_means)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(widths*8, sv_means, color='steelblue', linewidth=2, zorder=2)
    ax.scatter(widths*8, sv_means, color='steelblue', s=60, zorder=3)
    ax.axvline(x=N_SAMPLES, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label=f'n_samples={N_SAMPLES}')
    ax.legend(fontsize=10)

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Activation Dimension Width (k)', fontsize=12)
    ax.set_ylabel('Mean Singular Value of Activations', fontsize=12)
    ax.set_title('ResNet-18k  |  s10k_n15  |  Epoch 2000\n'
                 f'Activation Mean SV  (n={N_SAMPLES} train samples)', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to '{SAVE_PATH}'")
    plt.show()


if __name__ == '__main__':
    main()