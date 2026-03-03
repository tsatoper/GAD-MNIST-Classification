import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# ====== assumes utilities.py is importable from working directory ======
from utilities import FCNN, mnist_loader

# ====== CONFIGURATION ======
plot_id        = 'N1_1e-4'
epoch          = 2000
n_samples_list = [1000]
yscale         = 'log'
save_path      = f'loss_sv_{plot_id}_e{epoch}.png'

WEIGHTS_DIR = f'/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/{plot_id}/weights'
METRICS_DIR = f'./models/{plot_id}/metrics'
DEVICE      = torch.device('cpu')

# ====== HELPERS ======

def get_hidden_activations(model, loader, device=DEVICE):
    """Run forward pass and collect post-ReLU hidden activations (N, hidden_dim)."""
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, hidden = model(x, return_hidden=True)
            all_hidden.append(hidden.cpu())
    return torch.cat(all_hidden, dim=0)


def compute_sv_mean(activations):
    _, S, _ = torch.linalg.svd(activations, full_matrices=False)
    return float(S.mean())


def parse_width_from_filename(filename):
    m = re.search(r'w(\d+)_', filename)
    return int(m.group(1)) if m else None


def parse_epoch_from_filename(filename):
    m = re.search(r'_e(\d+)\.pth$', filename)
    return int(m.group(1)) if m else None


# ====== BUILD WEIGHT FILE INDEX ======
print("Indexing weight files...")
weight_index = {}
for fname in os.listdir(WEIGHTS_DIR):
    if not fname.endswith('.pth'):
        continue
    w = parse_width_from_filename(fname)
    if w is None or w > 10000 or w < 10:
        continue
    e = parse_epoch_from_filename(fname)
    if e is not None and e == epoch:
        weight_index[w] = os.path.join(WEIGHTS_DIR, fname)

print(f"  Found {len(weight_index)} weight files for epoch {epoch}.")

# ====== LOAD METRICS ======
print("Loading metrics JSON files...")
metrics_data      = {}
n_samples_dataset = None

for fname in os.listdir(METRICS_DIR):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(METRICS_DIR, fname)
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        width = data['width']
        if width > 5000:
            continue
        if n_samples_dataset is None and 'samples' in data:
            n_samples_dataset = data['samples']
        tk = f'epoch{epoch}_train_loss'
        ek = f'epoch{epoch}_test_loss'
        if tk in data and ek in data:
            metrics_data[width] = {'train_loss': data[tk], 'test_loss': data[ek]}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Error reading {fname}: {e}")

print(f"  Loaded {len(metrics_data)} width entries.")

# ====== BUILD DATA LOADERS ======
print("Building MNIST loaders...")
train_loaders = {ns: mnist_loader(train=True,  n_samples=ns, batch_size=256) for ns in n_samples_list}
test_loaders  = {ns: mnist_loader(train=False, n_samples=ns, batch_size=256) for ns in n_samples_list}
print("  Done.")

# ====== COMPUTE SVD ======
print("Computing activations and singular values...")
sv_data = {}   # sv_data[width] = {ns: {'train': sv_mean, 'test': sv_mean}}

for width in sorted(weight_index):
    if width not in metrics_data:
        continue
    model = FCNN(input_dim=784, hidden_dim=width, output_dim=10).to(DEVICE)
    state = torch.load(weight_index[width], map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    sv_data[width] = {}
    for ns in n_samples_list:
        train_acts = get_hidden_activations(model, train_loaders[ns])
        test_acts  = get_hidden_activations(model, test_loaders[ns])
        sv_data[width][ns] = {
            'train': compute_sv_mean(train_acts),
            'test':  compute_sv_mean(test_acts),
        }
        sv_tr = [f"{sv_data[width][ns]['train']:.4f}" for ns in n_samples_list]
        sv_te = [f"{sv_data[width][ns]['test']:.4f}"  for ns in n_samples_list]
        print(f"  w={width:6d}  sv_train={sv_tr}  sv_test={sv_te}")
print("SVD computation complete.")

# ====== ASSEMBLE SORTED ARRAYS ======
widths       = np.array(sorted(sv_data))
test_losses  = np.array([metrics_data[w]['test_loss']  for w in widths])
train_losses = np.array([metrics_data[w]['train_loss'] for w in widths])
sv_train_means = {ns: np.array([sv_data[w][ns]['train'] for w in widths]) for ns in n_samples_list}
sv_test_means  = {ns: np.array([sv_data[w][ns]['test']  for w in widths]) for ns in n_samples_list}


sv_colors = [
    '#e06c75', '#61afef', '#98c379', '#e5c07b',
    '#c678dd', '#56b6c2', '#d19a66', '#abb2bf',
    '#be5046', '#2e86c1',
]

fig, ax1 = plt.subplots(figsize=(10, 5))

# Vertical lines
if n_samples_dataset is not None:
    ax1.axvline(x=n_samples_dataset, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
for ns, sc in zip(n_samples_list, sv_colors):
    ax1.axvline(x=ns, color=sc, linestyle='--', linewidth=1.2, alpha=0.7)

ax1.set_xlabel('Model Width', fontsize=11)
ax1.set_ylabel(f'Loss ({yscale})', fontsize=11, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
ax1.tick_params(axis='x', labelsize=8)
ax1.set_xscale('log')
ax1.set_yscale(yscale)
ax1.grid(True, alpha=0.3)

# Right axis: train and test mean SV per n_samples
ax2 = ax1.twinx()
for ns, sc in zip(n_samples_list, sv_colors):
    # Train SV: solid line, square markers
    ax2.plot(widths, sv_train_means[ns], linestyle='-.',          linewidth=2,   alpha=0.75, color=sc)
    ax2.scatter(widths, sv_train_means[ns], s=55, alpha=0.75, marker='s', zorder=5, color=sc)
    # Test SV: dash-dot-dot line, diamond markers
    ax2.plot(widths, sv_test_means[ns],  linestyle=(0,(3,1,1,1)), linewidth=2,   alpha=0.75, color=sc)
    ax2.scatter(widths, sv_test_means[ns],  s=55, alpha=0.75, marker='D', zorder=5, color=sc)

ax2.set_ylabel('Mean Singular Value', fontsize=11, color='purple')
ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
ax2.set_yscale('log')

# Legend
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2,   linestyle='-',           label='Test Loss'),
    Line2D([0], [0], color='gray', linewidth=2,   linestyle=':',           label='Train Loss'),
    Line2D([0], [0], color='gray', linewidth=2,   linestyle='-.',          label='Train Mean SV'),
    Line2D([0], [0], color='gray', linewidth=2,   linestyle=(0,(3,1,1,1)), label='Test Mean SV'),
    Line2D([0], [0], color='red',  linewidth=1.5, linestyle='--',          label=f'n_train={n_samples_dataset}'),
] + [
    Line2D([0], [0], color=sc, linewidth=2, linestyle='-',
           markersize=6, label=f'n={ns:,}  (vline + Mean SV)')
    for ns, sc in zip(n_samples_list, sv_colors)
]
ax1.legend(handles=legend_elements, fontsize=7, loc='lower right')

plt.title(f'{plot_id}  |  Epoch {epoch}', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved to "{save_path}"')
plt.show()