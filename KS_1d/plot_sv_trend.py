import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
model_dir = 'N3'
epochs    = [50, 100]
yscale    = 'linear'

sv_dir      = f"/glade/derecho/scratch/tsatoperry/GAD/KS_1d/deep/{model_dir}/singular_values"
metrics_dir = f"/glade/derecho/scratch/tsatoperry/GAD/KS_1d/deep/{model_dir}/metrics"
filename    = f'sv_loss_{model_dir}.png'

# ── Load SV files ─────────────────────────────────────────────────────────────
def load_sv_files(directory, pattern):
    files = []
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            files.append((int(m.group(1)), os.path.join(directory, fname)))
    files.sort(key=lambda x: x[0])

    sv_arrays, widths = [], []
    for w, path in files:
        sv = torch.load(path, map_location="cpu", weights_only=True)
        sv[sv < 1e-16] = 1e-16
        sv_arrays.append(sv)
        widths.append(w)
    return sv_arrays, widths

# Load SVs for each epoch
sv_data = {}
for epoch in epochs:
    train_sv, train_w = load_sv_files(sv_dir, re.compile(rf"h_(\d+)_job(\d+)_e{epoch}\.pt$"))
    test_sv,  test_w  = load_sv_files(sv_dir, re.compile(rf"h_(\d+)_job(\d+)test_set_e{epoch}\.pt$"))
    sv_data[epoch] = dict(train_sv=train_sv, train_w=train_w, test_sv=test_sv, test_w=test_w)
    print(f"Epoch {epoch} — Train widths: {train_w}  |  Test widths: {test_w}")

# ── Load metrics JSON ─────────────────────────────────────────────────────────
train_samples, val_samples = None, None
loss_data = {epoch: {'width': [], 'train_losses': [], 'val_losses': []} for epoch in epochs}

for fname in os.listdir(metrics_dir):
    m = re.match(rf"h_(\d+)_job(\d+)\.json", fname)
    if not m:
        continue
    width    = int(m.group(1))
    filepath = os.path.join(metrics_dir, fname)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        train_samples = data.get("train_samples", train_samples)
        val_samples   = data.get("val_samples",   val_samples)
        for epoch in epochs:
            train_key = f'epoch{epoch}_train_loss'
            val_key   = f'epoch{epoch}_val_loss'
            if train_key in data and val_key in data:
                loss_data[epoch]['width'].append(width)
                loss_data[epoch]['train_losses'].append(data[train_key])
                loss_data[epoch]['val_losses'].append(data[val_key])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {fname}: {e}")

# Sort loss data by width
for epoch in epochs:
    if loss_data[epoch]['width']:
        idx = np.argsort(loss_data[epoch]['width'])
        loss_data[epoch]['width']        = np.array(loss_data[epoch]['width'])[idx]
        loss_data[epoch]['train_losses'] = np.array(loss_data[epoch]['train_losses'])[idx]
        loss_data[epoch]['val_losses']   = np.array(loss_data[epoch]['val_losses'])[idx]

print(f"Train samples: {train_samples:,}  |  Val samples: {val_samples:,}")

# ── Shared colormap across all epochs ────────────────────────────────────────
all_widths = sorted(set(
    w for epoch in epochs
    for w in sv_data[epoch]['train_w'] + sv_data[epoch]['test_w']
))
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(all_widths), vmax=max(all_widths))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
ax_sv50, ax_sv100, ax_loss = axes

fig.suptitle(
    f"KS_1D 6-layer MLP — {model_dir}  "
    f"(train samples: {train_samples:,}  |  val samples: {val_samples:,})",
    fontsize=14, fontweight='bold'
)

# ── Colorbar on the far left ──────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.01, pad=0.02)

cbar.set_label("Model Width")

# ── SV subplot helper ─────────────────────────────────────────────────────────
def plot_sv(ax, epoch, show_ylabel=True):
    d = sv_data[epoch]
    for sv, w in zip(d['train_sv'], d['train_w']):
        color   = cmap(norm(w))
        indices = np.arange(1, len(sv) + 1)
        ax.plot(indices, sv, '-', color=color, alpha=0.7)
        ax.plot(indices[-1], sv.mean(), 'o', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=0.5)

    for sv, w in zip(d['test_sv'], d['test_w']):
        color   = cmap(norm(w))
        indices = np.arange(1, len(sv) + 1)
        ax.plot(indices, sv, '--', color=color, alpha=0.7)
        ax.plot(indices[-1], sv.mean(), 's', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=0.5)

    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-',  label='Training SVs'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Testing SVs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Train mean SV'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Test mean SV'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    ax.set_title(f"Singular Values — Epoch {epoch}")
    ax.set_xlabel("Index")
    if show_ylabel:
        ax.set_ylabel("Singular Value")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

plot_sv(ax_sv50,  epochs[0], show_ylabel=True)
plot_sv(ax_sv100, epochs[1], show_ylabel=False)
# ── Right: Train vs Val Loss ──────────────────────────────────────────────────
epoch_colors = {epochs[0]: 'red', epochs[1]: 'blue'}

for epoch in epochs:
    ld        = loss_data[epoch]
    color     = epoch_colors[epoch]

    ax_loss.scatter(ld['width'], ld['val_losses'],   s=80, color=color, zorder=5,
                    edgecolors='black', linewidths=0.5, marker='o')
    ax_loss.scatter(ld['width'], ld['train_losses'], s=80, color=color, zorder=5,
                    edgecolors='black', linewidths=0.5, marker='s')
    ax_loss.plot(ld['width'], ld['val_losses'],   '--', linewidth=1.5, color=color, alpha=0.6)
    ax_loss.plot(ld['width'], ld['train_losses'], '-', linewidth=1.5, color=color, alpha=0.4)

loss_legend = [
    Line2D([0], [0], color='red',  linewidth=2, label=f'Epoch {epochs[0]}'),
    Line2D([0], [0], color='blue', linewidth=2, label=f'Epoch {epochs[1]}'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8,
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Test Loss'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Train Loss'),
]
ax_loss.legend(handles=loss_legend, fontsize=9, loc='best')
ax_loss.set_title("Train vs Test Loss by Width")
ax_loss.set_xlabel("Width", fontsize=13)
ax_loss.set_ylabel(f"Loss ({yscale})", fontsize=13)
ax_loss.set_xscale('log')
ax_loss.set_yscale(yscale)
ax_loss.grid(True, alpha=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved to {filename}")
plt.show()