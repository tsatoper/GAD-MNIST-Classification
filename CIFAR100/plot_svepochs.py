import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
model_dir = 'n_10000'
epochs    = [1, 50, 200]
directory = f'/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/{model_dir}/depth28/singular_values/'
filename  = f'sv_epochs_{model_dir}.png'

# ── Load SV files for a given epoch ──────────────────────────────────────────
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
        sv = sv.numpy() if hasattr(sv, 'numpy') else sv
        sv[sv < 1e-8] = 1e-8
        sv_arrays.append(sv)
        widths.append(w)
    return sv_arrays, widths

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(epochs), figsize=(26, 5), sharey=True)
fig.suptitle(f"Singular Values — CIFAR100 WRN-28 ({model_dir})", fontsize=14, fontweight='bold')

# Collect all widths across all epochs to build a consistent colormap
all_widths_global = set()
for epoch in epochs:
    _, tw = load_sv_files(directory, re.compile(rf"wrn28_(\d+)_job(\d+)_e{epoch}\.pt$"))
    if model_dir == 'n_10000':
        _, xw = load_sv_files(directory, re.compile(rf"wrn28_(\d+)_job(\d+)test_e{epoch}.pt$"))
    else:
        _, xw = load_sv_files(directory, re.compile(rf"wrn28_(\d+)_job(\d+)_e200_test_e{epoch}.pt$"))
    all_widths_global.update(tw + xw)

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(all_widths_global), vmax=max(all_widths_global))

for ax, epoch in zip(axes, epochs):
    train_sv_arrays, train_widths = load_sv_files(
        directory, re.compile(rf"wrn28_(\d+)_job(\d+)_e{epoch}\.pt$")
    )
    if model_dir == 'n_10000':
        test_sv_arrays, test_widths = load_sv_files(
            directory, re.compile(rf"wrn28_(\d+)_job(\d+)test_e{epoch}\.pt$")
        )   
    else:
        test_sv_arrays, test_widths = load_sv_files(
            directory, re.compile(rf"wrn28_(\d+)_job(\d+)_e200_test_e{epoch}\.pt$")
        )

    for sv, w in zip(train_sv_arrays, train_widths):
        color   = cmap(norm(w))
        indices = np.arange(1, len(sv) + 1)
        ax.plot(indices, sv, '-', color=color, alpha=0.7)
        ax.plot(indices[-1], sv.mean(), 'o', color=color, markersize=7,
                markeredgecolor='black', markeredgewidth=0.5)

    for sv, w in zip(test_sv_arrays, test_widths):
        color   = cmap(norm(w))
        indices = np.arange(1, len(sv) + 1)
        ax.plot(indices, sv, '--', color=color, alpha=0.7)
        ax.plot(indices[-1], sv.mean(), 's', color=color, markersize=7,
                markeredgecolor='black', markeredgewidth=0.5)

    ax.set_title(f"Epoch {epoch}", fontsize=11)
    ax.set_xlabel("Index")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

axes[0].set_ylabel("Singular Value")

# Shared legend on the rightmost axis
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, linestyle='-',  label='Train SVs'),
    Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Test SVs'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7,
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Train mean'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=7,
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Test mean'),
]
axes[-1].legend(handles=legend_elements, loc='best', fontsize=9)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.01, pad=0.02)
cbar.set_label("Model Width")

plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved: {filename}")
plt.show()