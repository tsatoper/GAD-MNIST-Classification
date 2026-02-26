import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PLOT_IDS = [
    'N1_1e-3',
    'N1_1e-4',
    'N2_1e-3',
    'N2_1e-4',
    'N3_1e-3',
    'N3_1e-4'
]

epoch    = 550
base_dir = '/glade/derecho/scratch/tsatoperry/GAD/MNIST/models'
filename = 'sv_trend_mnist.png'

# ===== Grid layout: one subplot per model =====
n_models = len(PLOT_IDS)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

for col, model_dir in enumerate(PLOT_IDS):
    ax = axes[col]
    directory = f'{base_dir}/{model_dir}/singular_values/'

    # ===== Load SV Data =====
    # Each .pt file is a dict with 'train' and 'test' keys
    train_sv_arrays, train_widths = [], []
    test_sv_arrays,  test_widths  = [], []

    if os.path.exists(directory):
        pattern = re.compile(rf"w(\d+)_job(\d+)_e{epoch}\.pt$")
        files = []
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                files.append((int(m.group(1)), os.path.join(directory, fname)))
        files.sort(key=lambda x: x[0])

        for width, path in files:
            data = torch.load(path, map_location="cpu", weights_only=True)
            for key, sv_list, w_list in [('train', train_sv_arrays, train_widths),
                                          ('test',  test_sv_arrays,  test_widths)]:
                if key in data:
                    sv = data[key]
                    if isinstance(sv, torch.Tensor):
                        sv = sv.numpy()
                    sv = np.array(sv, dtype=float)
                    sv[sv < 1e-5] = 1e-5
                    sv_list.append(sv)
                    w_list.append(width)
    else:
        print(f"[{model_dir}] Warning: directory not found: {directory}")

    print(f"[{model_dir}] Train widths: {train_widths}")
    print(f"[{model_dir}] Test  widths: {test_widths}")

    # ===== Color normalization =====
    cmap = plt.cm.viridis
    all_widths = sorted(set(train_widths + test_widths)) or [1]
    norm = plt.Normalize(vmin=min(all_widths), vmax=max(all_widths))

    # ===== Plot =====
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

    ax.set_title(f"MNIST — {model_dir}", fontsize=11)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # Legend on first subplot only
    if col == 0:
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, ls='-',  label='Train SVs'),
            Line2D([0], [0], color='gray', lw=2, ls='--', label='Test SVs'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                   ls='None', label='Train mean'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                   ls='None', label='Test mean'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)

    # Colorbar on last subplot
    if col == n_models - 1 and all_widths:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Width')

plt.suptitle(f"MNIST Singular Values (epoch {epoch})", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved: {filename}")
plt.show()