import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'n_500'
epoch = 200
directory = f'/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/{model_dir}/depth28/singular_values/'

filename = f'ugly_{model_dir}.png'

# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt
pattern = re.compile(rf"wrn28_(\d+)\_job(\d+)\_e{epoch}.pt$")

files = []

# Collect all matching files
for fname in os.listdir(directory):
    m = pattern.match(fname)
    if m:
        width = int(m.group(1))
        if width % 2 == 1:
            continue
        files.append((width, os.path.join(directory, fname)))

# Sort by hidden_dim numerically
files.sort(key=lambda x: x[0])

# Load all singular value arrays
sv_arrays = []
width = []

for hd, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    sv[sv<1e-16] = 1e-8
    sv_arrays.append(sv)
    width.append(hd)

# ----- Plotting all on same plot with more distinct colors -----
fig, ax = plt.subplots(figsize=(10, 6))

# Use inferno colormap
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(width), vmax=max(width))

for sv, w in zip(sv_arrays, width):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a dot at the max index (last point)
    max_idx = len(sv)
    max_val = sv[-1]
    ax.plot(max_idx, max_val, 'o', color=color, markersize=7, markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(indices[-1], sv.mean(), 'o', color=color, markersize=7, markeredgecolor='black', markeredgewidth=0.5)


ax.set_title(f"Singular Values CIFAR100 image classification ({model_dir})")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")

# Add colorbar with width * 64 labels
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Model Width")
# Set colorbar ticks to show width values multiplied by 64
cbar.set_ticks([w for w in width])
cbar.set_ticklabels([str(w * 64) for w in width])

# Add legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, 
           markeredgecolor='black', markeredgewidth=0.5, label='train samples'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=7, 
           markeredgecolor='black', markeredgewidth=0.5, label='test samples')
]

plt.tight_layout()

plt.show()


pattern = re.compile(rf"wrn28_(\d+)\_job(\d+)_e200_test_e{epoch}.pt$")

files = []

# Collect all matching files
for fname in os.listdir(directory):
    m = pattern.match(fname)
    if m:
        width = int(m.group(1))
        if width % 2 == 1:
            continue
        
        files.append((width, os.path.join(directory, fname)))

# Sort by hidden_dim numerically
files.sort(key=lambda x: x[0])

# Load all singular value arrays
sv_arrays = []
width = []

for hd, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    sv[sv<1e-16] = 1e-8
    sv_arrays.append(sv)
    width.append(hd)
print(width)

# ----- Plotting all on same plot with log-normalized color gradient -----

for sv, w in zip(sv_arrays, width):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, '--', color=color, label=f"width={w}", alpha=0.7)

    # Plot a dot at the max index (last point)
    max_idx = len(sv)
    max_val = sv[-1]
    ax.plot(max_idx, max_val, 's', color=color, markersize=7, markeredgecolor='black', markeredgewidth=0.5)
    # ---- MEAN LINE ----
    ax.plot(indices[-1], sv.mean(), 's', color=color, markersize=7, markeredgecolor='black', markeredgewidth=0.5)


ax.legend(handles=legend_elements, loc='best')

ax.set_title(f"Singular Values CIFAR100 image classification ({model_dir})")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")


plt.tight_layout()
plt.savefig(filename, dpi=150)
print(filename)

plt.show()