import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'lr1e-3'
epoch = 50
directory = f"/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/{model_dir}/singular_values"
filename = f'sv_trend_{model_dir}.png'

# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt
pattern = re.compile(r"hidden_dim(\d+)(?:_sv)?\_epoch2000_training.pt$")
pattern = re.compile(r"w(\d+)\_job(\d+)\_e50.pt$")
pattern = re.compile(rf"w(\d+)\_job(\d+)\_e{epoch}.pt$")

files = []

# Collect all matching files
for fname in os.listdir(directory):
    m = pattern.match(fname)
    if m:
        width = int(m.group(1))

        files.append((width, os.path.join(directory, fname)))

# Sort by hidden_dim numerically
files.sort(key=lambda x: x[0])

# Load all singular value arrays
sv_arrays = []
width = []

for hd, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    sv[sv<1e-16] = 1e-16
    sv_arrays.append(sv)
    width.append(hd)

# ----- Plotting all on same plot with log-normalized color gradient -----
fig, ax = plt.subplots(figsize=(10, 6))

# Create color gradient with log normalization
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(width), vmax=max(width))

for sv, w in zip(sv_arrays, width):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a dot at the max index (last point)
    max_idx = len(sv)
    max_val = sv[-1]
    ax.plot(max_idx, max_val, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)

ax.set_title("Singular Values Comparison")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")

# Add colorbar with log scale to show width gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Model Width")

plt.tight_layout()
plt.savefig(filename, dpi=150)
print(filename)

plt.show()