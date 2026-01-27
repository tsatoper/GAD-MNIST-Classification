import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'omni'
directory = f"/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/{model_dir}/singular_values"

# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt
pattern = re.compile(r"hidden_dim(\d+)(?:_sv)?\.pt$")

files = []

# Collect all matching files
for fname in os.listdir(directory):
    m = pattern.match(fname)
    if m:
        hidden_dim = int(m.group(1))
        if hidden_dim > 50:
            continue
        files.append((hidden_dim, os.path.join(directory, fname)))

# Sort by hidden_dim numerically
files.sort(key=lambda x: x[0])

# Load all singular value arrays
sv_arrays = []
hidden_dims = []

for hd, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    sv_arrays.append(sv)
    hidden_dims.append(hd)

# ----- Plotting all on same plot with log-normalized color gradient -----
fig, ax = plt.subplots(figsize=(10, 6))

# Create color gradient with log normalization
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))

for sv, hd in zip(sv_arrays, hidden_dims):
    color = cmap(norm(hd))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, color=color, label=f"hidden_dim={hd}", alpha=0.7)
    
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

# Add colorbar with log scale to show hidden_dim gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Hidden Dimension")

plt.tight_layout()
plt.savefig(f"sm_sv_trend_{model_dir}.png", dpi=150)
print(f"sm_sv_trend_{model_dir}.png")

plt.show()