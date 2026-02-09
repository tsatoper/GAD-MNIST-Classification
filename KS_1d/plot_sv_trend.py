import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'gobig'
epoch = 100
directory = f"/glade/derecho/scratch/tsatoperry/GAD/KS_1d/AR_MLP_deep/{model_dir}/singular_values"

filename = f'sv_trend_{model_dir}.png'

# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt

pattern = re.compile(rf"h_(\d+)\_job(\d+)_e{epoch}.pt")

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
    indices = np.arange(1, len(sv) + 1)

    # main spectrum
    ax.plot(indices, sv, color=color, alpha=0.7)

    # endpoint marker
    ax.plot(len(sv), sv[-1], 'o',
            color=color, markersize=8,
            markeredgecolor='black', markeredgewidth=0.5)

    # ---- MEAN LINE ----
    ax.hlines(
        y=sv.mean(),
        xmin=indices[0],
        xmax=indices[-1],
        colors=color,
        linestyles='-',
        linewidth=2,
        alpha=0.9
    )

ax.set_title("Singular Values Comparison. KS_1D 6 layer MLP")
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