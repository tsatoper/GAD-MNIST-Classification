import os
import re
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

directory = "/glade/derecho/scratch/tsatoperry/GAD/models/mse/singular_values"

# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt
pattern = re.compile(r"hidden_dim(\d+)(?:_sv)?\.pt$")

files = []

# Collect all matching files
for fname in os.listdir(directory):
    m = pattern.match(fname)
    if m:
        hidden_dim = int(m.group(1))
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

# ----- Animation setup -----
fig, ax = plt.subplots(figsize=(6,4))

def update(frame_idx):
    ax.clear()
    sv = sv_arrays[frame_idx]
    hd = hidden_dims[frame_idx]

    ax.plot(sv, marker="o")
    ax.set_title(f"Singular Values (hidden_dim={hd})")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, max(max(arr) for arr in sv_arrays) * 1.1)
    return []

anim = FuncAnimation(
    fig,
    update,
    frames=len(sv_arrays),
    interval=300,   # ms per frame
    blit=False
)

writer = FFMpegWriter(fps=3)
anim.save("singular_values_animation.mp4", writer=writer)

print("Saved to singular_values_animation.mp4")
