import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import json

# Configuration
tag = 'lr1e-4'  
output_dir = '/glade/derecho/scratch/tsatoperry/GAD/TinyImageNet/models/' + tag
sv_directory = os.path.join(output_dir, 'singular_values')
metrics_directory = os.path.join(output_dir, 'metrics')

# Regex to match: w64_e10.pt, w128_e20.pt, etc.
# Captures width and epoch
training_dir = 'N3'

pattern = re.compile(rf"w(\d+)_e(\d+)\.pt$")
pattern = re.compile(rf"w(\d+)_{training_dir}_e(\d+)\.pt$")


files = []

# Collect all matching files
for fname in os.listdir(sv_directory):
    m = pattern.match(fname)
    if m:
        width = int(m.group(1))
        epoch = int(m.group(2))
        files.append((width, epoch, os.path.join(sv_directory, fname)))

# Sort by width, then by epoch
files.sort(key=lambda x: (x[0], x[1]))

# Group by width
from collections import defaultdict
width_groups = defaultdict(list)

for width, epoch, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    width_groups[width].append((epoch, sv))

# Load validation and train losses from metrics files
width_val_losses = {}
width_train_losses = {}
for width in width_groups.keys():
    metrics_file = os.path.join(metrics_directory, f'w{width}_{training_dir}.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            # Get epoch 100 validation and train loss (or whatever epoch you want)
            val_loss_key = 'epoch100_val_loss'
            train_loss_key = 'epoch100_train_loss'
            if val_loss_key in metrics and train_loss_key in metrics:
                width_val_losses[width] = metrics[val_loss_key]
                width_train_losses[width] = metrics[train_loss_key]
            else:
                print(f"Warning: Loss keys not found in {metrics_file}")
                width_val_losses[width] = None
                width_train_losses[width] = None
    else:
        print(f"Warning: Metrics file not found for width {width}: {metrics_file}")
        width_val_losses[width] = None
        width_train_losses[width] = None

# Filter out widths without validation loss data
valid_widths = [w for w in width_groups.keys() if width_val_losses.get(w) is not None]
valid_widths = [w for w in width_groups.keys() ]
if not valid_widths:
    print("Error: No validation loss data found!")
    exit(1)

# ----- Plot: One line per width, colored by width -----
fig, ax = plt.subplots(figsize=(10, 6))

# Create color gradient based on width (yellow for high, dark blue for low)
cmap = plt.cm.viridis  # Reversed Yellow-Green-Blue: yellow for high, dark blue for low
norm = plt.Normalize(vmin=min(valid_widths), vmax=max(valid_widths))

for width in valid_widths:
    epochs_and_svs = sorted(width_groups[width], key=lambda x: x[0])
    
    # Get the last epoch's singular values
    final_epoch, final_sv = epochs_and_svs[-1]
    
    # val_loss = width_val_losses[width]
    # train_loss = width_train_losses[width]
    color = cmap(norm(width))  # Color by width, not val_loss
    indices = np.arange(1, len(final_sv) + 1)
    
    # Create label with both train and validation loss
    # label = f"w={width} (train={train_loss:.2f}, val={val_loss:.2f})"
    
    ax.plot(indices, final_sv, color=color, alpha=0.7, linewidth=2)
    
    # Plot a dot at the last point
    max_idx = len(final_sv)
    max_val = final_sv[-1]
    ax.plot(max_idx, max_val, 'o', color=color, markersize=8, 
            markeredgecolor='black', markeredgewidth=0.5)

ax.set_title(f"Singular Values Comparison by Width - {training_dir}")
ax.set_xlabel("Index")
ax.set_ylabel("Singular Value")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc='best', fontsize=8)

# Add colorbar showing width
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Width")

plt.tight_layout()
output_filename = f"sv_by_width_{training_dir}.png"
plt.savefig(output_filename, dpi=150)
print(f"Saved: {output_filename}")
val_losses = [width_val_losses[w] for w in valid_widths]
print(f"Width range: {min(valid_widths)} to {max(valid_widths)}")
print(f"Validation loss range: {min(val_losses):.4f} to {max(val_losses):.4f}")

plt.show()
