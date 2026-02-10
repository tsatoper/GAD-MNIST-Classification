import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'n_5000'
epoch = 200
directory = f'/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/{model_dir}/depth28/singular_values/'
metrics_dir = f'/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/{model_dir}/depth28/metrics/'

filename = f'sv_trend_{model_dir}.png'

# ===== Load Loss Data from Metrics =====
loss_data = {}  # {width: {'train_loss': value, 'test_loss': value}}

if os.path.exists(metrics_dir):
    for fname in os.listdir(metrics_dir):
        match = re.match(r'wrn28_(\d+)_job(\d+)\.json', fname)
        if match:
            width = int(match.group(1))
            filepath = os.path.join(metrics_dir, fname)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                train_key = f'epoch{epoch}_train_loss'
                test_key = f'epoch{epoch}_test_loss'
                
                if train_key in data and test_key in data:
                    loss_data[width] = {
                        'train_loss': data[train_key],
                        'test_loss': data[test_key]
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {fname}: {e}")
                continue
else:
    print(f"Warning: Metrics directory '{metrics_dir}' not found")

# ===== Load SV Data =====
# Regex to catch both hidden_dim1024_sv.pt AND hidden_dim4096.pt
pattern = re.compile(rf"wrn28_(\d+)\_job(\d+)\_e{epoch}.pt$")

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
    sv[sv<1e-16] = 1e-8
    sv_arrays.append(sv)
    width.append(hd)

# ----- Plotting all on same plot with log-normalized color gradient -----
fig, ax = plt.subplots(figsize=(12, 6))

# Create twin axis for loss values
ax2 = ax.twinx()

# Create color gradient with log normalization
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(width), vmax=max(width))

for sv, w in zip(sv_arrays, width):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a dot at the mean (removed min value marker)
    mean_val = sv.mean()
    min_val = sv[-1]  # Last value (minimum)
    ax.plot(indices[-1], mean_val, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    # Plot loss values as stars on twin axis
    if w in loss_data:
        train_loss = loss_data[w]['train_loss']
        # Plot star at same x position as mean
        ax2.plot(indices[-1], train_loss, '*', color=color, markersize=12, 
                markeredgecolor='black', markeredgewidth=0.5)

ax.set_title(f"Singular Values CIFAR100 image classification ({model_dir})")
ax.set_xlabel("Index")
ax.set_ylabel("Singular Value")
ax2.set_ylabel("Loss Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")

# Add colorbar with log scale to show width gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Model Width")

plt.tight_layout()

plt.show()


# ===== Load Test SV Data =====
pattern = re.compile(rf"wrn28_(\d+)_job(\d+)_e200_test_e{epoch}.pt$")

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
    sv[sv<1e-16] = 1e-8
    sv_arrays.append(sv)
    width.append(hd)
print(width)

# ----- Plotting all on same plot with log-normalized color gradient -----

for sv, w in zip(sv_arrays, width):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)  # Start from 1 for log scale
    ax.plot(indices, sv, '--', color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a square at the mean (removed min value marker)
    mean_val = sv.mean()
    ax.plot(indices[-1], mean_val, 's', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    # Plot loss values as stars on twin axis
    if w in loss_data:
        test_loss = loss_data[w]['test_loss']
        # Plot star at same x position as mean
        ax2.plot(indices[-1], test_loss, '*', color=color, markersize=12, 
                markeredgecolor='red', markeredgewidth=0.5)

# Add legend for line styles and markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='Training sample SVs'),
    Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Testing sample SVs'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Training sample SV mean'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Testing sample SV mean'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, 
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Training loss'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, 
           markeredgecolor='red', markeredgewidth=0.5, linestyle='None', label='Testing loss')
]

ax.legend(handles=legend_elements, loc='best')

ax.set_title(f"Singular Values CIFAR100 image classification ({model_dir})")
ax.set_xlabel("Index")
ax.set_ylabel("Singular Value")
ax2.set_ylabel("Loss Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")


plt.tight_layout()
plt.savefig(filename, dpi=150)
print(filename)

plt.show()