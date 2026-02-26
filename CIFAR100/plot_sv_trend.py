import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'n_50000'
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

# ===== Load Training SV Data =====
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
train_sv_arrays = []
train_widths = []

for hd, path in files:
    sv = torch.load(path, map_location="cpu", weights_only=True)
    sv[sv<1e-8] = 1e-8
    train_sv_arrays.append(sv)
    train_widths.append(hd)

# ===== Load Test SV Data =====
pattern = re.compile(rf"wrn28_(\d+)_job(\d+)_e200_test_e{epoch}.pt$")
if model_dir == 'n_10000':
    pattern = re.compile(rf"wrn28_(\d+)_job(\d+)test_e{epoch}.pt$")

# pattern = re.compile(rf"wrn28_(\d+)_job(\d+)test_e{epoch}.pt$")

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
test_sv_arrays = []
test_widths = []

for hd, path in files:
    t = torch.load(path, map_location="cpu", weights_only=True)
    sv = t.numpy()
    sv[sv<1e-8] = 1e-8
    test_sv_arrays.append(sv)
    test_widths.append(hd)

print(f"Train widths: {train_widths}")
print(f"Test widths: {test_widths}")

# ===== Create Side-by-Side Plots =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Create color gradient with log normalization
cmap = plt.cm.viridis
all_widths = sorted(set(train_widths + test_widths))
norm = plt.Normalize(vmin=min(all_widths), vmax=max(all_widths))

# ----- LEFT PLOT: Singular Values -----
for sv, w in zip(train_sv_arrays, train_widths):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)
    ax1.plot(indices, sv, color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a dot at the mean
    ax1.plot(indices[-1], (sv.mean()), 'o', color=color, markersize=8, 
             markeredgecolor='black', markeredgewidth=0.5)

for sv, w in zip(test_sv_arrays, test_widths):
    color = cmap(norm(w))
    indices = np.arange(1, len(sv) + 1)
    ax1.plot(indices, sv, '--', color=color, label=f"width={w}", alpha=0.7)
    
    # Plot a square at the mean
    ax1.plot(indices[-1], sv.mean(), 's', color=color, markersize=8, 
             markeredgecolor='black', markeredgewidth=0.5)

ax1.set_title(f"Singular Values CIFAR100 ({model_dir})")
ax1.set_xlabel("Index")
ax1.set_ylabel("Singular Value")
ax1.set_xscale("linear")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3, which="both")

# Add legend for line styles and markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='Training sample SVs'),
    Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Testing sample SVs'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Training sample SV mean'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, 
           markeredgecolor='black', markeredgewidth=0.5, linestyle='None', label='Testing sample SV mean')
]
ax1.legend(handles=legend_elements, loc='best')

# ----- RIGHT PLOT: Loss Values -----
# Sort loss data by width
sorted_widths = sorted(loss_data.keys())
train_losses = [loss_data[w]['train_loss'] for w in sorted_widths]
test_losses = [loss_data[w]['test_loss'] for w in sorted_widths]

if sorted_widths:
    # Plot train losses
    for w, train_loss in zip(sorted_widths, train_losses):
        color = cmap(norm(w))
        ax2.scatter(w, train_loss, s=100, alpha=0.8, color=color, zorder=5, 
                   edgecolors='black', linewidths=0.5)
    ax2.plot(sorted_widths, train_losses, linestyle='-', linewidth=2, 
            color='gray', alpha=0.6, label='Training Loss')
    
    # Plot test losses
    for w, test_loss in zip(sorted_widths, test_losses):
        color = cmap(norm(w))
        ax2.scatter(w, test_loss, s=100, alpha=0.8, color=color, zorder=5, 
                   edgecolors='black', linewidths=0.5, marker='s')
    ax2.plot(sorted_widths, test_losses, linestyle='--', linewidth=2, 
            color='gray', alpha=0.6, label='Testing Loss')

ax2.set_title(f"Loss Values CIFAR100 ({model_dir})")
ax2.set_xlabel("Width")
ax2.set_ylabel("Loss")
ax2.set_xscale("linear")
ax2.set_yscale("linear")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

plt.tight_layout()
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved: {filename}")

plt.show()