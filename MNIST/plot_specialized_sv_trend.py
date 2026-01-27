import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/test"

# Collect all test subdirectories
test_dirs = []
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path) and item.isdigit():
        test_dirs.append((int(item), item_path))

# Sort by test number
test_dirs.sort(key=lambda x: x[0])

# Load singular values and test losses
data = []

for test_num, test_path in test_dirs:
    sv_dir = os.path.join(test_path, "singular_values")
    sv_file = os.path.join(sv_dir, "hidden_dim10000_epoch1000.pt")
    json_file = os.path.join(test_path, "final_metrics_hidden_dim10000.json")
    
    # Check if both files exist
    if not os.path.exists(sv_file):
        print(f"Warning: Singular values file not found: {sv_file}")
        continue
    if not os.path.exists(json_file):
        print(f"Warning: JSON file not found: {json_file}")
        continue
    
    # Load singular values
    try:
        sv_tensor = torch.load(sv_file, map_location="cpu", weights_only=True)
        sv = sv_tensor.numpy()
    except Exception as e:
        print(f"Error loading {sv_file}: {e}")
        continue
    
    # Load test loss from JSON
    try:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
            test_loss = metrics.get("epoch1000_test_loss")
            if test_loss is None:
                print(f"Warning: epoch100_test_loss not found in {json_file}")
                continue
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        continue
    
    data.append({
        'test_num': test_num,
        'sv': sv,
        'test_loss': test_loss
    })

if not data:
    print("No valid data found!")
    exit(1)

print(f"Loaded {len(data)} test runs")

# Extract test losses for color mapping
test_losses = [d['test_loss'] for d in data]
min_loss = min(test_losses)
max_loss = max(test_losses)

print(f"Test loss range: {min_loss:.6f} to {max_loss:.6f}")

# Calculate mean singular value spectrum
# First, find the minimum length to ensure all arrays can be aligned
min_length = min(len(d['sv']) for d in data)
print(f"Minimum SV array length: {min_length}")

# Truncate all SV arrays to the same length and stack them
sv_arrays = np.array([d['sv'][:min_length] for d in data])

# Calculate mean spectrum
mean_sv = np.mean(sv_arrays, axis=0)

# Calculate differences from mean for each spectrum
for d in data:
    d['sv_diff'] = d['sv'][:min_length] - mean_sv

# ----- Plotting differences from mean with color gradient based on test loss -----
fig, ax = plt.subplots(figsize=(10, 6))

# Create color gradient based on test loss
# Using viridis_r (reversed) so lower loss = darker/cooler color
cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=min_loss, vmax=max_loss)

for d in data:
    sv_diff = d['sv_diff']
    test_loss = d['test_loss']
    test_num = d['test_num']
    
    color = cmap(norm(test_loss))
    indices = np.arange(1, len(sv_diff) + 1)  # Start from 1 for log scale
    ax.plot(indices[-200:], ((sv_diff)[-200:]), color=color, label=f"test/{test_num} (loss={test_loss:.4f})", alpha=0.7)
    
    # Plot a dot at the last point
    max_idx = len(sv_diff)
    max_val = sv_diff[-1]
    # ax.plot(max_idx, max_val, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)

# Add horizontal line at zero
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Mean')

ax.set_title("Singular Values Difference from Mean by Test Loss")
ax.set_xlabel("Index")
ax.set_ylabel("SV Difference from Mean")
ax.set_xscale("log")
ax.set_yscale("linear")

ax.grid(True, alpha=0.3, which="both")

# Add colorbar to show test loss gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Epoch 100 Test Loss")

# Only show legend if there aren't too many runs
if len(data) <= 10:
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
filename = 'sv_difference_from_mean_by_test_loss.png'
plt.savefig(filename, dpi=150)
print(f"Saved plot to {filename}")

plt.show()
