import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

model_dir = 'overfit'
epoch = 100
directory = f"/glade/derecho/scratch/tsatoperry/GAD/KS_1d/AR_MLP_one_layer/{model_dir}/singular_values"
filename = f'sv_trend_{model_dir}_simple.png'

# Regex for normal and test_set files
pattern_normal = re.compile(rf"h_(\d+)\_job(\d+)_e{epoch}.pt")
pattern_test = re.compile(rf"h_(\d+)\_job(\d+)test_set_e{epoch}.pt")

files_normal = []
files_test = []

# Collect matching files
for fname in os.listdir(directory):
    m_normal = pattern_normal.match(fname)
    m_test = pattern_test.match(fname)
    if m_normal:
        width = int(m_normal.group(1))
        files_normal.append((width, os.path.join(directory, fname)))
    elif m_test:
        width = int(m_test.group(1))
        files_test.append((width, os.path.join(directory, fname)))

# Sort by hidden_dim numerically
files_normal.sort(key=lambda x: x[0])
files_test.sort(key=lambda x: x[0])

# Function to load singular values
def load_svs(file_list):
    sv_arrays = []
    widths = []
    for hd, path in file_list:
        t = torch.load(path, map_location="cpu", weights_only=True)
        sv = t.numpy()
        sv[sv < 1e-16] = 1e-16
        sv_arrays.append(sv)
        widths.append(hd)
    return widths, sv_arrays

width_normal, sv_normal = load_svs(files_normal)
width_test, sv_test = load_svs(files_test)

# ----- Plot -----
fig, ax = plt.subplots(figsize=(10, 6))

# Plot normal files in blue
for sv in sv_normal:
    indices = np.arange(1, len(sv) + 1)
    ax.plot(indices, sv, color='blue', alpha=0.7)
    ax.plot(len(sv), sv[-1], 'o', color='blue', markersize=6, markeredgecolor='black', markeredgewidth=0.5)

# Plot test_set files in red
for sv in sv_test:
    indices = np.arange(1, len(sv) + 1)
    ax.plot(indices, sv, color='red', alpha=0.7)
    ax.plot(len(sv), sv[-1], 'o', color='red', markersize=6, markeredgecolor='black', markeredgewidth=0.5)

# Add legend for color groups
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Training set'),
    Line2D([0], [0], color='red', lw=2, label='Test set')
]
ax.legend(handles=legend_elements)

ax.set_title("Singular Values Comparison. KS 1D with euler")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_xscale("linear")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(filename, dpi=150)
print(filename)
plt.show()
