import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

directory = '/glade/derecho/scratch/tsatoperry/GAD/models/mse/weights/'

# Dictionary to store results: {hidden_dim: min_singular_value}
results = {}

# Get all .pth files in the directory
pth_files = [f for f in os.listdir(directory) if f.endswith('.pth')]

# Extract hidden dimension from filename and process
for filename in pth_files:
    # Extract hidden dimension using regex
    match = re.search(r'hidden_dim(\d+)_', filename)
    if match:
        hidden_dim = int(match.group(1))
        
        try:
            model = torch.load(os.path.join(directory, filename), weights_only=True)
            hidden_layer = model['fc1.weight']
            
            # Compute SVD
            U, S, Vh = torch.svd(hidden_layer)
            S = S.cpu().numpy()
            
            # Store the minimum singular value
            results[hidden_dim] = S.min()
            
            print(f'{filename}: Hidden dim {hidden_dim}, min singular value = {S.min():.6f}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')
            continue
    else:
        print(f'Could not extract hidden_dim from {filename}')

# Sort by hidden dimension
hidden_dims = sorted(results.keys())
min_singular_values = [results[dim] for dim in hidden_dims]

# Plot minimum singular values across hidden dimensions
plt.figure(figsize=(12, 6))
plt.plot(hidden_dims, min_singular_values, 'o-', linewidth=2, markersize=6)

plt.xlabel('Hidden Layer Dimension')
plt.ylabel('Minimum Singular Value')
plt.title('Minimum Singular Value vs Hidden Layer Dimension')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('min_singular_values_all_models.png', dpi=150)
plt.show()

print(f'\nSummary:')
print(f'Total models processed: {len(results)}')
print(f'Hidden dims range: {min(hidden_dims)} to {max(hidden_dims)}')