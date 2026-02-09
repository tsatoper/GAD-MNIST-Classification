import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

plot_id = 'gobig'
dir_name = f'/glade/derecho/scratch/tsatoperry/GAD/KS_1d/AR_MLP_deep/{plot_id}/metrics/'
yscale = 'linear'
epochs = [200]


# Dictionary to store data for each epoch
epoch_data = {epoch: {'width': [], 'train_losses': [], 'val_losses': []} 
              for epoch in epochs}

# Read all JSON files
if not os.path.exists(dir_name):
    print(f"Warning: Directory '{dir_name}' not found, skipping...")
    exit(1)

for filename in os.listdir(dir_name):
    # Match pattern: w{width}_ttrain_N{number}.json
    # Example: w2_ttrain_N2.json
    match = re.match(rf'h_(\d+)\_job(\d+)\.json', filename)  

    width = int(match.group(1))
    filepath = os.path.join(dir_name, filename)
    
    try:

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract metrics for each epoch
        for epoch in epochs:
            train_key = f'epoch{epoch}_train_loss'
            val_key = f'epoch{epoch}_val_loss'
            
            if train_key in data and val_key in data:
                epoch_data[epoch]['width'].append(width)
                epoch_data[epoch]['train_losses'].append(data[train_key])
                epoch_data[epoch]['val_losses'].append(data[val_key])
                    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {filename}: {e}")
        continue

# Sort data for each epoch by width
for epoch in epochs:
    if len(epoch_data[epoch]['width']) > 0:
        sorted_indices = np.argsort(epoch_data[epoch]['width'])
        epoch_data[epoch]['width'] = np.array(epoch_data[epoch]['width'])[sorted_indices]
        epoch_data[epoch]['train_losses'] = np.array(epoch_data[epoch]['train_losses'])[sorted_indices]
        epoch_data[epoch]['val_losses'] = np.array(epoch_data[epoch]['val_losses'])[sorted_indices]

print(f"Loaded data for {len([e for e in epochs if len(epoch_data[e]['width']) > 0])} epochs")

# ====== LOSS PLOT ======
plt.figure(figsize=(12, 7))

# Define colors for each epoch
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, epoch in enumerate(epochs):
    if len(epoch_data[epoch]['width']) > 0:
        width = epoch_data[epoch]['width']
        train_losses = epoch_data[epoch]['train_losses']
        val_losses = epoch_data[epoch]['val_losses']
        
        print(f"\nEpoch {epoch}:")
        print(f"  Widths: {width}")
        print(f"  Train losses: {train_losses}")
        print(f"  Val losses: {val_losses}")
        
        # Plot val losses
        plt.plot(width, val_losses, linestyle='-', 
                linewidth=2, label=f'Epoch {epoch} - Val Loss', 
                alpha=0.8, color=colors[idx])
        plt.scatter(width, val_losses, s=60, alpha=0.8, 
                   zorder=5, color=colors[idx])
        
        # Plot train losses
        plt.plot(width, train_losses, linestyle='--', 
                linewidth=2, label=f'Epoch {epoch} - Train Loss', 
                alpha=0.6, color=colors[idx])
        plt.scatter(width, train_losses, s=60, alpha=0.6, 
                   zorder=5, color=colors[idx])

# Formatting

plt.xlabel('Width', fontsize=13)
plt.ylabel(f'Loss ({yscale})', fontsize=13)
plt.title(f'Train vs Val Loss by Width ({plot_id})', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=10, loc='best', ncol=2)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale(yscale)
plt.tight_layout()

# Save plot
output_filename = f'loss_{plot_id}_{yscale}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{output_filename}"')
plt.show()