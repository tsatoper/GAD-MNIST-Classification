import os
import json
import matplotlib.pyplot as plt
import numpy as np

plot_id = 'mse' 

# Define epochs to plot
epochs = [2000]
yscale = 'linear'
save_path = f'loss_sv_{plot_id}_{epochs[0]}.png'

# Dictionary to store data for each plot_id and epoch
all_data = {plot_id: {epoch: {'width': [], 'test_losses': [], 'train_losses': [], 'sv_min': []} 
                      for epoch in epochs}}

# Read all JSON files for each plot_id
dir_name = f'../models/{plot_id}/metrics'
print(dir_name)



for filename in os.listdir(dir_name):
    if filename.endswith('.json'):
        filepath = os.path.join(dir_name, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            width = data['hidden']
            # Extract metrics for each epoch
            for epoch in epochs:
                train_key = f'epoch{epoch}_train_loss'
                test_key = f'epoch{epoch}_test_loss'
                sv_min_key = f'epoch{epoch}_sv_min'
                
                if train_key in data and test_key in data and sv_min_key in data:
                    all_data[plot_id][epoch]['width'].append(width)
                    all_data[plot_id][epoch]['train_losses'].append(data[train_key])
                    all_data[plot_id][epoch]['test_losses'].append(data[test_key])
                    all_data[plot_id][epoch]['sv_min'].append(data[sv_min_key])
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {filename}: {e}")
            continue

# Sort data for each epoch
for epoch in epochs:
    if len(all_data[plot_id][epoch]['width']) > 0:
        sorted_indices = np.argsort(all_data[plot_id][epoch]['width'])
        all_data[plot_id][epoch]['width'] = np.array(all_data[plot_id][epoch]['width'])[sorted_indices]
        all_data[plot_id][epoch]['test_losses'] = np.array(all_data[plot_id][epoch]['test_losses'])[sorted_indices]
        all_data[plot_id][epoch]['train_losses'] = np.array(all_data[plot_id][epoch]['train_losses'])[sorted_indices]
        all_data[plot_id][epoch]['sv_min'] = np.array(all_data[plot_id][epoch]['sv_min'])[sorted_indices]


# ====== DUAL AXIS PLOT: LOSS AND SV_MIN ======
fig, ax1 = plt.subplots(figsize=(14, 7))

# Get all unique widths across epochs for colormap
all_widths = []
for epoch in epochs:
    if len(all_data[plot_id][epoch]['width']) > 0:
        all_widths.extend(all_data[plot_id][epoch]['width'])
all_widths = np.unique(all_widths)

# Set up colormap
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(all_widths), vmax=max(all_widths))

# Plot losses on left axis
for idx, epoch in enumerate(epochs):
    if len(all_data[plot_id][epoch]['width']) > 0:
        width = all_data[plot_id][epoch]['width']
        test_losses = all_data[plot_id][epoch]['test_losses']
        train_losses = all_data[plot_id][epoch]['train_losses']
        
        # Use different line styles for different epochs
        linestyle = '-' if idx == 0 else '--'
        
        for i, w in enumerate(width):
            color = cmap(norm(w))
            
            # Plot test losses
            if i < len(width) - 1:
                ax1.plot(width[i:i+2], test_losses[i:i+2], linestyle=linestyle, 
                        linewidth=2, alpha=0.8, color=color)
            ax1.scatter(w, test_losses[i], s=40, alpha=0.8, 
                        zorder=5, color=color)
            
            # Plot train losses with lighter alpha
            if i < len(width) - 1:
                ax1.plot(width[i:i+2], train_losses[i:i+2], linestyle=':', 
                        linewidth=2, alpha=0.6, color=color)
            ax1.scatter(w, train_losses[i], s=40, alpha=0.6, 
                        marker='^', zorder=5, color=color)

# Formatting for left axis (Loss)
ax1.set_xlabel('Model Width', fontsize=13)
ax1.set_ylabel(f'Loss ({yscale})', fontsize=13, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xscale('log')
ax1.set_yscale(yscale)
ax1.grid(True, alpha=0.3)

# Create second y-axis for SV_MIN
ax2 = ax1.twinx()

# Plot sv_min on right axis
for idx, epoch in enumerate(epochs):
    if len(all_data[plot_id][epoch]['width']) > 0:
        width = all_data[plot_id][epoch]['width']
        sv_min = all_data[plot_id][epoch]['sv_min']
        sv_min[sv_min<1e-10] = 1e-10
        
        # Use different markers for different epochs
        marker = 's' if idx == 0 else 'D'
        
        for i, w in enumerate(width):
            color = cmap(norm(w))
            
            # Use dotted line with square marker for sv_min
            if i < len(width) - 1:
                ax2.plot(width[i:i+2], sv_min[i:i+2], linestyle='-.', 
                        linewidth=2.5, alpha=0.7, color=color)
            ax2.scatter(w, sv_min[i], s=60, alpha=0.7, 
                        marker=marker, zorder=5, color=color)

# Formatting for right axis (SV_MIN)
ax2.set_ylabel('Minimum Singular Value', fontsize=13, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_yscale('log')

# # Add colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.1)
# cbar.set_label('Model Width', fontsize=12)

# Add manual legend for line styles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label=f'Epoch {epochs[0]} - Test Loss'),
    Line2D([0], [0], color='gray', linewidth=2, linestyle=':', label=f'Epoch {epochs[0]} - Train Loss'),
    Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-.', label=f'Epoch {epochs[0]} - Min SV'),
]
ax1.legend(handles=legend_elements, fontsize=9, loc='upper left')

# Title
plt.title(f'Train/Test Loss and Min Singular Value - {plot_id}', 
          fontsize=15, fontweight='bold')

fig.tight_layout()

# Save plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')

plt.show()