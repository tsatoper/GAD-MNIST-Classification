import os
import json
import matplotlib.pyplot as plt
import numpy as np

plot_id = 'testing' 

# Define epochs to plot
epochs = [100, 500]
yscale = 'linear'
save_path = f'loss_sv_{plot_id}_{epochs[0]}.png'

# Dictionary to store data for each plot_id and epoch
all_data = {plot_id: {epoch: {'width': [], 'test_losses': [], 'train_losses': [], 'sv_min': []} 
                      for epoch in epochs}}

# Read all JSON files for each plot_id
dir_name = f'./models/{plot_id}/metrics'
print(dir_name)



for filename in os.listdir(dir_name):
    if filename.endswith('.json'):
        filepath = os.path.join(dir_name, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            width = data['width']
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

# Define colors for each learning rate
lr_colors = {
'lr1e-4': '#1f77b4',  # blue
'lr1e-3': '#ff7f0e',  # orange
}

# Plot losses on left axis
for idx, epoch in enumerate(epochs):
    if len(all_data[plot_id][epoch]['width']) > 0:
        width = all_data[plot_id][epoch]['width']
        test_losses = all_data[plot_id][epoch]['test_losses']
        train_losses = all_data[plot_id][epoch]['train_losses']
        
        color = lr_colors.get(plot_id, '#000000')
        
        # Plot test losses
        ax1.plot(width, test_losses, linestyle='-', 
                linewidth=2, label=f'{plot_id} - Epoch {epoch} - Test Loss', 
                alpha=0.8, color=color)
        ax1.scatter(width, test_losses, s=40, alpha=0.8, 
                    zorder=5, color=color)
        
        # Plot train losses
        ax1.plot(width, train_losses, linestyle='--', 
                linewidth=2, label=f'{plot_id} - Epoch {epoch} - Train Loss', 
                alpha=0.6, color=color)
        ax1.scatter(width, train_losses, s=40, alpha=0.6, 
                    zorder=5, color=color)

# Formatting for left axis (Loss)
ax1.set_xlabel('Model Width', fontsize=13)
ax1.set_ylabel(f'Loss ({yscale})', fontsize=13, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xscale('log')
ax1.set_yscale(yscale)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper left', ncol=2)

# Create second y-axis for SV_MIN
ax2 = ax1.twinx()

# Plot sv_min on right axis
for idx, epoch in enumerate(epochs):
    if len(all_data[plot_id][epoch]['width']) > 0:
        width = all_data[plot_id][epoch]['width']
        sv_min = all_data[plot_id][epoch]['sv_min']
        
        color = lr_colors.get(plot_id, '#000000')
        
        # Use dotted line with square marker for sv_min
        ax2.plot(width, sv_min, linestyle=':', 
                linewidth=2.5, label=f'{plot_id} - Epoch {epoch} - Min SV', 
                alpha=0.7, color=color)
        ax2.scatter(width, sv_min, s=60, alpha=0.7, 
                    marker='s', zorder=5, color=color)

# Formatting for right axis (SV_MIN)
ax2.set_ylabel('Minimum Singular Value', fontsize=13, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_yscale('log')
ax2.legend(fontsize=9, loc='upper right')

# Title
plt.title(f'Train/Test Loss and Min Singular Value - Learning Rate Comparison', 
          fontsize=15, fontweight='bold')

fig.tight_layout()

# Save plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')

plt.show()