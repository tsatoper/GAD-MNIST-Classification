import os
import json
import matplotlib.pyplot as plt
import numpy as np

plot_id = '-4' 

dir_name = f'./models/{plot_id}/metrics'
# Define epochs to plot
epochs = [100, 500]
yscale = 'linear'
save_path = f'loss_{plot_id}_{yscale}.png'


# Dictionary to store data for each epoch
epoch_data = {epoch: {'width': [], 'test_losses': [], 'train_losses': []} 
              for epoch in epochs}

# Read all JSON files
if not os.path.exists(dir_name):
    print(f"Warning: Directory '{dir_name}' not found, skipping...")
else:
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
                    
                    if train_key in data and test_key in data:
                        epoch_data[epoch]['width'].append(width)
                        epoch_data[epoch]['train_losses'].append(data[train_key])
                        epoch_data[epoch]['test_losses'].append(data[test_key])
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Sort data for each epoch
    for epoch in epochs:
        if len(epoch_data[epoch]['width']) > 0:
            sorted_indices = np.argsort(epoch_data[epoch]['width'])
            epoch_data[epoch]['width'] = np.array(epoch_data[epoch]['width'])[sorted_indices]
            epoch_data[epoch]['test_losses'] = np.array(epoch_data[epoch]['test_losses'])[sorted_indices]
            epoch_data[epoch]['train_losses'] = np.array(epoch_data[epoch]['train_losses'])[sorted_indices]


    # ====== LOSS PLOT ======
    plt.figure(figsize=(12, 7))
    
        # Define colors for each epoch
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    for idx, epoch in enumerate(epochs):
        if len(epoch_data[epoch]['width']) > 0:
            width = epoch_data[epoch]['width']
            test_losses = epoch_data[epoch]['test_losses']
            train_losses = epoch_data[epoch]['train_losses']
            
            # Plot test losses
            plt.plot(width, test_losses, linestyle='-', 
                    linewidth=2, label=f'Epoch {epoch} - Test Loss', 
                    alpha=0.8, color=colors[idx])
            plt.scatter(width, test_losses, s=40, alpha=0.8, 
                       zorder=5, color=colors[idx])
            
            # Plot train losses
            plt.plot(width, train_losses, linestyle='--', 
                    linewidth=2, label=f'Epoch {epoch} - Train Loss', 
                    alpha=0.6, color=colors[idx])
            plt.scatter(width, train_losses, s=40, alpha=0.6, 
                       zorder=5, color=colors[idx])

    # Formatting
    plt.xlabel('Model Width', fontsize=13)
    plt.ylabel(f'Loss ({yscale})', fontsize=13)
    plt.title(f'Train vs Test Loss by Model Width ({plot_id})', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale(yscale)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved to "{save_path}"')