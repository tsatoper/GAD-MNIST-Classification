import os
import json
import matplotlib.pyplot as plt
import numpy as np

plot_id = 'mse' #'epoch2000_tracking_epoch_loss'
dir_name = f'./models/{plot_id}'

# Define epochs to plot
epochs = [500, 1000, 1500, 2000]
yscale = 'linear'

# Dictionary to store data for each epoch
epoch_data = {epoch: {'num_parameters': [], 'test_losses': [], 'train_losses': []} 
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
                num_params = data['num_parameters']
                
                # Extract metrics for each epoch
                for epoch in epochs:
                    train_key = f'epoch{epoch}_train_loss'
                    test_key = f'epoch{epoch}_test_loss'
                    
                    if train_key in data and test_key in data:
                        epoch_data[epoch]['num_parameters'].append(num_params)
                        epoch_data[epoch]['train_losses'].append(data[train_key])
                        epoch_data[epoch]['test_losses'].append(data[test_key])
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Sort data for each epoch
    for epoch in epochs:
        if len(epoch_data[epoch]['num_parameters']) > 0:
            sorted_indices = np.argsort(epoch_data[epoch]['num_parameters'])
            epoch_data[epoch]['num_parameters'] = np.array(epoch_data[epoch]['num_parameters'])[sorted_indices]
            epoch_data[epoch]['test_losses'] = np.array(epoch_data[epoch]['test_losses'])[sorted_indices]
            epoch_data[epoch]['train_losses'] = np.array(epoch_data[epoch]['train_losses'])[sorted_indices]


    # ====== LOSS PLOT ======
    plt.figure(figsize=(12, 7))
    
    # Define colors for each epoch
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, epoch in enumerate(epochs):
        if len(epoch_data[epoch]['num_parameters']) > 0:
            num_params = epoch_data[epoch]['num_parameters']
            test_losses = epoch_data[epoch]['test_losses']
            train_losses = epoch_data[epoch]['train_losses']
            
            # Plot test losses
            plt.plot(num_params, test_losses, linestyle='-', 
                    linewidth=2, label=f'Epoch {epoch} - Test Loss', 
                    alpha=0.8, color=colors[idx])
            plt.scatter(num_params, test_losses, s=40, alpha=0.8, 
                       zorder=5, color=colors[idx])
            
            # Plot train losses
            plt.plot(num_params, train_losses, linestyle='--', 
                    linewidth=2, label=f'Epoch {epoch} - Train Loss', 
                    alpha=0.6, color=colors[idx])
            plt.scatter(num_params, train_losses, s=40, alpha=0.6, 
                       zorder=5, color=colors[idx])

    # Formatting
    plt.xlabel('Number of Model Parameters', fontsize=13)
    plt.ylabel(f'Loss ({yscale})', fontsize=13)
    plt.title(f'Train vs Test Loss by Number of Model Parameters ({plot_id})', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=9, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale(yscale)
    plt.tight_layout()

    # Save plot
    plt.savefig(f'train_vs_test_loss_{plot_id}_all_epochs_{yscale}.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved to "train_vs_test_loss_{plot_id}_all_epochs_{yscale}.png"')