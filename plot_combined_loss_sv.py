import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_min_singular_values(directory):
    """Load minimum singular values from hidden_dimX_sv.pt files."""
    pattern = re.compile(r"hidden_dim(\d+)_sv\.pt$")
    pattern2 = re.compile(r"hidden_dim(\d+)\.pt$")

    results = {}
    
    if not os.path.exists(directory):
        return results
    
    for fname in os.listdir(directory):
        if (match := pattern.match(fname)) or (match := pattern2.match(fname)):
            hidden_dim = int(match.group(1))
            print(f'{hidden_dim=}')
            try:
                sv_tensor = torch.load(os.path.join(directory, fname), map_location='cpu', weights_only=True)
                results[hidden_dim] = float(sv_tensor.min().item())
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
    
    return results

def main():
    plot_id = 'mse'
    model_dir = f'./models/{plot_id}'
    sv_directory = f"{model_dir}/singular_values"
    
    # Load singular values to filter valid hidden dimensions
    sv_results = load_min_singular_values(sv_directory)
    valid_hidden_dims = set(sv_results.keys())
    
    if not valid_hidden_dims:
        print("No singular value files found.")
        return
    
    # Collect data
    data = {'num_parameters': [], 'test_losses': [], 'train_losses': [], 'hidden_dims': []}
    
    for filename in os.listdir(model_dir):
        if not filename.endswith('.json'):
            continue
            
        try:
            with open(os.path.join(model_dir, filename), 'r') as f:
                file_data = json.load(f)
            
            # Extract hidden_dim
            hidden_dim = file_data.get('hidden_dim') or (int(m.group(1)) if (m := re.search(r'hidden_dim(\d+)', filename)) else None)
            
            if hidden_dim in valid_hidden_dims and 'epoch2000_train_loss' in file_data and 'epoch2000_test_loss' in file_data:
                data['num_parameters'].append(file_data['num_parameters'])
                data['train_losses'].append(file_data['epoch2000_train_loss'])
                data['test_losses'].append(file_data['epoch2000_test_loss'])
                data['hidden_dims'].append(hidden_dim)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    # Sort by number of parameters
    sorted_idx = np.argsort(data['num_parameters'])
    num_params = np.array(data['num_parameters'])[sorted_idx]
    test_losses = np.array(data['test_losses'])[sorted_idx]
    train_losses = np.array(data['train_losses'])[sorted_idx]
    hidden_dims = np.array(data['hidden_dims'])[sorted_idx]
    
    # Get min singular values for each model
    min_svs = np.array([sv_results[h] for h in hidden_dims])
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot losses on left axis
    ax1.plot(num_params, test_losses, '-o', linewidth=2, label='Test Loss', alpha=0.8, color='#1f77b4')
    ax1.plot(num_params, train_losses, '--o', linewidth=2, label='Train Loss', alpha=0.6, color='#1f77b4')
    ax1.set_xlabel('Number of Model Parameters', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot singular values on right axis
    ax2 = ax1.twinx()
    ax2.plot(num_params, min_svs, '-s', linewidth=2, label='Min Singular Value', alpha=0.8, color='#ff7f0e')
    ax2.set_ylabel('Minimum Singular Value', fontsize=13, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='best')
    
    plt.title(f'Train vs Test Loss and Min Singular Value({plot_id})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'combined_plot_{plot_id}.png', dpi=300, bbox_inches='tight')
    print(f'Plotted {len(num_params)} models. Saved combined plot.')

if __name__ == "__main__":
    main()