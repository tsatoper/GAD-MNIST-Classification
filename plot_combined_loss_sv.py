import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_min_singular_values(directory):
    """Load minimum singular values from hidden_dimX_sv.pt files."""
    pattern = re.compile(r"hidden_dim(\d+)(?:_sv)?\.pt$")
    results = {}
    
    if not os.path.exists(directory):
        return results
    
    for fname in os.listdir(directory):
        if match := pattern.match(fname):
            hidden_dim = int(match.group(1))
            print(f'{hidden_dim=}')
            try:
                sv_tensor = torch.load(os.path.join(directory, fname), map_location='cpu', weights_only=True)
                results[hidden_dim] = float(sv_tensor.min().item())
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
    
    return results

def load_model_data(model_name):
    """Load data for a single model."""
    model_dir = f'./models/{model_name}'
    sv_directory = f"{model_dir}/singular_values"
    
    # Load singular values to filter valid hidden dimensions
    sv_results = load_min_singular_values(sv_directory)
    valid_hidden_dims = set(sv_results.keys())
    
    if not valid_hidden_dims:
        print(f"No singular value files found for {model_name}.")
        return None
    
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
    
    if not data['num_parameters']:
        return None
    
    # Sort by number of parameters
    sorted_idx = np.argsort(data['num_parameters'])
    num_params = np.array(data['num_parameters'])[sorted_idx]
    test_losses = np.array(data['test_losses'])[sorted_idx]
    train_losses = np.array(data['train_losses'])[sorted_idx]
    hidden_dims = np.array(data['hidden_dims'])[sorted_idx]
    
    # Get min singular values for each model
    min_svs = np.array([sv_results[h] for h in hidden_dims])
    
    return {
        'num_params': num_params,
        'test_losses': test_losses,
        'train_losses': train_losses,
        'min_svs': min_svs
    }

def create_combined_plot(all_data, yscale1, yscale2):
    """Create a single plot with all models and mean lines."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, green, red
    model_names = ['omni', 'omni2', 'omni3']
    
    # Collect all data points for computing means
    all_num_params = []
    all_test_losses = []
    all_train_losses = []
    all_min_svs = []
    
    # Plot individual model data
    for i, (model_name, data) in enumerate(zip(model_names, all_data)):
        if data is None:
            continue
        
        color = colors[i]
        alpha = 0.4
        
        # Plot losses on left axis
        ax1.plot(data['num_params'], data['test_losses'], 'o', 
                linewidth=1, label=f'{model_name} Test Loss', alpha=alpha, color=color, markersize=6)
        ax1.plot(data['num_params'], data['train_losses'], 's', 
                linewidth=1, alpha=alpha*0.7, color=color, markersize=5)
        
        # Collect data for mean calculation
        all_num_params.extend(data['num_params'])
        all_test_losses.extend(data['test_losses'])
        all_train_losses.extend(data['train_losses'])
        all_min_svs.extend(data['min_svs'])
    
    # Compute and plot mean lines for losses
    if all_num_params:
        # Group by num_params and compute means
        unique_params = sorted(set(all_num_params))
        mean_test_losses = []
        mean_train_losses = []
        
        for param in unique_params:
            indices = [i for i, p in enumerate(all_num_params) if p == param]
            mean_test_losses.append(np.mean([all_test_losses[i] for i in indices]))
            mean_train_losses.append(np.mean([all_train_losses[i] for i in indices]))
        
        ax1.plot(unique_params, mean_test_losses, '-', linewidth=3, 
                label='Mean Test Loss', color='black', alpha=0.8)
        ax1.plot(unique_params, mean_train_losses, '--', linewidth=3, 
                label='Mean Train Loss', color='black', alpha=0.6)
    
    ax1.set_xlabel('Number of Model Parameters', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xscale('log')
    ax1.set_yscale(yscale1)
    ax1.grid(True, alpha=0.3)
    
    # Plot singular values on right axis
    ax2 = ax1.twinx()
    ax2.set_yscale(yscale2)
    
    # Plot individual model SVs
    for i, (model_name, data) in enumerate(zip(model_names, all_data)):
        if data is None:
            continue
        
        color = colors[i]
        ax2.plot(data['num_params'], data['min_svs'], '^', 
                linewidth=1, label=f'{model_name} Min SV', alpha=0.4, color=color, markersize=6)
    
    # Compute and plot mean line for SVs
    if all_num_params:
        mean_min_svs = []
        for param in unique_params:
            indices = [i for i, p in enumerate(all_num_params) if p == param]
            mean_min_svs.append(np.mean([all_min_svs[i] for i in indices]))
        
        ax2.plot(unique_params, mean_min_svs, '-', linewidth=3, 
                label='Mean Min SV', color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('Minimum Singular Value', fontsize=13, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best', ncol=2)
    
    plt.title(f'Combined Models: Loss ({yscale1}) and Min Singular Value ({yscale2})', 
             fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'combined_all_models_{yscale1}_{yscale2}.png', dpi=300, bbox_inches='tight')
    print(f'Saved: combined_all_models_{yscale1}_{yscale2}.png')
    plt.close()

def main():
    model_names = ['mse']
    
    # Load data for all models
    all_data = []
    for model_name in model_names:
        print(f'\nLoading data for {model_name}...')
        data = load_model_data(model_name)
        all_data.append(data)
    
    # Generate plots with different scale combinations
    plot_configs = [
        ('log', 'log'),
        ('linear', 'linear')
    ]
    
    print('\nGenerating combined plots...\n')
    for yscale1, yscale2 in plot_configs:
        create_combined_plot(all_data, yscale1, yscale2)
    
    print('\nAll plots generated successfully!')

if __name__ == "__main__":
    main()