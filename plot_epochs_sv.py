import os
import re
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_singular_values_by_epoch(directory):
    """Load singular values from files, organized by epoch."""
    # Pattern to match: hidden_dim{N}_epoch{E}_sv.pt or hidden_dim{N}_sv.pt
    pattern_with_epoch = re.compile(r"hidden_dim(\d+)_epoch(\d+)_sv\.pt$")
    pattern_no_epoch = re.compile(r"hidden_dim(\d+)_sv\.pt$")
    
    results = {
        500: {},
        1000: {},
        'final': {}  # Files without epoch suffix
    }
    
    if not os.path.exists(directory):
        return results
    
    for fname in os.listdir(directory):
        # Try matching with epoch
        if match := pattern_with_epoch.match(fname):
            hidden_dim = int(match.group(1))
            epoch = int(match.group(2))
            
            if epoch in results:
                try:
                    sv_tensor = torch.load(os.path.join(directory, fname), map_location='cpu', weights_only=True)
                    results[epoch][hidden_dim] = float(sv_tensor.min().item())
                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")
        
        # Try matching without epoch (final models)
        elif match := pattern_no_epoch.match(fname):
            hidden_dim = int(match.group(1))
            try:
                sv_tensor = torch.load(os.path.join(directory, fname), map_location='cpu', weights_only=True)
                results['final'][hidden_dim] = float(sv_tensor.min().item())
                print(f'Loaded {fname}: hidden_dim={hidden_dim}, final epoch, min_sv={results["final"][hidden_dim]:.6e}')
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
    
    return results

def load_model_data_by_epoch(model_name, epoch_key):
    """Load data for a single model at a specific epoch."""
    model_dir = f'/glade/derecho/scratch/tsatoperry/GAD/models/{model_name}'
    sv_directory = f"{model_dir}/singular_values"
    
    # Load singular values for all epochs
    all_sv_results = load_singular_values_by_epoch(sv_directory)
    sv_results = all_sv_results[epoch_key]
    valid_hidden_dims = set(sv_results.keys())
    
    if not valid_hidden_dims:
        print(f"No singular value files found for {model_name} at epoch {epoch_key}.")
        return None
    
    # Collect data
    data = {'num_parameters': [], 'hidden_dims': []}
    
    for filename in os.listdir(model_dir):
        if not filename.endswith('.json'):
            continue
            
        try:
            with open(os.path.join(model_dir, filename), 'r') as f:
                file_data = json.load(f)
            
            # Extract hidden_dim
            hidden_dim = file_data.get('hidden_dim') or (int(m.group(1)) if (m := re.search(r'hidden_dim(\d+)', filename)) else None)
            
            if hidden_dim in valid_hidden_dims:
                data['num_parameters'].append(file_data['num_parameters'])
                data['hidden_dims'].append(hidden_dim)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    if not data['num_parameters']:
        return None
    
    # Sort by number of parameters
    sorted_idx = np.argsort(data['num_parameters'])
    num_params = np.array(data['num_parameters'])[sorted_idx]
    hidden_dims = np.array(data['hidden_dims'])[sorted_idx]
    
    # Get min singular values for each model
    min_svs = np.array([sv_results[h] for h in hidden_dims])
    
    return {
        'num_params': num_params,
        'min_svs': min_svs
    }

def create_combined_plot(all_epoch_data, yscale):
    """Create a single plot with singular values for all epochs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epochs = [500, 1000, 'final']
    epoch_labels = {500: 'Epoch 500', 1000: 'Epoch 1000', 'final': 'Final'}
    
    # Colors for epochs
    epoch_colors = {
        500: '#1f77b4',      # Blue
        1000: '#2ca02c',     # Green
        'final': '#d62728'   # Red
    }
    
    # Plot data for each epoch
    for epoch_key in epochs:
        data = all_epoch_data[epoch_key]
        if data is None:
            continue
        
        epoch_color = epoch_colors[epoch_key]
        epoch_label = epoch_labels[epoch_key]
        
        # Plot singular values
        ax.plot(data['num_params'], data['min_svs'], 'o-', 
                label=f'{epoch_label}', color=epoch_color, 
                markersize=7, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Number of Model Parameters', fontsize=14)
    ax.set_ylabel('Minimum Singular Value', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    plt.title(f'Minimum Singular Value ({yscale} scale) Across Epochs', 
             fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    filename = f'sv_all_epochs_{yscale}_omni.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()

def main():
    model_name = 'omni'
    epochs = [500, 1000, 'final']
    
    # Load data for all epochs
    all_epoch_data = {}
    
    for epoch_key in epochs:
        print(f'Processing {epoch_key}...')
        
        data = load_model_data_by_epoch(model_name, epoch_key)
        all_epoch_data[epoch_key] = data
        
    # Generate plots with different scales
    plot_configs = ['log', 'linear']
    
    print('Generating plots...')
    
    for yscale in plot_configs:
        create_combined_plot(all_epoch_data, yscale)
    
    print('All plots generated successfully!')

if __name__ == "__main__":
    main()