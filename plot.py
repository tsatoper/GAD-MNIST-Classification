import os
import json
import matplotlib.pyplot as plt
import numpy as np

directories = ['./models/']

plt.figure(figsize=(10, 6))

for dir_name in directories:
    if not os.path.exists(dir_name):
        print(f"Warning: Directory '{dir_name}' not found, skipping...")
        continue
    
    hidden_dims = []
    test_losses = []
    train_losses = []
    for filename in os.listdir(dir_name):
        if filename.endswith('.json'):
            filepath = os.path.join(dir_name, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Extract metrics
                hidden_dims.append(data['hidden_dim'])
                test_losses.append(data['final_test_loss'])
                train_losses.append(data['final_train_loss'])
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    # Sort by hidden_dim
    sorted_indices = np.argsort(hidden_dims)
    hidden_dims = np.array(hidden_dims)[sorted_indices]
    test_losses = np.array(test_losses)[sorted_indices]
    train_losses = np.array(train_losses)[sorted_indices]
    
    print(f"\n{dir_name}: Found {len(hidden_dims)} models")
    print(f"  Hidden dimensions: {hidden_dims}")
    
    # Get clean label from directory name
    label = dir_name.strip('./').rstrip('/')
    
    # Plot with different markers for each directory
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    marker_idx = directories.index(dir_name) % len(markers)
    
    plt.plot(hidden_dims, test_losses, marker=markers[marker_idx], linestyle='-', 
             linewidth=2, markersize=8, label=f'{label} - Test Loss', alpha=0.8)
    plt.plot(hidden_dims, train_losses, marker=markers[marker_idx], linestyle='--', 
             linewidth=2, markersize=8, label=f'{label} - Train Loss', alpha=0.8)

# Formatting
plt.xlabel('Hidden Dimension', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.title('Train vs Test Loss Across Hidden Dimensions', fontsize=15, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()

# Save plot
plt.savefig('train_vs_test_loss.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'train_vs_test_loss.png'")