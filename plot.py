import os
import json
import matplotlib.pyplot as plt
import numpy as np

plot_id = 'epoch1000'
dir_name = f'./models/{plot_id}'
yscale = 'linear'
if not os.path.exists(dir_name):
    print(f"Warning: Directory '{dir_name}' not found, skipping...")
else:
    hidden_dims = []
    test_losses = []
    train_losses = []
    test_accs = []
    train_accs = []
    
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
                test_accs.append(data['final_test_acc'])
                train_accs.append(data['final_train_acc'])
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Sort by hidden_dim
    sorted_indices = np.argsort(hidden_dims)
    hidden_dims = np.array(hidden_dims)[sorted_indices]
    test_losses = np.array(test_losses)[sorted_indices]
    train_losses = np.array(train_losses)[sorted_indices]
    test_accs = np.array(test_accs)[sorted_indices]
    train_accs = np.array(train_accs)[sorted_indices]

    print(f"\n{dir_name}: Found {len(hidden_dims)} models")
    print(f"  Hidden dimensions: {hidden_dims}")
    print("\nHidden Dim, Train Loss, Test Loss, Train Acc, Test Acc")
    for i in range(len(hidden_dims)):
        print(f'{hidden_dims[i]}, {train_losses[i]}, {test_losses[i]}, {train_accs[i]}, {test_accs[i]}')

    # Get clean label from directory name
    label = dir_name.strip('./').rstrip('/')

    # ====== LOSS PLOT ======
    plt.figure(figsize=(10, 6))
    
    plt.plot(hidden_dims, test_losses, linestyle='-', 
                linewidth=2, markersize=8, label=f'{label} - Test Loss', alpha=0.8)
    plt.plot(hidden_dims, train_losses, linestyle='--', 
                linewidth=2, markersize=8, label=f'{label} - Train Loss', alpha=0.8)

    # Formatting
    plt.xlabel('Hidden Dimension', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Train vs Test Loss Across Hidden Dimensions', fontsize=15, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale(yscale)
    plt.tight_layout()

    # Save plot
    plt.savefig(f'train_vs_test_loss_{plot_id}_{yscale}.png', dpi=300, bbox_inches='tight')
    print(f'saved to "train_vs_test_loss_{plot_id}_{yscale}.png"')
