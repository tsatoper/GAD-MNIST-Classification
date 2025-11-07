import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

directory = '/glade/derecho/scratch/tsatoperry/GAD/models/mse/'
weight_directory = os.path.join(directory, 'weights')
cache_path = os.path.join(directory, 'combined_svd_loss_data.pt')
save_path = 'combined_svd_train_test_loss.png'

epoch = 2000  # single epoch

# ====== Try to load cached tensors ======
if os.path.exists(cache_path):
    print(f"Loading cached data from {cache_path}...")
    cached = torch.load(cache_path, weights_only=True)
    num_parameters = cached['num_parameters']
    min_svs = cached['min_svs']
    train_losses = cached['train_losses']
    test_losses = cached['test_losses']

else:
    print("No cache found — computing from JSON and weight files...")

    # ====== Extract SVD data ======
    svd_results = {}
    for filename in os.listdir(weight_directory):
        if not filename.endswith('.pth'):
            continue

        match = re.search(r'hidden_dim(\d+)_', filename)
        if not match:
            continue

        hidden_dim = int(match.group(1))
        try:
            state_dict = torch.load(os.path.join(weight_directory, filename), weights_only=True, map_location='cpu')

            U, S, Vh = torch.svd(state_dict['fc1.weight'])
            num_params = sum(p.numel() for p in state_dict.values())
            svd_results[hidden_dim] = (S.cpu().numpy().min(), num_params)
        except Exception as e:
            print(f'Error loading {filename}: {e}')

    if len(svd_results) == 0:
        raise RuntimeError("No valid weight files found for SVD computation.")

    # Sort by number of parameters
    sorted_items = sorted(svd_results.items(), key=lambda x: x[1][1])
    num_parameters = torch.tensor([item[1][1] for item in sorted_items], dtype=torch.float32)
    min_svs = torch.tensor([item[1][0] for item in sorted_items], dtype=torch.float32)

    # ====== Extract loss data ======
    num_params_list = []
    train_losses_list = []
    test_losses_list = []

    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)

            num_params = data.get('num_parameters', None)
            if num_params is None:
                continue

            train_key = f'epoch{epoch}_train_loss'
            test_key = f'epoch{epoch}_test_loss'

            if train_key in data and test_key in data:
                num_params_list.append(num_params)
                train_losses_list.append(data[train_key])
                test_losses_list.append(data[test_key])
        except Exception as e:
            print(f'Error reading {filename}: {e}')

    if len(num_params_list) == 0:
        raise RuntimeError("No matching JSON files with losses found.")

    # Sort to match parameter order
    idx = np.argsort(num_params_list)
    num_params_sorted = torch.tensor(np.array(num_params_list)[idx], dtype=torch.float32)
    train_losses = torch.tensor(np.array(train_losses_list)[idx], dtype=torch.float32)
    test_losses = torch.tensor(np.array(test_losses_list)[idx], dtype=torch.float32)

    # ====== Save cached tensors ======
    torch.save({
        'num_parameters': num_params_sorted,
        'min_svs': min_svs,
        'train_losses': train_losses,
        'test_losses': test_losses
    }, cache_path)
    print(f"Saved tensor data to {cache_path}")

# ====== Combined plot ======
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot singular values
ax1.plot(num_parameters, min_svs, 'o-', linewidth=2, markersize=8,
         color='tab:purple', label='Min Singular Value', alpha=0.7)
ax1.set_xlabel('Number of Parameters', fontsize=13)
ax1.set_ylabel('Minimum Singular Value', fontsize=13, color='tab:purple')
ax1.tick_params(axis='y', labelcolor='tab:purple')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Secondary axis for loss
ax2 = ax1.twinx()
ax2.set_ylabel('Loss (log)', fontsize=13)

# Plot train and test losses
ax2.plot(num_parameters[:len(train_losses)], train_losses,
         '--', linewidth=2, label=f'Training Error',
         alpha=0.7, color='tab:blue')
ax2.plot(num_parameters[:len(test_losses)], test_losses,
         '--', linewidth=2, label=f'Test Error',
         alpha=0.7, color='tab:orange')

# Log scales for all axes
ax2.set_xscale('log')
ax2.set_yscale('log')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc='lower left')

plt.title(f'Minimum Singular Value and Train/Test Loss vs Number of Parameters',
          fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Saved plot to {save_path}")
