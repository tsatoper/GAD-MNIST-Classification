import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import re

from utilities import AR_MLP_one_layer, AR_MLP_deep
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--weights-dir', type=str, default='deep/long',help='Directory containing model weights')
parser.add_argument('--epoch', type=int, default=100, help='Epoch number to load weights from')
parser.add_argument('--val-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy', help='Path to validation data')
parser.add_argument('--num-rollout-steps', type=int, default=100, help='Number of autoregressive rollout steps')
args = parser.parse_args()

"""
p3 evaluate_multistep.py --epoch 100
"""

# Find all weight files with the given epoch (any hidden dimension)
weights_subdir = os.path.join('/glade/derecho/scratch/tsatoperry/GAD/KS_1d/', args.weights_dir, 'weights')
print(weights_subdir)
pattern = os.path.join(weights_subdir, f'h_*_job*_e{args.epoch}.pth')
matching_files = glob.glob(pattern)

if len(matching_files) == 0:
    raise FileNotFoundError(f"No weight files found matching pattern: {pattern}")

# Extract hidden dimensions and organize files
hidden_dim_files = {}
for filepath in matching_files:
    filename = os.path.basename(filepath)
    # Extract hidden dimension from filename: h_{hidden_dim}_job{X}_e{epoch}.pth
    match = re.search(r'h_(\d+)_job', filename)
    if match:
        hidden_dim = int(match.group(1))
        if hidden_dim not in hidden_dim_files:
            hidden_dim_files[hidden_dim] = []
        hidden_dim_files[hidden_dim].append(filepath)

print(f"Found {len(matching_files)} weight files for epoch {args.epoch}")
print(f"Hidden dimensions found: {sorted(hidden_dim_files.keys())}")
print()

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dt = 1e-3
input_dim = 1024  # KS system dimension
output_dim = input_dim

# Load validation data once
val_data = np.load(args.val_data_path)
print(f"Validation data shape: {val_data.shape}")

if len(val_data.shape) != 2:
    raise ValueError(f"Expected validation data shape (total_timesteps, spatial_dim), got {val_data.shape}")

# Determine how many rollout windows we can evaluate
n_rollouts = (val_data.shape[0] - args.num_rollout_steps) // args.num_rollout_steps
print(f"Evaluating {n_rollouts} rollout windows of {args.num_rollout_steps} steps each")
print("="*50)

# Loss function
loss_fn = nn.MSELoss()

# Store results for each hidden dimension
all_results = {}

# Evaluate each hidden dimension
for hidden_dim in sorted(hidden_dim_files.keys()):
    print(f"\nEvaluating hidden_dim = {hidden_dim}...")
    
    # Use the first file for this hidden dimension (or could average across jobs)
    weights_path = hidden_dim_files[hidden_dim][0]
    print(f"  Using: {os.path.basename(weights_path)}")

    # Initialize model
    model = AR_MLP_deep(hidden_dim=hidden_dim).to(device)
    
    # Load pretrained weights
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_parameters:,} ({num_parameters/1e6:.2f}M)")
    
    # Evaluate rollout
    step_losses = np.zeros(args.num_rollout_steps)
    total_loss = 0.0
    
    with torch.no_grad():
        for rollout_idx in range(n_rollouts):
            start_idx = rollout_idx * args.num_rollout_steps
            
            # Initial state
            initial_state = torch.from_numpy(val_data[start_idx:start_idx+1]).float().to(device)
            current_state = initial_state
            
            # Rollout
            rollout_loss = 0.0
            for step in range(args.num_rollout_steps):
                # Predict next state
                output = model(current_state)
                predicted_next = current_state + output * dt
                
                # Ground truth
                ground_truth = torch.from_numpy(val_data[start_idx + step + 1:start_idx + step + 2]).float().to(device)
                
                # Compute loss
                step_loss = loss_fn(predicted_next, ground_truth)
                step_losses[step] += step_loss.item()
                rollout_loss += step_loss.item()
                
                # Update state
                current_state = predicted_next
            
            total_loss += rollout_loss / args.num_rollout_steps
    
    # Average over all rollout windows
    step_losses /= n_rollouts
    avg_total_loss = total_loss / n_rollouts
    
    all_results[hidden_dim] = {
        'step_losses': step_losses,
        'avg_total_loss': avg_total_loss,
        'num_parameters': num_parameters
    }
    
    print(f"  Average rollout loss: {avg_total_loss:.6f}")

print("\n" + "="*50)
print("All evaluations complete!")
print("="*50)

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 7))

timesteps = np.arange(1, args.num_rollout_steps + 1)

# Use a colormap for different hidden dimensions
colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

for idx, (hidden_dim, results) in enumerate(sorted(all_results.items())):
    ax.plot(timesteps, results['step_losses'], 
            linewidth=2.5, 
            color=colors[idx],
            label=f'h={hidden_dim} ({results["num_parameters"]/1e6:.1f}M params)',
            alpha=0.8)

ax.set_xlabel('Rollout Step', fontsize=14)
ax.set_ylabel('MSE Loss', fontsize=14)
ax.set_title(f'Rollout Loss Comparison Across Hidden Dimensions (epoch={args.epoch})', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, args.num_rollout_steps + 1])
ax.legend(fontsize=11, loc='best')

# Save plot
plot_file = f'rollout_comparison_e{args.epoch}.png'
plt.tight_layout()
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nComparison plot saved to: {plot_file}")

# Print summary table
print("\nSummary:")
print(f"{'Hidden Dim':<12} {'Parameters':<15} {'Avg Rollout Loss':<20} {'Step 1 Loss':<15} {'Step 100 Loss':<15}")
print("-" * 85)
for hidden_dim, results in sorted(all_results.items()):
    print(f"{hidden_dim:<12} {results['num_parameters']/1e6:>6.2f}M        "
          f"{results['avg_total_loss']:>8.6f}           "
          f"{results['step_losses'][0]:>8.6f}        "
          f"{results['step_losses'][-1]:>8.6f}")

print("\nEvaluation complete!")