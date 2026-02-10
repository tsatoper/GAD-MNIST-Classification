import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from utilities import AR_MLP_1_layer, AR_MLP_deep, compute_and_save_singular_values, load_ks_data, train_ks, test_ks, train_ks_noEuler

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--model', type=str, default='AR_MLP_1_layer', choices=['AR_MLP_1_layer', 'AR_MLP_deep'])
parser.add_argument('--output-dir', type=str, default='default')
parser.add_argument('--train-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/train_KS_1024.npy')
parser.add_argument('--val-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy')
parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden layer dimension')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
args = parser.parse_args()


# Setup paths
args.output_dir = os.path.join(
    '/glade/derecho/scratch/tsatoperry/GAD/KS_1d',
    args.model[7:],
    args.output_dir
)
model_name = f'h_{args.hidden_dim}_job{args.job_idx}'
print('Training model: ', model_name)
# Configuration
num_epochs = args.epochs
batch_size = 1024
learning_rate = 1e-3
dt = 1e-3
save_interval = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, input_dim, n_train, n_val = load_ks_data(
    args.train_data_path,
    args.val_data_path,
    batch_size=batch_size,
    num_workers=4
)

# Initialize model
if args.model == 'AR_MLP_1_layer':
    model = AR_MLP_1_layer(hidden_dim=args.hidden_dim).to(device)
elif args.model == 'AR_MLP_deep':
    model = AR_MLP_deep(hidden_dim=args.hidden_dim).to(device)
else:
    raise ValueError(f"Unknown model: {args.model}")

# Setup training
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
scaler = GradScaler() if torch.cuda.is_available() else None

# Print configuration
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size

print("="*60)
print(f"Kuramoto-Sivashinsky Training - {args.model}")
print("="*60)
print(f"Model: {args.model}")
print(f"Hidden dimension: {args.hidden_dim}")
print(f"Input dimension: {input_dim}")
print(f"Parameters: {num_parameters:,} ({num_parameters/1e6:.2f}M)")
print(f"Model size: {total_size / 1e6:.2f} MB ({total_size / (1024**3):.4f} GB)")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Time step (dt): {dt}")
print(f"Training samples: {n_train}")
print(f"Validation samples: {n_val}")
print(f"Device: {device}")
print(f"Loss function: {loss_fn}")
print(f"Optimizer: AdamW")
print(f"Scheduler: ExponentialLR (gamma=0.975)")
print(f"Mixed precision: {'Enabled' if scaler else 'Disabled'}")
print(f"Total epochs: {num_epochs}")
print(f"Save interval: {save_interval} epochs")
print("="*60)

# Output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)

# Initialize metrics storage
json_input = {
    'model': args.model,
    'num_epochs': num_epochs,
    'hidden_dim': args.hidden_dim,
    'input_dim': input_dim,
    'batch_size': batch_size,
    'num_parameters': num_parameters,
    'loss_function': str(loss_fn),
    'learning_rate': learning_rate,
    'dt': dt,
    'train_samples': n_train,
    'val_samples': n_val,
    'job_idx': args.job_idx,
    'mixed_precision': scaler is not None
}

# Training loop
for epoch in range(1, num_epochs + 1):
    # Train
    train_loss = train_ks_noEuler(
        model, train_loader, loss_fn, optimizer, scheduler,
        device, epoch, dt=dt, scaler=scaler
    )
    
    json_input[f'epoch{epoch}_train_loss'] = train_loss

    # Validate and save checkpoints
    if epoch % save_interval == 0 or epoch == num_epochs:
        # Validation
        val_loss = test_ks(model, val_loader, loss_fn, device, dt=dt)
        json_input[f'epoch{epoch}_val_loss'] = val_loss

        # Save weights
        weight_path = f'{args.output_dir}/weights/{model_name}_e{epoch}.pth'
        torch.save(model.state_dict(), weight_path)
        print(f"Model weights saved at epoch {epoch} to {weight_path}")

        # Compute and save singular values
        S, sv_path = compute_and_save_singular_values(
            model, val_loader, device, model_name+'test_set', epoch, args.output_dir
        )
        S, sv_path = compute_and_save_singular_values(
            model, train_loader, device, model_name, epoch, args.output_dir
        )
        print(f"Singular Values saved at epoch {epoch} to {sv_path}")
        
        json_input[f'epoch{epoch}_sv_max'] = float(S[0].cpu())
        json_input[f'epoch{epoch}_sv_min'] = float(S[-1].cpu())
        json_input[f'epoch{epoch}_sv_path'] = sv_path
        json_input[f'epoch{epoch}_learning_rate'] = optimizer.param_groups[0]['lr']

# Save metrics
with open(f'{args.output_dir}/metrics/{model_name}.json', 'w') as f:
    json.dump(json_input, f, indent=4)

print(f"\nConfig and Metrics saved to '{args.output_dir}/metrics/{model_name}.json'")

print("\n" + "="*60)
print("Training and singular value computation complete!")
print("="*60)