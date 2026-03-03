import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from utilities import AR_MLP_1_layer, AR_MLP_deep, save_hidden_activations, load_ks_data, train_ks, test_ks

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--model', type=str, default='AR_MLP_1_layer', choices=['AR_MLP_1_layer', 'AR_MLP_deep'])
parser.add_argument('--output-dir', type=str, default='default')
parser.add_argument('--train-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/train_KS_1024.npy')
parser.add_argument('--test-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/test_KS_1024.npy')
parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden layer dimension')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--n-samples', type=int, default=None, help='Number of training samples to use (None = use all)')
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
save_at_this_epoch = list(range(50, num_epochs + 1, 50))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, test_loader, input_dim, n_train, n_test = load_ks_data(
    args.train_data_path,
    args.test_data_path,
    batch_size=batch_size,
    num_workers=4,
    n_samples=args.n_samples
)

# Initialize model
if args.model == 'AR_MLP_1_layer':
    model = AR_MLP_1_layer(hidden_dim=args.hidden_dim).to(device)
elif args.model == 'AR_MLP_deep':
    model = AR_MLP_deep(hidden_dim=args.hidden_dim).to(device)
else:
    model = AR_MLP_deep(hidden_dim=args.hidden_dim).to(device)
    print(f"Unknown model: {args.model} ... Switching to Deep")

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
print(f"Testing samples: {n_test}")
print(f"Training samples (requested): {args.n_samples if args.n_samples else 'all'}")
print(f"Device: {device}")
print(f"Loss function: {loss_fn}")
print(f"Optimizer: AdamW")
print(f"Scheduler: ExponentialLR (gamma=0.975)")
print(f"Mixed precision: {'Enabled' if scaler else 'Disabled'}")
print(f"Total epochs: {num_epochs}")
print(f"Save at epochs: {save_at_this_epoch}")
print("="*60)

# Output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'),          exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'),          exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'activations'),      exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'),  exist_ok=True)

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
    'test_samples': n_test,
    'job_idx': args.job_idx,
    'mixed_precision': scaler is not None,
    'n_samples_requested': args.n_samples,
}

# Training loop
for epoch in range(1, num_epochs + 1):
    train_loss = train_ks(
        model, train_loader, loss_fn, optimizer, scheduler,
        device, epoch, dt=dt, scaler=scaler
    )

    if epoch in save_at_this_epoch:
        test_loss = test_ks(model, test_loader, loss_fn, device, dt=dt)

        json_input[f'epoch{epoch}_train_loss'] = float(train_loss)
        json_input[f'epoch{epoch}_test_loss']   = float(test_loss)
        json_input[f'epoch{epoch}_learning_rate'] = optimizer.param_groups[0]['lr']

        # Save weights
        weight_path = os.path.join(args.output_dir, 'weights', f'{model_name}_e{epoch}.pth')
        torch.save(model.state_dict(), weight_path)
        print(f"Model weights saved at epoch {epoch} to {weight_path}")

        # Save activations and singular values for both train and val
        acts_path, sv_path, sv_mean_train, sv_mean_test = save_hidden_activations(
            model, train_loader, test_loader, device, model_name, epoch, args.output_dir
        )

        json_input[f'epoch{epoch}_acts_path'] = acts_path
        json_input[f'epoch{epoch}_sv_path']   = sv_path
        json_input[f'epoch{epoch}_sv_mean_train'] = sv_mean_train
        json_input[f'epoch{epoch}_sv_mean_test']   = sv_mean_test

# Save metrics
with open(f'{args.output_dir}/metrics/{model_name}.json', 'w') as f:
    json.dump(json_input, f, indent=4)

print(f"\nConfig and Metrics saved to '{args.output_dir}/metrics/{model_name}.json'")

print("\n" + "="*60)
print("Training and singular value computation complete!")
print("="*60)