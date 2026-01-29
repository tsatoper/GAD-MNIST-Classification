import os
import sys
import argparse
import json
import torch
import torch.nn as nn

from utilities import MLP_AR, compute_and_save_singular_values, load_ks_data

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--output-dir', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/models/default')
parser.add_argument('--train-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/train_KS_1024.npy')
parser.add_argument('--val-data-path', type=str, default='/glade/derecho/scratch/tsatoperry/GAD/KS_1d/training_data/val_KS_1024.npy')
parser.add_argument('--hidden1-dim', type=int, default=512, help='First hidden layer dimension')
parser.add_argument('--hidden2-dim', type=int, default=512, help='Second hidden layer dimension')
args = parser.parse_args()

model_name = f'h1{args.hidden1_dim}_h2{args.hidden2_dim}'

# Configuration
num_epochs = 300
batch_size = 1024
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, input_dim, n_train, n_val = load_ks_data(
    args.train_data_path,
    args.val_data_path,
    batch_size=batch_size,
    num_workers=4
)

output_dim = input_dim  # 1024

model = MLP_AR(
    input_dim=input_dim,
    hidden1_dim=args.hidden1_dim,
    hidden2_dim=args.hidden2_dim,
    output_dim=output_dim
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_parameters:,} ({num_parameters/1e6:.2f}M)")

# Output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'singular_values'), exist_ok=True)

# Training loop
json_input = {
    'num_epochs': num_epochs,
    'hidden1_dim': args.hidden1_dim,
    'hidden2_dim': args.hidden2_dim,
    'input_dim': input_dim,
    'output_dim': output_dim,
    'batch_size': batch_size,
    'num_parameters': num_parameters,
    'learning_rate': learning_rate,
    'train_samples': n_train,
    'val_samples': n_val,
    'job_idx': args.job_idx
}

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}')
    json_input[f'epoch{epoch}_train_loss'] = train_loss

    # Validate and compute singular values every 100 epochs
    if epoch % 100 == 0:
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'  Val Loss: {val_loss:.6f}')

        json_input[f'epoch{epoch}_val_loss'] = val_loss

        # Save weights
        torch.save(model.state_dict(), f'{args.output_dir}/weights/{model_name}_e{epoch}.pth')

        # Compute and save singular values
        S = compute_and_save_singular_values(model, train_loader, device, model_name, epoch, args.output_dir)
        json_input[f'epoch{epoch}_sv_max'] = float(S[0].cpu())
        json_input[f'epoch{epoch}_sv_min'] = float(S[-1].cpu())

# Save metrics
with open(f'{args.output_dir}/metrics/{model_name}.json', 'w') as f:
    json.dump(json_input, f, indent=4)

print(f"\nMetrics saved to '{args.output_dir}/metrics/{model_name}.json'")
print("\n" + "="*50)
print("Training complete!")
print("="*50)