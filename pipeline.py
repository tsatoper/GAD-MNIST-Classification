import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--output-dir', type=str, default='./outputs')
parser.add_argument('--loss-fn', type=str, default='mse', choices=['ce', 'mse'],
                    help='Loss function: ce (CrossEntropy) or mse (MSE)')
args = parser.parse_args()

hidden_dim_list = [i for i in range(1, 29 +1)] + [10*i for i in range(30//10, 70//10)] + [2**i for i in range(6, 25)]  # 0-28, 29-32, 33-51


hidden_dim = int(hidden_dim_list[args.job_idx])

num_epochs = 2000
samples = 4000
batch_size = 128
save_at_this_epoch = [500, 1000, 1500, num_epochs]
learning_rate = 1e-4

file_id = f'hidden_dim{hidden_dim}_epochs{num_epochs}_{args.loss_fn}'

class FCNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten (N, 1, 28, 28) -> (N, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCNN(hidden_dim=hidden_dim).to(device)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Running with hidden_dim = {hidden_dim}")
print(f"Running with loss function = {args.loss_fn}")
print(f"Running with parameters = {num_parameters}")


param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size
print(f"Model size: {total_size / 1e6:.2f} MB "
        f"({total_size / (1024**3):.2f} GB)")

json_input = {
    'num_epochs': num_epochs,
    'hidden_dim': hidden_dim,
    'batch_size': batch_size,
    'samples': samples,
    'num_parameters': num_parameters,
    'loss_function': args.loss_fn,
    'learning_rate': learning_rate
}

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))   # standard MNIST normalization
])
train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = Subset(train_dataset_full, range(samples))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = Subset(test_dataset_full, range(samples//4))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set loss function based on argument
if args.loss_fn == 'ce':
    loss_fn = nn.CrossEntropyLoss()
else:  # mse
    loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train(model, train_loader, loss_fn, optimizer, device, epoch, use_mse=True, n_classes=10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data)
        
        if use_mse:
            # Convert target to one-hot encoding for MSE loss
            target_one_hot = torch.zeros(target.size(0), n_classes, device=device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            loss = loss_fn(output, target_one_hot)
        else:
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy

def test(model, test_loader, loss_fn, device, use_mse=True, n_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            output = model(data)
            
            if use_mse:
                # Convert target to one-hot encoding for MSE loss
                target_one_hot = torch.zeros(target.size(0), n_classes, device=device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += loss_fn(output, target_one_hot).item()
            else:
                test_loss += loss_fn(output, target).item()
                
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy


print(f"Training on device: {device}")

use_mse = (args.loss_fn == 'mse')

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device, epoch, use_mse=use_mse)
    test_loss, test_acc = test(model, test_loader, loss_fn, device, use_mse=use_mse)
    if epoch in save_at_this_epoch:
        json_input[f'epoch{epoch}_train_loss'] = train_loss
        json_input[f'epoch{epoch}_test_loss'] = test_loss
        json_input[f'epoch{epoch}_train_acc'] = train_acc
        json_input[f'epoch{epoch}_test_acc'] = test_acc
        


# SAVING
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)

with open(f'{args.output_dir}/final_metrics_{file_id}.json', 'w') as f:
    json.dump(json_input, f, indent=4)

print(f"Config and Metrics saved to '{args.output_dir}/final_metrics_{file_id}.json'")

torch.save(model.state_dict(), f'{args.output_dir}/weights/mnist_{file_id}.pth')
print(f"Model saved to '{args.output_dir}/weights/mnist_{file_id}.pth'")