import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument('--job-idx', type=int, required=True)
parser.add_argument('--output-dir', type=str, default='./outputs')
args = parser.parse_args()

# hidden_dim_list = [5, 25, 50, 100, 500, 1000, 2000, 4000, 8000, 10000, 12000, 14000, 20000, 30000, 35000, 37000, 40000, 43000, 45000, 50000, 75000, 100000]
hidden_dim_list = [2e6, 5e6, 1e7, 2e7, 5e7, 1e8]

hidden_dim = int(hidden_dim_list[args.job_idx])
num_epochs = 30

file_id = f'{hidden_dim}_{num_epochs}'

print(f"Running with hidden_dim = {hidden_dim}")

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

model = FCNN(hidden_dim=hidden_dim)
print('model made')
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))   # standard MNIST normalization
])
train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = Subset(train_dataset_full, range(4000))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = Subset(test_dataset_full, range(1000))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

loss_fn = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
total_size = param_size + buffer_size
print(f"Model size: {total_size / 1e6:.2f} MB "
        f"({total_size / (1024**3):.2f} GB)")

def train(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Flatten images from [batch, 1, 28, 28] to [batch, 784]
        data = data.view(data.size(0), -1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy

def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Flatten images
            data = data.view(data.size(0), -1)
            
            # Forward pass
            output = model(data)
            test_loss += loss_fn(output, target).item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy

print(f"Training on device: {device}")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device, epoch)
    test_loss, test_acc = test(model, test_loader, loss_fn, device)

# SAVING
os.makedirs(args.output_dir, exist_ok=True)

metrics = {
    'final_train_loss': train_loss,
    'final_test_loss': test_loss,
    'final_train_acc': train_acc,
    'final_test_acc': test_acc,
    'num_epochs': num_epochs,
    'hidden_dim': hidden_dim
}
with open(f'{args.output_dir}/final_metrics_{file_id}.json', 'w') as f:
    json.dump(metrics, f, indent=4)

torch.save(model.state_dict(), f'{args.output_dir}/mnist_{file_id}.pth')
print(f"Model saved to '{args.output_dir}/mnist_{file_id}.pth'")