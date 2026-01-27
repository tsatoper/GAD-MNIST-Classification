import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)  # Flatten (N, 1, 28, 28) → (N, 784)
        x = self.fc1(x)
        hidden = F.relu(x)

        if return_hidden:
            return self.fc2(hidden), hidden

        return self.fc2(hidden)

def train(model, train_loader, loss_fn, optimizer, device, epoch, n_classes=10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if isinstance(loss_fn, nn.MSELoss):
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

def test(model, test_loader, loss_fn, device, n_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            if isinstance(loss_fn, nn.MSELoss):
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


def compute_and_save_singular_values(model, test_loader, device, hidden_dim, epoch, output_dir):
    """Compute and save singular values of hidden layer activations."""
    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_hidden = []
    
    count = 0
    total_batches = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, hidden = model(data, return_hidden=True)
            all_hidden.append(hidden)
            count += 1
            if count % 10 == 0 or count == total_batches:
                print(f'Batch {count}/{total_batches}')
    
    all_hidden = torch.cat(all_hidden, dim=0)
    print(f"Hidden activations shape: {all_hidden.shape}")
    
    U, S, Vh = torch.linalg.svd(all_hidden, full_matrices=False)
    print(f"\nTop 10 singular values: {S[:10].cpu().numpy()}")
    
    # Save singular values
    sv_path = os.path.join(output_dir, 'singular_values', f'hidden_dim{hidden_dim}_epoch{epoch}.pt')

    torch.save(S.cpu(), sv_path)
    print(f"Singular values saved to {sv_path}")
    
    return S


if __name__=="__main__":
    model = FCNN(hidden_dim=1)
    
