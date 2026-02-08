import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, scaler=None, n_classes=100):
    """
    Training function with mixed precision support.
    
    Args:
        scaler: GradScaler for mixed precision training (None to disable)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast(device_type='cuda'):
                output = model(data)
                loss = loss_fn(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    # Step scheduler after epoch
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%), LR: {current_lr:.6f}')
    return avg_loss, accuracy


def test(model, test_loader, loss_fn, device, n_classes=100):
    """Test/validation function."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += loss_fn(output, target).item()
                
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy

    
def compute_and_save_singular_values(model, data_loader, device, filename, epoch, output_dir):
    """
    Compute and save singular values of penultimate layer activations.
    For WideResNet, we extract features before the final FC layer.
    """

    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_feats = []

    # Create a hook to extract penultimate features
    features = []
    def hook_fn(module, input, output):
        # Extract features before FC layer (after global pooling and flatten)
        features.append(input[0].detach())
    
    # Register hook on the final linear layer
    hook = model.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader, 1):
            data = data.to(device)
            features = []
            _ = model(data)  # Forward pass triggers the hook
            all_feats.append(features[0].cpu())

            if batch_idx % 50 == 0 or batch_idx == len(data_loader):
                print(f'Processed batch {batch_idx}/{len(data_loader)}')

    # Remove hook
    hook.remove()

    Phi = torch.cat(all_feats, dim=0)
    print(f"Collected hidden activations shape: {Phi.shape}")

    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")
        
    sv_path = os.path.join(output_dir, 'singular_values', f'{filename}_e{epoch}.pt')

    torch.save(S.cpu(), sv_path)
    print(f"Singular values saved to {sv_path}")
    
    return S, sv_path