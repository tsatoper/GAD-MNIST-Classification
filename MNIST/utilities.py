import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class FCNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.fc1.weight.mul_(10)
            self.fc2.weight.mul_(10)
            
    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)  # Flatten (N, 1, 28, 28) → (N, 784)
        x = self.fc1(x)
        hidden = F.relu(x)
        out = self.fc2(hidden)
        if return_hidden:
            return out, hidden
        return out

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def mnist_loader(
    train=True,
    n_samples=None,
    batch_size=128,
    root="/glade/derecho/scratch/tsatoperry/GAD/MNIST/data",
    seed=0,
    num_workers=0,
    pin_memory=False,
    shuffle=False,
    noise=0.0,  # fraction of labels to corrupt
):
    if not (0.0 <= noise <= 1.0):
        raise ValueError("noise must be in [0,1]")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )

    # --- Deterministic label corruption ---
    if noise > 0.0:
        g = torch.Generator()
        g.manual_seed(0)

        targets = torch.tensor(dataset.targets)
        N = len(targets)
        n_corrupt = int(noise * N)

        perm = torch.randperm(N, generator=g)
        corrupt_idx = perm[:n_corrupt].tolist()  # FIX: convert to plain ints

        num_classes = 10

        for idx in corrupt_idx:
            original = targets[idx].item()
            # FIX: sample from [0, num_classes - 2] then remap to avoid
            # the original class, giving a uniform distribution over the
            # other num_classes-1 classes instead of over-representing class 9.
            new_label = torch.randint(0, num_classes - 1, (1,), generator=g).item()
            if new_label >= original:
                new_label += 1
            targets[idx] = new_label

        dataset.targets = targets.tolist()

    # --- Optional subsampling ---
    if n_samples is not None:
        if n_samples > len(dataset):
            raise ValueError(
                f"Requested {n_samples} samples, but dataset has {len(dataset)}"
            )

        g = torch.Generator()
        g.manual_seed(seed)

        perm = torch.randperm(len(dataset), generator=g)
        dataset = Subset(dataset, perm[:n_samples].tolist())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,   # FIX: always respect caller's shuffle argument
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, n_classes=10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        target_one_hot = torch.zeros(target.size(0), n_classes, device=device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        loss = loss_fn(output, target_one_hot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%), LR = {lr:.6g}')
    return avg_loss, accuracy

def test(model, test_loader, loss_fn, device, n_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            target_one_hot = torch.zeros(target.size(0), n_classes, device=device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            test_loss += loss_fn(output, target_one_hot).item()
                
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy
    
def save_hidden_activations(model, train_loader, test_loader, device, filename, epoch, output_dir):
    # Save hidden layer activations (penultimate features) for train and test sets.

    print("\n" + "="*50)
    print(f"Saving hidden activations at epoch {epoch}...")
    print("="*50)
    
    model.eval()

    def collect_features(data_loader, split):
        all_feats = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader, 1):
                data = data.to(device)
                _, feats = model(data, return_hidden=True)
                all_feats.append(feats.cpu())

                if batch_idx % 10 == 0 or batch_idx == len(data_loader):
                    print(f'[{split}] Processed batch {batch_idx}/{len(data_loader)}')

        Phi = torch.cat(all_feats, dim=0)
        print(f"[{split}] Collected hidden activations shape: {Phi.shape}")
        return Phi

    train_feats = collect_features(train_loader, "train")
    test_feats  = collect_features(test_loader,  "test")

    acts_path = os.path.join(output_dir, 'activations', f'{filename}_e{epoch}.pt')
    torch.save({'train': train_feats, 'test': test_feats}, acts_path)
    print(f"Hidden activations saved to {acts_path}")

    # Compute and save singular values for train and test feature matrices
    print("\nComputing singular values...")
    sv_dict = {} 
    sv_mean = {}   
    for split, feats in [("train", train_feats), ("test", test_feats)]:
        S = torch.linalg.svdvals(feats)
        sv_dict[split] = S
        sv_mean[split] = S.mean()
        print(f"[{split}] Singular values shape: {S.shape}  |  bot 5: {S[-5:].tolist()}")

    sv_path = os.path.join(output_dir, 'singular_values', f'{filename}_e{epoch}.pt')
    torch.save(sv_dict, sv_path)
    print(f"Singular values saved to {sv_path}")

    return acts_path, sv_path, sv_mean
   


