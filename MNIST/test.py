import os
import torch
from torchvision import datasets, transforms

def save_noised_mnist(
    noise=0.0,
    seed=0,
    train=True,
    original_root="/glade/derecho/scratch/tsatoperry/GAD/MNIST/data",
    new_root="/glade/derecho/scratch/tsatoperry/GAD/MNIST/data/original",
):
    """
    Creates and saves a deterministic label-noised MNIST dataset.
    """

    if not (0.0 <= noise <= 1.0):
        raise ValueError("noise must be in [0,1]")

    os.makedirs(new_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root=original_root,
        train=train,
        download=True,
        transform=transform
    )

    # --- deterministic label corruption ---
    g = torch.Generator()
    g.manual_seed(seed)

    targets = torch.tensor(dataset.targets)
    N = len(targets)
    n_corrupt = int(noise * N)

    perm = torch.randperm(N, generator=g)
    corrupt_idx = perm[:n_corrupt]

    num_classes = 10

    for idx in corrupt_idx:
        original = targets[idx].item()
        new_label = torch.randint(
            0, num_classes - 1, (1,), generator=g
        ).item()
        if new_label >= original:
            new_label += 1
        targets[idx] = new_label

    dataset.targets = targets.tolist()

    # --- save ---
    split = "train" if train else "test"
    save_path = os.path.join(new_root, f"mnist_{split}_noise{int(noise*100)}.pt")

    torch.save(dataset, save_path)

    print(f"Saved noised MNIST to:\n{save_path}")

if __name__ == "__main__":
    save_noised_mnist()