import torch
import torch.nn as nn
from utilities import FCNN, train, test, save_hidden_activations, mnist_loader

print(f"Using device: {device}")

# Small loaders for quick test
train_loader = mnist_loader(train=True,  n_samples=512, batch_size=64, shuffle=True)
test_loader = mnist_loader(
    train=False,
    batch_size=2,
    shuffle=False,
)
print(f"Test loader batches: {len(test_loader)}")
print(f"Test loader samples: {len(test_loader.dataset)}")