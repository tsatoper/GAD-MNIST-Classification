import os
import re
import glob
import argparse
import torch

from pytorch_ood.model import WideResNet
from utilities import CIFAR100Loader, compute_and_save_singular_values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WideResNet(
    num_classes=100,
    depth=28,
    widen_factor=1,
    drop_rate=0.3
).to(device)


# -----------------------------
# Data
# -----------------------------
loader = CIFAR100Loader(
    n_samples=100,
    batch_size=1024,
)
train_loader, test_loader = loader.get_loaders()
print("Number of test batches:", len(test_loader))
print("Dataset size:", len(test_loader.dataset))
