## ResNet18 for CIFAR
## Based on: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_hidden:
                    return out, hidden
        return out

def make_resnet18k(k=64, num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)


def cifar10_loader(
    train=True,
    n_samples=None,
    batch_size=128,
    root="/glade/derecho/scratch/tsatoperry/GAD/CIFAR10/data",
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
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
    )

    # --- Deterministic label corruption ---
    if noise > 0.0:
        g = torch.Generator()
        g.manual_seed(0)  # always same noise

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

    # --- Optional subsampling ---
    if n_samples is not None:
        if n_samples > len(dataset):
            raise ValueError(
                f"Requested {n_samples} samples, but dataset has {len(dataset)}"
            )

        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=g)
        dataset = Subset(dataset, perm[:n_samples])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if n_samples is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader

