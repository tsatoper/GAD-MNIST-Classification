from pytorch_ood.model import WideResNet

# Create a CIFAR-100 WRN

model = WideResNet(
    num_classes=100,
    depth=28,         # you can choose 28, 40, etc.
    widen_factor=10,    # you can set any integer: 2, 4, 6, 10...
    drop_rate=0.3
)

# This should work as-is, but you can verify with:
print(list(model.named_children())[-1])