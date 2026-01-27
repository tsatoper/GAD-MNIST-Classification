import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SimpleWideCNN(nn.Module):
    """
    Width-parameterized CNN for Tiny ImageNet (64x64, 200 classes).
    """
    def __init__(self, width=64, num_classes=200):
        super().__init__()

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)]
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, width),
            conv_block(width, width),
            nn.MaxPool2d(2),          # 64x64 → 32x32

            conv_block(width, 2 * width),
            conv_block(2 * width, 2 * width),
            nn.MaxPool2d(2),          # 32x32 → 16x16

            conv_block(2 * width, 4 * width),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4 * width, num_classes)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.classifier(feats)

        if return_features:
            return logits, feats
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                nn.init.zeros_(m.bias)


def compute_and_save_singular_values(model, data_loader, device, model_name, epoch, output_dir):
    # Compute and save singular values of hidden layer activations (penultimate features).

    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_feats = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader, 1):
            data = data.to(device)
            _, feats = model(data, return_features=True)  # directly get features
            all_feats.append(feats.cpu())

            if batch_idx % 10 == 0 or batch_idx == len(data_loader):
                print(f'Processed batch {batch_idx}/{len(data_loader)}')

    # Concatenate features
    Phi = torch.cat(all_feats, dim=0)
    print(f"Collected hidden activations shape: {Phi.shape}")

    # Center and normalize
    Phi = Phi - Phi.mean(dim=0, keepdim=True)
    Phi = Phi / Phi.shape[0]**0.5

    # Compute SVD
    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)

    # Print summary
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")

    # Save singular values
    sv_dir = os.path.join(output_dir, 'singular_values')
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, f'{model_name}_e{epoch}.pt')
    torch.save(S, sv_path)
    print(f"Singular values saved to {sv_path}")

    return S
