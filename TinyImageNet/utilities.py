import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCNN(nn.Module):
    """
    Simple stable CNN for Tiny ImageNet (64x64, 200 classes)
    """

    def __init__(self, width=64, num_classes=200):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            # 64x64
            block(3, width),
            block(width, width),
            nn.MaxPool2d(2),        # → 32x32

            block(width, 2 * width),
            block(2 * width, 2 * width),
            nn.MaxPool2d(2),        # → 16x16

            block(2 * width, 4 * width),
            block(4 * width, 4 * width),
            nn.MaxPool2d(2),        # → 8x8

            block(4 * width, 4 * width),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4 * width, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.classifier(feats)

        if return_features:
            return logits, feats
        return logits



def compute_and_save_singular_values(model, data_loader, device, filename, epoch, output_dir):
    # Compute and save singular values of hidden layer activations (penultimate features).

    print("\n" + "="*50)
    print(f"Computing singular values at epoch {epoch}...")
    print("="*50)
    
    model.eval()
    all_feats = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader, 1):
            data = data.to(device)
            _, feats = model(data, return_hidden=True)  # directly get features
            all_feats.append(feats.cpu())

            if batch_idx % 10 == 0 or batch_idx == len(data_loader):
                print(f'Processed batch {batch_idx}/{len(data_loader)}')

    # Concatenate features
    Phi = torch.cat(all_feats, dim=0)
    print(f"Collected hidden activations shape: {Phi.shape}")

    # Compute SVD
    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)

    # Print summary
    print(f"\nSingular values: {S[:5].numpy()}...{S[-5:].numpy()}")
        
    # Save singular values
    sv_path = os.path.join(output_dir, 'singular_values', f'{filename}_e{epoch}.pt')

    torch.save(S.cpu(), sv_path)
    print(f"Singular values saved to {sv_path}")
    
    return S, sv_path