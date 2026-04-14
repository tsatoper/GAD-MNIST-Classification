import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Fixed epochs: early and overfit
EARLY_EPOCH = 1
OVERFIT_EPOCH = 2000

yscale = 'linear'
xscale = 'log'
save_path = f'loss_comparison_{yscale}.png'

# Colors: early, optimal, overfit
colors = {
    'early':   '#ADD8E6',   # light blue
    'optimal': '#FF6B6B',   # red
    'overfit': '#00008B',   # dark blue
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

dir_name = './models/main/metrics'

# Per-role data: {role: {width, test_losses, train_losses}}
role_data = {r: {'width': [], 'test_losses': [], 'train_losses': []}
             for r in ('early', 'optimal', 'overfit')}

if not os.path.exists(dir_name):
    print(f"Error: Directory '{dir_name}' not found.")
else:
    all_widths = {}  # {width: {epoch: {train_loss, test_loss}}}

    for filename in os.listdir(dir_name):
        if filename.endswith('.json'):
            filepath = os.path.join(dir_name, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                width = data['width']
                if width < 8:
                    continue

                if width not in all_widths:
                    all_widths[width] = {}

                for key in data.keys():
                    if 'epoch' not in key:
                        continue
                    for field, suffix in [('test_loss', '_test_loss'),
                                          ('train_loss', '_train_loss')]:
                        if key.endswith(suffix):
                            parts = key.replace('epoch', '').replace(suffix, '')
                            try:
                                epoch = int(parts)
                                if epoch not in all_widths[width]:
                                    all_widths[width][epoch] = {}
                                all_widths[width][epoch][field] = data[key]
                            except ValueError:
                                pass
                            break

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                continue

    for width in sorted(all_widths.keys()):
        # Find optimal epoch for this width
        best_epoch, best_test_loss = None, float('inf')
        for epoch, ed in all_widths[width].items():
            if 'test_loss' in ed and ed['test_loss'] < best_test_loss:
                best_test_loss = ed['test_loss']
                best_epoch = epoch

        # Build epoch→role mapping with collapsing logic
        epoch_roles = {}
        if best_epoch == EARLY_EPOCH:
            epoch_roles[EARLY_EPOCH]  = 'optimal'
            epoch_roles[OVERFIT_EPOCH] = 'overfit'
        elif best_epoch == OVERFIT_EPOCH:
            epoch_roles[EARLY_EPOCH]   = 'early'
            epoch_roles[OVERFIT_EPOCH] = 'optimal'
        else:
            epoch_roles[EARLY_EPOCH]   = 'early'
            epoch_roles[best_epoch]    = 'optimal'
            epoch_roles[OVERFIT_EPOCH] = 'overfit'

        for epoch, role in epoch_roles.items():
            if epoch in all_widths[width]:
                ed = all_widths[width][epoch]
                if 'test_loss' in ed and 'train_loss' in ed:
                    role_data[role]['width'].append(width)
                    role_data[role]['test_losses'].append(ed['test_loss'])
                    role_data[role]['train_losses'].append(ed['train_loss'])

    # Sort each role by width
    for role in role_data:
        if len(role_data[role]['width']) > 0:
            idx = np.argsort(role_data[role]['width'])
            for key in role_data[role]:
                role_data[role][key] = np.array(role_data[role][key])[idx]

    labels = {'early': f'Epoch {EARLY_EPOCH}',
              'optimal': 'Optimal Stopping',
              'overfit': f'Epoch {OVERFIT_EPOCH}'}
    markers = {'early': 'o', 'optimal': 's', 'overfit': 'o'}

    for role in ('early', 'optimal', 'overfit'):
        if len(role_data[role]['width']) > 0:
            w = role_data[role]['width']
            ax.plot(w, role_data[role]['test_losses'],
                    f"{markers[role]}-", linewidth=2,
                    label=f'{labels[role]} - Test Loss',
                    color=colors[role])
            ax.plot(w, role_data[role]['train_losses'],
                    f"{markers[role]}--", linewidth=2,
                    label=f'{labels[role]} - Train Loss',
                    color=colors[role], alpha=0.7)

    ax.set_xlabel('Model Width', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('n=50,000 samples', fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

fig.suptitle('ResNet-18 CIFAR-10 — MSE Loss during Training vs Model Width',
             fontsize=13, y=0.995)
plt.tight_layout()

plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')
