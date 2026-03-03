import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import tempfile
from matplotlib.lines import Line2D

# ====== CONFIGURATION ======
plot_id = 'N1_1e-4'
epochs = list(range(50, 2000, 50))
yscale = 'log'
save_path = f'loss_sv_{plot_id}_animated.gif'
fps = 4  # frames per second

# ====== LOAD ALL DATA UPFRONT ======
dir_name = f'./models/{plot_id}/metrics'
sv_dir = f'./models/{plot_id}/singular_values'
print(f"Reading: {dir_name}")

# all_data[epoch] = {width, test_losses, train_losses, sv_min}
all_data = {epoch: {'width': [], 'test_losses': [], 'train_losses': [], 'sv_min': []}
            for epoch in epochs}
n_samples = None

if not os.path.isdir(dir_name):
    raise FileNotFoundError(f"Directory not found: {dir_name}")

for filename in os.listdir(dir_name):
    if not filename.endswith('.json'):
        continue
    filepath = os.path.join(dir_name, filename)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        width = data['width']

        if n_samples is None and 'samples' in data:
            n_samples = data['samples']

        for epoch in epochs:
            train_key = f'epoch{epoch}_train_loss'
            test_key  = f'epoch{epoch}_test_loss'
            if train_key not in data or test_key not in data:
                continue

            sv_filename = filename.replace('.json', f'_e{epoch}.pt')
            sv_filepath = os.path.join(sv_dir, sv_filename)
            if not os.path.exists(sv_filepath):
                continue

            sv = torch.load(sv_filepath, weights_only=True)['train']
            sv_mean = float(sv.mean())

            all_data[epoch]['width'].append(width)
            all_data[epoch]['train_losses'].append(data[train_key])
            all_data[epoch]['test_losses'].append(data[test_key])
            all_data[epoch]['sv_min'].append(sv_mean)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {filename}: {e}")
        continue

# Sort by width for each epoch
for epoch in epochs:
    d = all_data[epoch]
    if len(d['width']) > 0:
        idx = np.argsort(d['width'])
        d['width']        = np.array(d['width'])[idx]
        d['test_losses']  = np.array(d['test_losses'])[idx]
        d['train_losses'] = np.array(d['train_losses'])[idx]
        d['sv_min']       = np.array(d['sv_min'])[idx]

# ====== COMPUTE GLOBAL AXIS LIMITS for stable animation ======
all_widths_flat = np.unique(np.concatenate(
    [d['width'] for d in all_data.values() if len(d['width']) > 0]
))
all_test   = np.concatenate([d['test_losses']  for d in all_data.values() if len(d['test_losses'])  > 0])
all_train  = np.concatenate([d['train_losses'] for d in all_data.values() if len(d['train_losses']) > 0])
all_sv     = np.concatenate([d['sv_min']       for d in all_data.values() if len(d['sv_min'])       > 0])

loss_ymin = min(np.min(all_test), np.min(all_train)) * 0.8
loss_ymax = max(np.max(all_test), np.max(all_train)) * 1.2
sv_ymin   = np.min(all_sv) * 0.8
sv_ymax   = np.max(all_sv) * 1.2

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(all_widths_flat), vmax=max(all_widths_flat))

# ====== GENERATE FRAMES ======
print(f"Generating {len(epochs)} frames...")
frame_paths = []
tmpdir = tempfile.mkdtemp()

for epoch in epochs:
    d = all_data[epoch]
    if len(d['width']) == 0:
        print(f"  Epoch {epoch}: no data, skipping frame.")
        continue

    fig, ax1 = plt.subplots(figsize=(10, 5))

    width        = d['width']
    test_losses  = d['test_losses']
    train_losses = d['train_losses']
    sv_min       = d['sv_min']

    # Left axis: losses
    for i, w in enumerate(width):
        color = cmap(norm(w))
        if i < len(width) - 1:
            ax1.plot(width[i:i+2], test_losses[i:i+2],  linestyle='-',  linewidth=2,   alpha=0.8, color=color)
            ax1.plot(width[i:i+2], train_losses[i:i+2], linestyle=':',  linewidth=2,   alpha=0.6, color=color)
        ax1.scatter(w, test_losses[i],  s=40, alpha=0.8, zorder=5, color=color)
        ax1.scatter(w, train_losses[i], s=40, alpha=0.6, marker='^', zorder=5, color=color)

    if n_samples is not None:
        ax1.axvline(x=n_samples, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Model Width', fontsize=11)
    ax1.set_ylabel(f'Loss ({yscale})', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_xscale('log')
    ax1.set_yscale(yscale)
    ax1.set_ylim(loss_ymin, loss_ymax)
    ax1.grid(True, alpha=0.3)

    # Right axis: mean SV
    ax2 = ax1.twinx()
    for i, w in enumerate(width):
        color = cmap(norm(w))
        if i < len(width) - 1:
            ax2.plot(width[i:i+2], sv_min[i:i+2], linestyle='-.', linewidth=2.5, alpha=0.7, color=color)
        ax2.scatter(w, sv_min[i], s=60, alpha=0.7, marker='s', zorder=5, color=color)

    ax2.set_ylabel('Mean Singular Value', fontsize=11, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
    ax2.set_yscale('log')
    ax2.set_ylim(sv_ymin, sv_ymax)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2,   linestyle='-',  label='Test Loss'),
        Line2D([0], [0], color='gray', linewidth=2,   linestyle=':',  label='Train Loss'),
        Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-.', label='Mean SV'),
        Line2D([0], [0], color='red',  linewidth=1.5, linestyle='--', label=f'n_samples={n_samples}'),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc='lower left')

    plt.title(f'{plot_id}  |  Epoch {epoch}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    frame_path = os.path.join(tmpdir, f'frame_{epoch:05d}.png')
    plt.savefig(frame_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    frame_paths.append(frame_path)
    print(f"  Saved frame: epoch {epoch}")

# ====== ASSEMBLE GIF ======
print(f"\nAssembling GIF with {len(frame_paths)} frames at {fps} fps...")
frames = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(save_path, frames, fps=fps, loop=0)
print(f'Saved GIF to "{save_path}"')