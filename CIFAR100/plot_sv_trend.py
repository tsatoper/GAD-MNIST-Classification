import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
import tempfile
from matplotlib.lines import Line2D

# ====== CONFIGURATION ======
model_dir = 'n_50000'
depth     = 28
plot_id   = f'{model_dir}_depth{depth}'
epochs    = [200] #list(range(50, 200+1, 50))
yscale    = 'log'
save_path = f'loss_sv_cifar_{plot_id}_animated.gif'
fps       = 1

base_dir   = f'/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models/{model_dir}/depth{depth}'
dir_name   = os.path.join(base_dir, 'metrics')
sv_dir     = os.path.join(base_dir, 'singular_values')

print(f"Reading metrics : {dir_name}")
print(f"Reading SVs     : {sv_dir}")

# ====== LOAD ALL DATA UPFRONT ======
# all_data[epoch] = {width, test_losses, train_losses, sv_train_mean, sv_test_mean}
all_data = {epoch: {'width': [], 'test_losses': [], 'train_losses': [],
                    'sv_train_mean': [], 'sv_test_mean': []}
            for epoch in epochs}
n_samples = None

if not os.path.isdir(dir_name):
    raise FileNotFoundError(f"Metrics directory not found: {dir_name}")

# Filename pattern matches CIFAR metrics: wrn28_<width>_job<jobnum>.json
metrics_pattern = re.compile(r'wrn28_(\d+)_job(\d+)\.json')

for filename in os.listdir(dir_name):
    m = metrics_pattern.match(filename)
    if not m:
        continue

    width    = int(m.group(1))
    filepath = os.path.join(dir_name, filename)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if n_samples is None and 'samples' in data:
            n_samples = data['samples']

        for epoch in epochs:
            train_key    = f'epoch{epoch}_train_loss'
            test_key     = f'epoch{epoch}_test_loss'
            sv_train_key = f'epoch{epoch}_train_sv_mean'
            sv_test_key  = f'epoch{epoch}_test_sv_mean'

            if train_key not in data or test_key not in data:
                continue

            sv_train = data.get(sv_train_key, None)
            sv_test  = data.get(sv_test_key,  None)

            # Fall back to loading .pt files if sv_mean keys not in JSON
            if sv_train is None or sv_test is None:
                job_num = m.group(2)

                # Train SV file: wrn28_<width>_job<job>_e<epoch>.pt
                sv_train_fname = f'wrn28_{width}_job{job_num}_e{epoch}.pt'
                sv_train_path  = os.path.join(sv_dir, sv_train_fname)

                # Test SV file: wrn28_<width>_job<job>_e200_test_e<epoch>.pt
                # (mirrors the pattern used in the static plot script)
                sv_test_fname  = f'wrn28_{width}_job{job_num}_e200_test_e{epoch}.pt'
                sv_test_path   = os.path.join(sv_dir, sv_test_fname)

                if sv_train is None and os.path.exists(sv_train_path):
                    sv_t = torch.load(sv_train_path, map_location='cpu', weights_only=True)
                    sv_train = float(sv_t.mean())

                if sv_test is None and os.path.exists(sv_test_path):
                    sv_t = torch.load(sv_test_path, map_location='cpu', weights_only=True)
                    sv_test = float(sv_t.mean())

            if sv_train is None and sv_test is None:
                continue

            all_data[epoch]['width'].append(width)
            all_data[epoch]['train_losses'].append(data[train_key])
            all_data[epoch]['test_losses'].append(data[test_key])
            all_data[epoch]['sv_train_mean'].append(sv_train if sv_train is not None else float('nan'))
            all_data[epoch]['sv_test_mean'].append(sv_test  if sv_test  is not None else float('nan'))

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {filename}: {e}")
        continue

# Sort by width for each epoch
for epoch in epochs:
    d = all_data[epoch]
    if len(d['width']) > 0:
        idx = np.argsort(d['width'])
        d['width']         = np.array(d['width'])[idx]
        d['test_losses']   = np.array(d['test_losses'])[idx]
        d['train_losses']  = np.array(d['train_losses'])[idx]
        d['sv_train_mean'] = np.array(d['sv_train_mean'])[idx]
        d['sv_test_mean']  = np.array(d['sv_test_mean'])[idx]

# ====== COMPUTE GLOBAL AXIS LIMITS for stable animation ======
all_test  = np.concatenate([d['test_losses']  for d in all_data.values() if len(d['test_losses'])  > 0])
all_train = np.concatenate([d['train_losses'] for d in all_data.values() if len(d['train_losses']) > 0])
all_sv    = np.concatenate([
    np.concatenate([d['sv_train_mean'], d['sv_test_mean']])
    for d in all_data.values() if len(d['sv_train_mean']) > 0
])
all_sv = all_sv[~np.isnan(all_sv)]

loss_ymin = min(np.min(all_test), np.min(all_train)) * 0.8
loss_ymax = max(np.max(all_test), np.max(all_train)) * 1.2
sv_ymin   = np.min(all_sv) * 0.8
sv_ymax   = np.max(all_sv) * 1.2

TEST_COLOR  = 'red'
TRAIN_COLOR = 'blue'

# ====== GENERATE FRAMES ======
print(f"Generating {len(epochs)} frames...")
frame_paths = []
tmpdir = tempfile.mkdtemp()

for epoch in epochs:
    d = all_data[epoch]
    if len(d['width']) == 0:
        print(f"  Epoch {epoch}: no data, skipping frame.")
        continue

    fig, ax1 = plt.subplots(figsize=(11, 5))

    width         = d['width']
    test_losses   = d['test_losses']
    train_losses  = d['train_losses']
    sv_train_mean = d['sv_train_mean']
    sv_test_mean  = d['sv_test_mean']

    # ---- Left axis: losses ----
    ax1.plot(width, test_losses,  linestyle='-', linewidth=2, alpha=0.8, color=TEST_COLOR)
    ax1.plot(width, train_losses, linestyle='-', linewidth=2, alpha=0.8, color=TRAIN_COLOR)
    ax1.scatter(width, test_losses,  s=40, alpha=0.8, zorder=5, color=TEST_COLOR)
    ax1.scatter(width, train_losses, s=40, alpha=0.8, zorder=5, color=TRAIN_COLOR)

    # if n_samples is not None:
        # ax1.axvline(x=n_samples, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Model Width', fontsize=11)
    ax1.set_ylabel(f'Loss ({yscale} scale)', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_xscale('linear')   # WRN widths are linear multiples (k=1,2,...)
    ax1.set_yscale(yscale)
    ax1.set_ylim(loss_ymin, loss_ymax)
    ax1.grid(True, alpha=0.3)

    # ---- Right axis: mean SVs ----
    ax2 = ax1.twinx()

    valid_train = ~np.isnan(sv_train_mean)
    valid_test  = ~np.isnan(sv_test_mean)

    if valid_train.any():
        ax2.plot(width[valid_train], sv_train_mean[valid_train],
                 linestyle='--', linewidth=2.5, alpha=0.7, color=TRAIN_COLOR)
        ax2.scatter(width[valid_train], sv_train_mean[valid_train],
                    s=60, alpha=0.7, marker='s', zorder=5, color=TRAIN_COLOR)
    if valid_test.any():
        ax2.plot(width[valid_test], sv_test_mean[valid_test],
                 linestyle='--', linewidth=2.5, alpha=0.7, color=TEST_COLOR)
        ax2.scatter(width[valid_test], sv_test_mean[valid_test],
                    s=60, alpha=0.7, marker='D', zorder=5, color=TEST_COLOR)

    ax2.set_ylabel('Mean Singular Value', fontsize=11, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
    ax2.set_yscale('log')
    ax2.set_ylim(sv_ymin, sv_ymax)

    # ---- Legend ----
    legend_elements = [
        Line2D([0], [0], color=TEST_COLOR,  linewidth=2,   linestyle='-',  label='Test Loss'),
        Line2D([0], [0], color=TRAIN_COLOR, linewidth=2,   linestyle='-',  label='Train Loss'),
        Line2D([0], [0], color=TRAIN_COLOR, linewidth=2.5, linestyle='--', label='Train Mean SV'),
        Line2D([0], [0], color=TEST_COLOR,  linewidth=2.5, linestyle='--', label='Test Mean SV'),
    ]
    if n_samples is not None:
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label=f'n_samples={n_samples}')
        )
    ax1.legend(handles=legend_elements, fontsize=8, loc='lower left')

    plt.title(f'CIFAR-100  |  WRN-{depth}  |  {model_dir}  |  Epoch {epoch}',
              fontsize=13, fontweight='bold')
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