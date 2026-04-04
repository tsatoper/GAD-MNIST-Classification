import os
import json
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tempfile
from matplotlib.lines import Line2D

# ====== CONFIGURATION ======
plot_id = 'testing'
epochs = list(range(50, 2000, 50))
yscale = 'log'
save_path = f'loss_sv_{plot_id}_animated.gif'
fps = 4  # frames per second

# ====== LOAD ALL DATA UPFRONT ======
dir_name = f'/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/{plot_id}/metrics'
print(f"Reading: {dir_name}")

# all_data[epoch] = {width, test_losses, train_losses, sv_train_mean, sv_test_mean}
all_data = {epoch: {'width': [], 'test_losses': [], 'train_losses': [],
                    'sv_train_mean': [], 'sv_test_mean': []}
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
            train_key      = f'epoch{epoch}_train_loss'
            test_key       = f'epoch{epoch}_test_loss'
            sv_train_key   = f'epoch{epoch}_train_sv_mean'
            sv_test_key    = f'epoch{epoch}_test_sv_mean'

            # Require at least loss keys; SV keys are optional
            if train_key not in data or test_key not in data:
                continue

            sv_train = data.get(sv_train_key, None)
            sv_test  = data.get(sv_test_key,  None)

            # Skip epoch if neither SV is available
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
all_widths_flat = np.unique(np.concatenate(
    [d['width'] for d in all_data.values() if len(d['width']) > 0]
))
all_test  = np.concatenate([d['test_losses']   for d in all_data.values() if len(d['test_losses'])   > 0])
all_train = np.concatenate([d['train_losses']  for d in all_data.values() if len(d['train_losses'])  > 0])
all_sv    = np.concatenate([
    np.concatenate([d['sv_train_mean'], d['sv_test_mean']])
    for d in all_data.values() if len(d['sv_train_mean']) > 0
])
all_sv = all_sv[~np.isnan(all_sv)]

loss_ymin = min(np.min(all_test), np.min(all_train)) * 0.8
loss_ymax = max(np.max(all_test), np.max(all_train)) * 1.2
sv_ymin   = np.min(all_sv) * 0.8
sv_ymax   = np.max(all_sv) * 1.2


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

    width          = d['width']
    test_losses    = d['test_losses']
    train_losses   = d['train_losses']
    sv_train_mean  = d['sv_train_mean']
    sv_test_mean   = d['sv_test_mean']

    TEST_COLOR  = 'red'
    TRAIN_COLOR = 'blue'

    # ---- Left axis: losses ----
    ax1.plot(width, test_losses,  linestyle='-',  linewidth=2,   alpha=0.8, color=TEST_COLOR,  label='Test Loss')
    ax1.plot(width, train_losses, linestyle='-',  linewidth=2,   alpha=0.8, color=TRAIN_COLOR, label='Train Loss')
    ax1.scatter(width, test_losses,  s=40, alpha=0.8, zorder=5, color=TEST_COLOR)
    ax1.scatter(width, train_losses, s=40, alpha=0.8, zorder=5, color=TRAIN_COLOR)

    if n_samples is not None:
        ax1.axvline(x=n_samples, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Model Width', fontsize=11)
    ax1.set_ylabel(f'Loss ({yscale} scale)', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_xscale('log')
    ax1.set_yscale(yscale)
    ax1.set_ylim(loss_ymin, loss_ymax)
    ax1.grid(True, alpha=0.3)

    # ---- Right axis: mean SVs (train + test) ----
    ax2 = ax1.twinx()

    valid_train = ~np.isnan(sv_train_mean)
    valid_test  = ~np.isnan(sv_test_mean)

    if valid_train.any():
        ax2.plot(width[valid_train], sv_train_mean[valid_train], linestyle='--', linewidth=2.5, alpha=0.7, color=TRAIN_COLOR, label='Train Mean SV')
        ax2.scatter(width[valid_train], sv_train_mean[valid_train], s=60, alpha=0.7, marker='s', zorder=5, color=TRAIN_COLOR)
    if valid_test.any():
        ax2.plot(width[valid_test],  sv_test_mean[valid_test],  linestyle='--', linewidth=2.5, alpha=0.7, color=TEST_COLOR,  label='Test Mean SV')
        ax2.scatter(width[valid_test],  sv_test_mean[valid_test],  s=60, alpha=0.7, marker='D', zorder=5, color=TEST_COLOR)

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

    plt.title(f'MNIST |  {plot_id}  |  Epoch {epoch}', fontsize=13, fontweight='bold')
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