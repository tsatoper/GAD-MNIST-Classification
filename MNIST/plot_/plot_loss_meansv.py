import os
import re
import sys
import glob
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/glade/derecho/scratch/tsatoperry/GAD/MNIST")
from utilities import FCNN

plot_id    = 'recreate_mse'
dir_name   = f'./models/{plot_id}/metrics'
act_dir    = f'/glade/derecho/scratch/tsatoperry/GAD/MNIST/models/{plot_id}/activations'

epochs       = [50, 2000]
MAX_SV_WIDTH = 1e10
yscale       = 'log'
xscale       = 'log'
save_path    = f'loss_sv_{plot_id}_{yscale}.png'

ACT_PATTERN = re.compile(r"w(\d+)_job(\d+)_e(\d+)\.pt")

colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
]

# ── Load loss metrics from JSON ───────────────────────────────────────────────

epoch_data = {epoch: {'width': [], 'test_losses': [], 'train_losses': []}
              for epoch in epochs}

if not os.path.exists(dir_name):
    print(f"Warning: Directory '{dir_name}' not found, skipping...")
else:
    for filename in os.listdir(dir_name):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(dir_name, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            width = data['width']
            if width > MAX_SV_WIDTH:
                continue
            for epoch in epochs:
                train_key = f'epoch{epoch}_train_loss'
                test_key  = f'epoch{epoch}_test_loss'
                if train_key in data and test_key in data:
                    epoch_data[epoch]['width'].append(width)
                    epoch_data[epoch]['train_losses'].append(data[train_key])
                    epoch_data[epoch]['test_losses'].append(data[test_key])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {filename}: {e}")

    for epoch in epochs:
        if len(epoch_data[epoch]['width']) > 0:
            idx = np.argsort(epoch_data[epoch]['width'])
            epoch_data[epoch]['width']        = np.array(epoch_data[epoch]['width'])[idx]
            epoch_data[epoch]['test_losses']  = np.array(epoch_data[epoch]['test_losses'])[idx]
            epoch_data[epoch]['train_losses'] = np.array(epoch_data[epoch]['train_losses'])[idx]

# ── Load activations, compute mean SV ────────────────────────────────────────

sv_train_buf = {}
sv_test_buf  = {}

for fpath in sorted(glob.glob(os.path.join(act_dir, '*.pt'))):
    m = ACT_PATTERN.search(os.path.basename(fpath))
    if not m:
        continue
    width, job_id, epoch = int(m.group(1)), m.group(2), int(m.group(3))
    if epoch not in epochs or width > MAX_SV_WIDTH:
        continue

    try:
        acts = torch.load(fpath, map_location='cpu', weights_only=True)
        Phi  = acts['train'].float()

        s = torch.linalg.svdvals(Phi)
        print(s[-5:])
        s = s[s>1e-5]
        sv_tr = s.min().item()
        sv_te = torch.linalg.svdvals(acts['test'].float()).mean().item()

        key = (width, epoch)
        sv_train_buf.setdefault(key, []).append(sv_tr)
        sv_test_buf.setdefault(key,  []).append(sv_te)

        print(f"  width={width} job={job_id} epoch={epoch}  "
              f"mean_sv_train={sv_tr:.4f}  mean_sv_test={sv_te:.4f}")

    except Exception as e:
        print(f"Error processing {fpath}: {e}")

# ── Average across jobs, sort by width ───────────────────────────────────────

sv_data = {epoch: {'width': [], 'mean_sv_train': [], 'mean_sv_test': []}
           for epoch in epochs}

for epoch in epochs:
    keys = sorted({(w, e) for (w, e) in sv_train_buf if e == epoch}, key=lambda x: x[0])
    for (w, e) in keys:
        key = (w, e)
        sv_data[epoch]['width'].append(w)
        sv_data[epoch]['mean_sv_train'].append(np.mean(sv_train_buf[key]))
        sv_data[epoch]['mean_sv_test'].append(np.mean(sv_test_buf[key]))
    for k in sv_data[epoch]:
        sv_data[epoch][k] = np.array(sv_data[epoch][k])

# ── Plot: single figure, loss (left y) + mean SV (right y) ───────────────────

fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()

for idx, epoch in enumerate(epochs):
    color = colors[idx % len(colors)]

    if len(epoch_data[epoch]['width']) > 0:
        w, tl = epoch_data[epoch]['width'], epoch_data[epoch]['test_losses']
        ax1.plot(w, tl, '-', lw=2, alpha=0.85, color=color,
                 label=f'Epoch {epoch} - Test Loss')
        ax1.scatter(w, tl, s=40, zorder=5, color=color, alpha=0.85)

    if len(sv_data[epoch]['width']) > 0:
        w = sv_data[epoch]['width']
        ax2.plot(w, sv_data[epoch]['mean_sv_train'], '--', lw=2, alpha=0.7,
                 color=color, label=f'Epoch {epoch} - Mean SV (train)')
        ax2.scatter(w, sv_data[epoch]['mean_sv_train'], s=30, zorder=4,
                    color=color, alpha=0.7, marker='s')

ax1.set_xlabel('Model Width', fontsize=13)
ax1.set_ylabel('Loss', fontsize=13)
ax2.set_ylabel('Mean Singular Value', fontsize=13, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax1.set_xscale(xscale)
ax1.set_yscale(yscale)
ax2.set_yscale('log')
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best', ncol=2)

plt.title(f'Test Loss & Mean SV by Width ({plot_id})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')