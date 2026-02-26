import os
import re
import glob
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_WIDTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
epochs        = [100 * i for i in range(1, 20)]
xscale        = 'log'
yscale        = 'linear'

PLOT_IDS = [
    'N1_1e-3',
    'N1_1e-4',
    'N2_1e-3',
    'N2_1e-4',
    'N3_1e-3',
    'N3_1e-4'
]

BASE_ROOT = '/glade/derecho/scratch/tsatoperry/GAD/MNIST/models'

JOB_PATTERN     = re.compile(r"w(\d+)_job(\d+)_e(\d+)")
METRICS_PATTERN = re.compile(r"w(\d+)_job(\d+)\.json")

save_path = f'loss_vs_meansv_widths_{"_".join(str(w) for w in TARGET_WIDTHS)}.png'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_job_losses(metrics_dir, target_width):
    """Return dict: job_id -> {epoch -> {train_loss, test_loss}}"""
    job_losses = {}
    for filename in sorted(os.listdir(metrics_dir)):
        m = METRICS_PATTERN.fullmatch(filename)
        if not m:
            continue
        width, job_id = int(m.group(1)), m.group(2)
        if width != target_width:
            continue
        filepath = os.path.join(metrics_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {filename}: {e}")
            continue
        job_losses[job_id] = {}
        for epoch in epochs:
            train_key = f'epoch{epoch}_train_loss'
            test_key  = f'epoch{epoch}_test_loss'
            if train_key in data and test_key in data:
                job_losses[job_id][epoch] = {
                    'train_loss': data[train_key],
                    'test_loss':  data[test_key],
                }
    return job_losses


def load_act_sv(sv_dir, target_width):
    """Return dict: (job_id, epoch) -> mean SV (train split)"""
    act_sv = {}
    for epoch in epochs:
        pattern = os.path.join(sv_dir, f'w{target_width}_job*_e{epoch}.pt')
        for fpath in sorted(glob.glob(pattern)):
            m = JOB_PATTERN.search(os.path.basename(fpath))
            if not m:
                continue
            job_id = m.group(2)
            try:
                sv_dict = torch.load(fpath, map_location='cpu', weights_only=True)
                sv = sv_dict['train'].float()
                act_sv[(job_id, epoch)] = sv.mean().item()
                print(f"  job={job_id} epoch={epoch}  mean_sv={act_sv[(job_id, epoch)]:.4f}")
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return act_sv


# ── Collect all (width, plot_id, job_id) panels ──────────────────────────────

norm = mcolors.Normalize(vmin=min(epochs), vmax=max(epochs))
cmap = cm.viridis

# panels: list of (width, plot_id, job_id, epoch_losses, act_sv)
panels = []
for target_width in TARGET_WIDTHS:
    for plot_id in PLOT_IDS:
        base_dir    = os.path.join(BASE_ROOT, plot_id)
        metrics_dir = os.path.join(base_dir, 'metrics')
        sv_dir      = os.path.join(base_dir, 'singular_values')

        if not os.path.isdir(metrics_dir) or not os.path.isdir(sv_dir):
            print(f"Skipping w{target_width}/{plot_id}: directory not found.")
            continue

        print(f"\n=== Loading w{target_width} / {plot_id} ===")
        job_losses = load_job_losses(metrics_dir, target_width)
        print(f"  Found {len(job_losses)} jobs for width={target_width}")
        act_sv = load_act_sv(sv_dir, target_width)

        for job_id in sorted(job_losses.keys()):
            panels.append((target_width, plot_id, job_id, job_losses[job_id], act_sv))

# ── Compute global x/y limits across all panels ──────────────────────────────

all_svs    = []
all_losses = []

for target_width, plot_id, job_id, epoch_losses, act_sv in panels:
    for epoch in epochs:
        if epoch not in epoch_losses:
            continue
        key = (job_id, epoch)
        if key not in act_sv:
            continue
        all_svs.append(act_sv[key])
        all_losses.append(epoch_losses[epoch]['test_loss'])
        all_losses.append(epoch_losses[epoch]['train_loss'])


# ── Build figure ─────────────────────────────────────────────────────────────

ncols = 3
nrows = max((len(panels) + ncols - 1) // ncols, 1)

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

for panel_idx, (target_width, plot_id, job_id, epoch_losses, act_sv) in enumerate(panels):
    ax = axes[panel_idx // ncols][panel_idx % ncols]

    for epoch in epochs:
        if epoch not in epoch_losses:
            continue
        key = (job_id, epoch)
        if key not in act_sv:
            print(f"  Warning: no SV for job={job_id} epoch={epoch}, skipping")
            continue

        color = cmap(norm(epoch))
        sv    = act_sv[key]
        ax.scatter(sv, epoch_losses[epoch]['test_loss'],  s=50, zorder=5, color=color,
                   alpha=0.9, marker='o', label='test'  if epoch == epochs[0] else '')
        ax.scatter(sv, epoch_losses[epoch]['train_loss'], s=50, zorder=5, color=color,
                   alpha=0.5, marker='^', label='train' if epoch == epochs[0] else '')

    ax.legend(fontsize=9)
    ax.set_title(f'w{target_width} / {plot_id}  job={job_id}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean SV', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(panels), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

# Shared colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

width_str = ', '.join(str(w) for w in TARGET_WIDTHS)
fig.suptitle(f'Loss vs Mean SV — Widths {width_str}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')