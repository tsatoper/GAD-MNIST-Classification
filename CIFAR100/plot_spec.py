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

TARGET_WIDTHS = [i for i in range(21)]
epochs        = [1, 50, 100, 200]
xscale        = 'log'
yscale        = 'linear'

PLOT_IDS = [
    # 'n_10000',
    # 'n_500',
    # 'n_5000',
    'n_50000'
]

BASE_ROOT  = '/glade/derecho/scratch/tsatoperry/GAD/CIFAR100/models'
MODEL_DIR  = 'n_10000'   # e.g. 'n_10000' — controls filename suffix logic (see load_act_sv)
DEPTH      = 28          # WRN depth, e.g. 28

# Filename patterns (mirrored from CIFAR SV script):
#   train: wrn{depth}_{width}_job{job}_e{epoch}.pt
#   test  (n_10000):      wrn{depth}_{width}_job{job}test_e{epoch}.pt
#   test  (other):        wrn{depth}_{width}_job{job}_e200_test_e{epoch}.pt
#   metrics:  w{width}_job{job}.json   (same as MNIST — adjust if needed)

JOB_PATTERN     = re.compile(r"wrn\d+_(\d+)_job(\d+)_e(\d+)\.pt$")
METRICS_PATTERN = re.compile(r"wrn28_(\d+)_job(\d+)\.json")

save_path = f'loss_vs_meansv_cifar_widths_{"_".join(str(w) for w in TARGET_WIDTHS)}.png'

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
    """Return dict: (job_id, epoch) -> mean SV (train split).

    Train file pattern : wrn{DEPTH}_{width}_job{job}_e{epoch}.pt
    The mean SV is taken over the train-split tensor (key 'train' if dict,
    otherwise the raw tensor), matching what the CIFAR SV script does.
    """
    act_sv = {}
    for epoch in epochs:
        # Train SV files
        train_pattern = os.path.join(
            sv_dir, f'wrn{DEPTH}_{target_width}_job*_e{epoch}.pt'
        )
        for fpath in sorted(glob.glob(train_pattern)):
            # Exclude test files that accidentally match (they contain 'test')
            if 'test' in os.path.basename(fpath):
                continue
            m = re.search(rf'wrn{DEPTH}_{target_width}_job(\d+)_e{epoch}\.pt$',
                          os.path.basename(fpath))
            if not m:
                continue
            job_id = m.group(1)
            try:
                sv_data = torch.load(fpath, map_location='cpu', weights_only=True)
                # Handle both raw tensor and dict-with-'train'-key formats
                if isinstance(sv_data, dict):
                    sv = sv_data['train'].float()
                else:
                    sv = sv_data.float()
                sv[sv < 1e-8] = 1e-8   # floor tiny values (mirrors CIFAR script)
                act_sv[(job_id, epoch)] = sv.mean().item()
                print(f"  job={job_id} epoch={epoch}  mean_sv={act_sv[(job_id, epoch)]:.4f}")
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return act_sv


# ── Collect all (width, plot_id, job_id) panels ──────────────────────────────

norm = mcolors.Normalize(vmin=min(epochs), vmax=max(epochs))
cmap = cm.viridis

panels = []
for target_width in TARGET_WIDTHS:
    for plot_id in PLOT_IDS:
        base_dir    = os.path.join(BASE_ROOT, plot_id)
        metrics_dir = os.path.join(base_dir, f'depth{DEPTH}', 'metrics')
        sv_dir      = os.path.join(base_dir, f'depth{DEPTH}', 'singular_values')

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

ncols = 2
nrows = max((len(panels) + ncols - 1) // ncols, 1)

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

def add_line(ax, xs, ys, colors, **kwargs):
    """Draw a multi-segment line where each segment is coloured by its epoch."""
    segs = [[(xs[i], ys[i]), (xs[i+1], ys[i+1])] for i in range(len(xs) - 1)]
    lc = LineCollection(segs, colors=colors[:-1], **kwargs)
    ax.add_collection(lc)

for panel_idx, (target_width, plot_id, job_id, epoch_losses, act_sv) in enumerate(panels):
    ax = axes[panel_idx // ncols][panel_idx % ncols]

    # Collect (sv, train_loss, test_loss, epoch) tuples, then draw lines sorted by sv
    points = []
    for epoch in epochs:
        if epoch not in epoch_losses:
            continue
        key = (job_id, epoch)
        if key not in act_sv:
            print(f"  Warning: no SV for job={job_id} epoch={epoch}, skipping")
            continue
        points.append((act_sv[key], epoch_losses[epoch]['train_loss'],
                       epoch_losses[epoch]['test_loss'], epoch))

    if points:
        points.sort(key=lambda p: p[3])   # sort by mean SV (x-axis)
        svs         = [p[0] for p in points]
        train_losses = [p[1] for p in points]
        test_losses  = [p[2] for p in points]
        ep_colors   = [cmap(norm(p[3])) for p in points]

        # Draw segments individually so each can be coloured by its epoch
        from matplotlib.collections import LineCollection
        def make_lc(xs, ys, colors, **kwargs):
            segs = [[(xs[i], ys[i]), (xs[i+1], ys[i+1])] for i in range(len(xs)-1)]
            seg_colors = colors[:-1]           # one colour per segment
            lc = LineCollection(segs, colors=seg_colors, **kwargs)
            ax.add_collection(lc)

        make_lc(svs, train_losses, ep_colors, linewidth=1.8, alpha=0.6, linestyle='--', zorder=4)
        make_lc(svs, test_losses,  ep_colors, linewidth=1.8, alpha=0.9, linestyle='-',  zorder=4)

        # Dot markers at each epoch point
        ax.scatter(svs, train_losses, s=30, c=ep_colors, zorder=5, marker='^',
                   alpha=0.6, label='train')
        ax.scatter(svs, test_losses,  s=30, c=ep_colors, zorder=5, marker='o',
                   alpha=0.9, label='test')
        ax.autoscale_view()

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
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.01, pad=0.02)
cbar.set_label('Epoch')

width_str = ', '.join(str(w) for w in TARGET_WIDTHS)
fig.suptitle(f'Loss vs Mean SV — CIFAR100 WRN-{DEPTH} — Widths {width_str}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'\nSaved to "{save_path}"')
plt.show()