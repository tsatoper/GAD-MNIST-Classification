import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── CONFIG ────────────────────────────────────────────────────────────────────
metrics_dir = '/glade/derecho/scratch/tsatoperry/GAD/RESNET18/models/s50k_n15/metrics'
epochs      = [2000]
save_path   = 'loss_sv_vs_width_by_epoch.png'
# ─────────────────────────────────────────────────────────────────────────────

collected = {e: {'width': [], 'sv_train': [], 'train_loss': [], 'test_loss': []}
             for e in epochs}

for filepath in sorted(glob.glob(os.path.join(metrics_dir, '*.json'))):
    with open(filepath) as f:
        data = json.load(f)
    width = data['width']
    for e in epochs:
        if f'epoch{e}_train_sv_mean' not in data:
            continue
        collected[e]['width'].append(width)
        collected[e]['sv_train'].append(data[f'epoch{e}_train_sv_mean'])
        collected[e]['train_loss'].append(data[f'epoch{e}_train_loss'])
        collected[e]['test_loss'].append(data[f'epoch{e}_test_loss'])

for e in epochs:
    idx = np.argsort(collected[e]['width'])
    for k in collected[e]:
        collected[e][k] = np.array(collected[e][k])[idx]

n_samples = data.get('samples', None)
fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()

for e in epochs:
    d = collected[e]
    if len(d['width']) == 0:
        continue
    ax1.plot(d['width'], d['train_loss'], color='blue', linewidth=1.0, alpha=0.4)
    ax1.plot(d['width'], d['test_loss'],  color='red',  linewidth=1.0, alpha=0.4)
    ax2.plot(d['width'], d['sv_train'],   color='blue', linewidth=1.0, alpha=0.4, linestyle='--')

if n_samples is not None:
    ax1.axvline(x=n_samples, color='black', linestyle='--', linewidth=1.5, alpha=0.9)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_xlabel('Model Width (k)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax2.set_ylabel('Train Mean Singular Value', fontsize=12, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax1.grid(True, which='both', alpha=0.3)

legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-',  label='Train Loss'),
    Line2D([0], [0], color='red',  linewidth=2, linestyle='-',  label='Test Loss'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Train Mean SV'),
]
if n_samples is not None:
    legend_elements.append(Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label=f'n_samples={n_samples}'))
ax1.legend(handles=legend_elements, fontsize=10, loc='upper right')


plt.title('ResNet-18k  |  s50k_n15  |  Loss & Mean SV vs Width', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to '{save_path}'")
plt.show()