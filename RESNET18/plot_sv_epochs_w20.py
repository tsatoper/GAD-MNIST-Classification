import json, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── CONFIG ────────────────────────────────────────────────────────────────────
width = 52
metrics_path = f'/glade/derecho/scratch/tsatoperry/GAD/RESNET18/models/main/metrics/w{width}_job5737936.json'
save_path    = f'sv_over_epochs_w{width}.png'
# ─────────────────────────────────────────────────────────────────────────────

with open(metrics_path) as f:
    data = json.load(f)

# Collect all epoch checkpoints present in the JSON
epoch_re = re.compile(r'^epoch(\d+)_sv_path$')
epochs = sorted(int(m.group(1)) for k in data for m in [epoch_re.match(k)] if m)

results = {
    'epoch': [],
    'train_sv_mean': [],
    'train_loss': [], 'test_loss': [],
}

for e in epochs:
    results['epoch'].append(e)
    results['train_sv_mean'].append(data[f'epoch{e}_train_sv_mean'])
    results['train_loss'].append(data[f'epoch{e}_train_loss'])
    results['test_loss'].append(data[f'epoch{e}_test_loss'])

epochs_arr    = np.array(results['epoch'])
train_sv_mean = np.array(results['train_sv_mean'])
train_loss    = np.array(results['train_loss'])
test_loss     = np.array(results['test_loss'])

TEST_COLOR  = 'red'
TRAIN_COLOR = 'blue'

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

# ── Left axis: losses ──
ax1.plot(epochs_arr, train_loss, linestyle='-', linewidth=2,   alpha=0.8, color=TRAIN_COLOR, label='Train Loss')
ax1.plot(epochs_arr, test_loss,  linestyle='-', linewidth=2,   alpha=0.8, color=TEST_COLOR,  label='Test Loss')
ax1.scatter(epochs_arr, train_loss, s=40, alpha=0.8, zorder=5, color=TRAIN_COLOR)
ax1.scatter(epochs_arr, test_loss,  s=40, alpha=0.8, zorder=5, color=TEST_COLOR)

# ── Right axis: train mean SV ──
ax2.plot(epochs_arr, train_sv_mean, linestyle='--', linewidth=2.5, alpha=0.7, color='orange', label='Train Mean SV')
ax2.scatter(epochs_arr, train_sv_mean, s=60, alpha=0.7, marker='s', zorder=5, color='orange')

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_xscale('log')
ax1.set_ylabel('Loss', fontsize=11, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
ax1.tick_params(axis='x', labelsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Mean Singular Value', fontsize=11, color='purple')
ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)

legend_elements = [
    Line2D([0], [0], color=TRAIN_COLOR, linewidth=2,   linestyle='-',  label='Train Loss'),
    Line2D([0], [0], color=TEST_COLOR,  linewidth=2,   linestyle='-',  label='Test Loss'),
    Line2D([0], [0], color='orange', linewidth=2.5, linestyle='--', label='Train Mean SV'),
]
ax1.legend(handles=legend_elements, fontsize=8, loc='lower left')

plt.title(f'ResNet-18  |  Width {width}  |  Loss & Train Mean SV over Epochs',
          fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to '{save_path}'")
plt.show()
