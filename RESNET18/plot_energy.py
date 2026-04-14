"""
Plot modal energy over training checkpoints for ResNet18 on CIFAR-10.

Usage:
    p3 plot_energy.py --model_dir ./models/main --width 20
"""

import sys
import os
import argparse
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ResNet import make_resnet18k

BASE_DIR = "/glade/derecho/scratch/tsatoperry/GAD/RESNET18"


def find_run(model_dir: str, width: int) -> str:
    metrics_dir = os.path.join(model_dir, "metrics")
    prefix = f"w{width}_"
    matches = [f for f in os.listdir(metrics_dir)
               if f.startswith(prefix) and f.endswith(".json")]
    if not matches:
        raise FileNotFoundError(
            f"No metrics file matching '{prefix}*.json' in {metrics_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for width {width} in {metrics_dir}: {matches}")
    return matches[0].replace(".json", "")


def load_metrics(model_dir: str, run: str) -> dict:
    path = os.path.join(model_dir, "metrics", f"{run}.json")
    with open(path) as f:
        return json.load(f)


def optimal_epoch(metrics: dict) -> int:
    best_epoch, best_loss = None, float("inf")
    for k, v in metrics.items():
        if k.endswith("_test_loss") and isinstance(v, float):
            epoch = int(k.replace("epoch", "").replace("_test_loss", ""))
            if v < best_loss:
                best_loss, best_epoch = v, epoch
    return best_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width",     type=int, required=True)
    p.add_argument("--model_dir", type=str,
                   default=os.path.join(BASE_DIR, "models", "main"))
    p.add_argument("--output",    type=str, default=None)
    return p.parse_args()


def load_activations(model_dir: str, run: str, epoch: int) -> torch.Tensor:
    """Load pre-saved hidden activations (train split) from .pt file."""
    path = os.path.join(model_dir, "activations", f"{run}_e{epoch}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Activations not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["train"]  # (N, D)


def load_model_weights(model_dir: str, run: str, epoch: int,
                       width: int) -> torch.Tensor:
    """Load model checkpoint and return the final linear weight matrix W."""
    path = os.path.join(model_dir, "weights", f"{run}_e{epoch}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = make_resnet18k(k=width, num_classes=10)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.linear.weight.detach()  # (10, D)


def spectral_analysis(H: torch.Tensor, Y_hat: torch.Tensor, W: torch.Tensor,
                      eps: float = 1e-8):
    """
    Returns (evals, pred_energy, amplification, w_proj, svdvals), all shape (D,).

    W : (10, D)  final-layer weight matrix (model.linear.weight)
    w_proj[k] = ||W q_k||  — how much the readout is aligned with mode k
    """
    N, D = H.shape
    Sigma = (H.T @ H) / N
    evals, Q = torch.linalg.eigh(Sigma.double())
    evals = evals.float()        # (D,)  ascending
    Q     = Q.float()            # (D, D)

    H_proj     = H @ Q                           # (N, D)
    Y_hat_proj = (H_proj.T @ Y_hat) / N          # (D, 10)
    pred_energy = torch.norm(Y_hat_proj, dim=1)  # (D,)

    amplification = pred_energy / (evals.abs() + eps)   # (D,)

    # W: (10, D),  Q: (D, D)  →  W @ Q : (10, D),  norm over classes → (D,)
    w_proj = torch.norm(W @ Q, dim=0)                   # (D,)

    # Singular values of H: σ_i = sqrt(λ_i * N), where λ_i = evals of H^T H / N
    svdvals = torch.sqrt(torch.clamp(evals * N, min=0))  # (D,)

    return evals, pred_energy, amplification, w_proj, svdvals


def make_plots(results: dict, epoch_labels: dict, output_path: str):
    """results : {epoch: (evals, pred_energy, amplification, w_proj, svdvals)}
       epoch_labels : {epoch: str} — role label for each epoch"""
    epochs_sorted = sorted(results.keys())
    palette = ["#3b82f6", "#22c55e", "#ef4444"]

    fig, (ax_s, ax_e, ax_a, ax_w) = plt.subplots(4, 1, figsize=(7, 18), sharex=True)
    fig.suptitle("Spectral Diagnostics over Training (ResNet18 / CIFAR-10)",
                 fontsize=14, fontweight="bold")

    for idx, epoch in enumerate(epochs_sorted):
        evals, pred_e, amp, w_proj, svdvals = results[epoch]
        color = palette[idx % len(palette)]
        label = f"epoch {epoch} ({epoch_labels.get(epoch, '')})"

        ev = evals.numpy()
        desc     = np.argsort(ev)[::-1]
        mode_idx = np.arange(1, len(ev) + 1)

        ax_s.plot(mode_idx, svdvals.numpy()[desc], color=color, label=label, linewidth=1.5)
        ax_e.plot(mode_idx, pred_e.numpy()[desc],  color=color, label=label, linewidth=1.5)
        ax_a.plot(mode_idx, amp.numpy()[desc],     color=color, label=label, linewidth=1.5)
        ax_w.plot(mode_idx, w_proj.numpy()[desc],  color=color, label=label, linewidth=1.5)

    for ax in (ax_s, ax_e, ax_a, ax_w):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    ax_s.set_ylabel("Singular Value  σ_k")
    ax_s.set_title("(A) Singular Values of Features H")

    ax_e.set_ylabel("Modal Energy  ||Ŷ_proj||")
    ax_e.set_title("(B) Modal Energy")

    ax_a.set_ylabel("Pred Energy / λ")
    ax_a.set_title("(C) Amplification")

    ax_w.set_xlabel("Mode Index (largest singular value → left)")
    ax_w.set_ylabel("||W* q_k||")
    ax_w.set_title("(D) Readout Alignment  ||W* q_k||")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    run        = find_run(args.model_dir, args.width)
    metrics    = load_metrics(args.model_dir, run)
    width      = metrics["width"]
    n_samples  = metrics["train_samples"]
    opt_epoch  = optimal_epoch(metrics)
    last_epoch = metrics["num_epochs"]

    # Build (epoch, label) pairs, collapsing duplicates.
    epoch_labels = {}
    if opt_epoch == 1:
        epoch_labels[1]          = "optimal (early-stop)"
        epoch_labels[last_epoch] = "overfit"
    elif opt_epoch == last_epoch:
        epoch_labels[1]          = "early"
        epoch_labels[last_epoch] = "optimal (early-stop)"
    else:
        epoch_labels[1]          = "early"
        epoch_labels[opt_epoch]  = "optimal (early-stop)"
        epoch_labels[last_epoch] = "overfit"


    model_dir_name = os.path.basename(args.model_dir)
    out = args.output or os.path.join(
        SCRIPT_DIR, f"energy_w{width}_{model_dir_name}.png")

    print(f"Run      : {run}")
    print(f"Width    : {width}  N samples: {n_samples}")
    print(f"Epochs   : {', '.join(f'{l}={e}' for e, l in sorted(epoch_labels.items()))}")

    results = {}
    for epoch in sorted(epoch_labels):
        print(f"Checkpoint epoch={epoch} ({epoch_labels[epoch]}) …")
        try:
            H = load_activations(args.model_dir, run, epoch)
            W = load_model_weights(args.model_dir, run, epoch, width)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        # Reconstruct predictions from saved activations: Y_hat = H @ W^T
        Y_hat = H @ W.T  # (N, 10)

        evals, pred_e, amp, w_proj, svdvals = spectral_analysis(H, Y_hat, W)
        results[epoch] = (evals, pred_e, amp, w_proj, svdvals)

    if not results:
        print("No checkpoints loaded — exiting.")
        return

    make_plots(results, epoch_labels, out)


if __name__ == "__main__":
    main()
