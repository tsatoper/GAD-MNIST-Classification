import re
import glob
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def get_epoch(path):
    match = re.search(r"_e(\d+)\.pt$", path)
    return int(match.group(1)) if match else -1


def load_metrics(w, job):
    path = f"./metrics/w{w}_job{job}.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {path}")
        return {}, {}, {}, {}

    train_acc, test_acc, train_loss, test_loss = {}, {}, {}, {}
    for key, val in data.items():
        m = re.match(r"epoch(\d+)_(train_acc|test_acc|train_loss|test_loss)$", key)
        if m:
            epoch = int(m.group(1))
            metric = m.group(2)
            if metric == "train_acc":    train_acc[epoch]  = val
            elif metric == "test_acc":   test_acc[epoch]   = val
            elif metric == "train_loss": train_loss[epoch] = val
            elif metric == "test_loss":  test_loss[epoch]  = val

    return train_acc, test_acc, train_loss, test_loss


def load_singular_values(w, job):
    pattern = f"./singular_values/w{w}_job{job}_e*.pt"
    files = sorted(glob.glob(pattern), key=get_epoch)
    print(f"Found {len(files)} singular value files.")

    epochs, svs = [], []
    for path in files:
        epoch = get_epoch(path)
        sv = torch.load(path, map_location="cpu", weights_only=True)['train'].float()
        epochs.append(epoch)
        svs.append(sv)
    
    return epochs, svs


def main(w=128, job=5192135):
    # --- Load metrics ---
    train_acc, test_acc, train_loss, test_loss = load_metrics(w, job)
    metric_epochs = sorted(set(train_acc) | set(test_acc))
    print(f"Loaded metrics for {len(metric_epochs)} epochs from JSON.")

    # --- Load singular values ---
    sv_epochs, svs = load_singular_values(w, job)

    if not sv_epochs and not metric_epochs:
        print("Nothing to plot.")
        return

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.13))

    all_lines = []

    # Accuracy (left axis, blue)
    if train_acc:
        ep = sorted(train_acc); vals = [train_acc[e] for e in ep]
        l1, = ax1.plot(ep, vals, linewidth=1.5, color="#4C72B0", label="Train Accuracy")
        all_lines.append(l1)
    if test_acc:
        ep = sorted(test_acc); vals = [test_acc[e] for e in ep]
        l2, = ax1.plot(ep, vals, linewidth=1.5, color="#4C72B0", linestyle="--", label="Test Accuracy")
        all_lines.append(l2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    # Mean/min/max singular values (right axis, orange)
    if sv_epochs:
        ep = np.array(sv_epochs)
        l3, = ax2.plot(ep, np.array(svs).mean(axis=1), linewidth=1.5, color="#DD8452", label="Mean SV")
        l3, = ax2.plot(ep, np.array(svs).max(axis=1), linewidth=1.5, color="#DD8452", linestyle="dashdot", label="max SV")
        l4, = ax2.plot(ep, np.array(svs).min(axis=1),  linewidth=1.5, color="#DD8452", linestyle="--", label="min SV")
        all_lines += [l3, l4]
    ax2.set_ylabel("Singular Value", fontsize=12, color="#DD8452")
    ax2.set_yscale('log')
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    # # Loss (far-right axis, green)
    # if train_loss:
    #     ep = sorted(train_loss); vals = [train_loss[e] for e in ep]
    #     l6, = ax3.plot(ep, vals, linewidth=1.5, color="#55A868", label="Train Loss")
    #     all_lines.append(l6)
    # if test_loss:
    #     ep = sorted(test_loss); vals = [test_loss[e] for e in ep]
    #     l7, = ax3.plot(ep, vals, linewidth=1.5, color="#55A868", linestyle="--", label="Test Loss")
    #     all_lines.append(l7)
    # ax3.set_ylabel("Loss", fontsize=12, color="#55A868")
    # ax3.tick_params(axis="y", labelcolor="#55A868")

    ax1.legend(all_lines, [l.get_label() for l in all_lines], fontsize=10, loc="center right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.title(f"Training Curves — w={w}, job={job}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = f"curves_w{w}_job{job}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out}")
    plt.show()

if __name__ == "__main__":
    main(w=256, job=5192135)