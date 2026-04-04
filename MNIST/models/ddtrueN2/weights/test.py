import re
import glob
import torch
import matplotlib.pyplot as plt


def get_epoch(path):
    match = re.search(r"_e(\d+)\.pth$", path)
    return int(match.group(1)) if match else -1


def compute_l2(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"Key '{key}' not found in state dict. Available keys: {list(state_dict.keys())}")
    return state_dict[key].float().norm(2).item()


def main(w=30, job=5033679):
    pattern = f"w{w}_job{job}_e*.pth"
    files = sorted(glob.glob(pattern), key=get_epoch)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} checkpoints.")

    epochs, fc1_norms, fc2_norms = [], [], []

    for path in files:
        epoch = get_epoch(path)
        state_dict = torch.load(path, map_location="cpu", weights_only=true)

        # Handle wrapped state dicts (e.g. from DataParallel or {'model': ...})
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # Find fc1 and fc2 weight keys
        fc1_key = next((k for k in state_dict if "fc1" in k and "weight" in k), None)
        fc2_key = next((k for k in state_dict if "fc2" in k and "weight" in k), None)

        if fc1_key is None or fc2_key is None:
            print(f"  [{path}] Could not find fc1/fc2 weight keys. Keys: {list(state_dict.keys())}")
            continue

        epochs.append(epoch)
        fc1_norms.append(compute_l2(state_dict, fc1_key))
        fc2_norms.append(compute_l2(state_dict, fc2_key))
        print(f"  Epoch {epoch:>5}: fc1 L2 = {fc1_norms[-1]:.4f}, fc2 L2 = {fc2_norms[-1]:.4f}")

    if not epochs:
        print("No valid checkpoints parsed.")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, fc1_norms, marker="o", linewidth=1.5, markersize=3, color="#4C72B0", label="fc1 weight L2 norm")
    ax.plot(epochs, fc2_norms, marker="s", linewidth=1.5, markersize=3, color="#DD8452", label="fc2 weight L2 norm")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(f"FC Layer Weight L2 Norms — w={w}, job={job}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = f"l2_norms_w{w}_job{job}.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to: {out}")
    plt.show()


if __name__ == "__main__":
    main(w=30, job=5033679)