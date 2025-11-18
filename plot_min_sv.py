import os
import re
import torch
import matplotlib.pyplot as plt

def load_min_singular_values(directory):
    """
    Scans directory for files named hidden_dimX_sv.pt,
    loads the tensor, and extracts the minimum singular value.
    """
    pattern = re.compile(r"hidden_dim(\d+)_sv\.pt$")
    results = {}

    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            hidden_dim = int(match.group(1))
            path = os.path.join(directory, fname)

            try:
                sv_tensor = torch.load(path, map_location='cpu', weights_only=True)
                min_sv = float(sv_tensor.min().item())
                results[hidden_dim] = min_sv
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

    return results


def main():
    directory = "/glade/derecho/scratch/tsatoperry/GAD/models/mse/singular_values"

    results = load_min_singular_values(directory)
    if not results:
        print("No matching files found.")
        return

    # Sort by hidden_dim
    hidden_dims = sorted(results.keys())
    min_svs = [results[h] for h in hidden_dims]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_dims, min_svs, marker='o')
    plt.xlabel("Hidden Dimension (X)")
    plt.ylabel("Minimum Singular Value")
    plt.title("Minimum Singular Value vs Hidden Dimension")
    plt.grid(True)

    out_path = os.path.join(directory, "min_sv_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")

    plt.savefig("min_sv.png")


if __name__ == "__main__":
    main()
