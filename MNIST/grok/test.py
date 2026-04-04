import re
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_log(filepath):
    train_epochs = []
    train_losses = []
    train_accuracies = []
    train_l2_norms = []
    test_epochs = []
    test_losses = []
    test_accuracies = []

    # Train Epoch 1: Avg Loss: 0.4353, Accuracy: 290/1000 (29.00%), LR = 0.001
    # Total weight L2 norm: 68.3340
    train_pattern = re.compile(
        r"Train Epoch\s+(\d+):\s+Avg Loss:\s+([\d.]+),\s+Accuracy:\s+\d+/\d+\s+\(([\d.]+)%\)"
    )
    l2_pattern = re.compile(r"Total weight L2 norm:\s*([\d.]+)")
    # Test set: Avg Loss: 4.8733, Accuracy: 1252/10000 (12.52%)
    test_pattern = re.compile(
        r"Test set:\s+Avg Loss:\s+([\d.]+),\s+Accuracy:\s+\d+/\d+\s+\(([\d.]+)%\)"
    )

    with open(filepath, "r") as f:
        lines = f.readlines()[500:]

    current_train_epoch = None
    pending_l2 = False  # expecting L2 norm on the next line(s)

    for line in lines:
        # Check for L2 norm line (may follow train line)
        if pending_l2:
            l2_match = l2_pattern.search(line)
            if l2_match:
                train_l2_norms[-1] = float(l2_match.group(1))
                pending_l2 = False
                continue

        train_match = train_pattern.search(line)
        if train_match:
            current_train_epoch = int(train_match.group(1))
            train_epochs.append(current_train_epoch)
            train_losses.append(float(train_match.group(2)))
            train_accuracies.append(float(train_match.group(3)))
            train_l2_norms.append(None)
            pending_l2 = True
            continue
        test_match = test_pattern.search(line)
        if test_match and current_train_epoch is not None:
            test_epochs.append(current_train_epoch)
            test_losses.append(float(test_match.group(1)))
            test_accuracies.append(float(test_match.group(2)))
            current_train_epoch = None
            pending_l2 = False

    return train_epochs, train_losses, train_accuracies, train_l2_norms, \
           test_epochs, test_losses, test_accuracies


def plot(train_epochs, train_losses, train_accs, train_l2_norms,
         test_epochs, test_losses, test_accs,
         output_path="train_plot.png"):

    has_test = len(test_epochs) > 0
    has_l2 = any(v is not None for v in train_l2_norms)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    # --- Accuracy (left axis, blue) ---
    l1, = ax1.plot(train_epochs, train_accs, linewidth=1.5, color="#4C72B0", label="Train Accuracy")
    if has_test:
        l2, = ax1.plot(test_epochs, test_accs, linewidth=1.5, color="#4C72B0", linestyle="--", label="Test Accuracy")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    # --- Loss (right axis, orange) ---
    l3, = ax2.plot(train_epochs, train_losses, linewidth=1.5, color="#DD8452", label="Train Loss")
    if has_test:
        l4, = ax2.plot(test_epochs, test_losses, linewidth=1.5, color="#DD8452", linestyle="--", label="Test Loss")
    ax2.set_ylabel("Avg Loss", fontsize=12, color="#DD8452")
    ax2.set_yscale('log')
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    all_lines = [l1, l3]
    if has_test:
        all_lines += [l2, l4]

    # --- L2 Norm (second right axis offset, green) ---
    if has_l2:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.13))
        l2_epochs = [e for e, v in zip(train_epochs, train_l2_norms) if v is not None]
        l2_vals   = [v for v in train_l2_norms if v is not None]
        l5, = ax3.plot(l2_epochs, l2_vals, linewidth=1.5, color="#55A868", linestyle="-.", label="L2 Norm")
        ax3.set_ylabel("Weight L2 Norm", fontsize=12, color="#55A868")
        ax3.set_yscale('log')
        ax3.tick_params(axis="y", labelcolor="#55A868")
        all_lines.append(l5)

    ax1.legend(all_lines, [l.get_label() for l in all_lines], fontsize=11, loc="center right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.title("Training Curves — Accuracy, Loss & L2 Norm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_train.py <log_file> [output_image.png]")
        sys.exit(1)

    log_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "train_plot.png"

    train_epochs, train_losses, train_accs, train_l2_norms, \
        test_epochs, test_losses, test_accs = parse_log(log_file)

    if not train_epochs:
        print("No training log entries found. Check your log format.")
        sys.exit(1)

    print(f"Found {len(train_epochs)} train epochs, {len(test_epochs)} test entries.")
    plot(train_epochs, train_losses, train_accs, train_l2_norms,
         test_epochs, test_losses, test_accs, out_file)