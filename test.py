import argparse
import torch
import torch.nn as nn
from utilities import FCNN, test
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------
# Parse job index argument
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-idx",
    type=int,
    required=True,
    help="Index into hidden_dim_list to choose hidden dimension"
)
args = parser.parse_args()

# Construct hidden dimension list
hidden_dim_list = (
    [i for i in range(1, 29 + 1)] +               # 1–29
    [10 * i for i in range(30 // 10, 70 // 10)] + # 3*10 to 6*10 → 30,40,50,60
    [2**i for i in range(6, 25)]                  # 2^6 to 2^24
)

hidden_dim = int(hidden_dim_list[args.job_idx])
print(f"Selected hidden_dim = {hidden_dim} (from job_idx {args.job_idx})")

# --------------------------
# Load model
# --------------------------
weights_path = (
    f"/glade/derecho/scratch/tsatoperry/GAD/models/mse/weights/"
    f"mnist_hidden_dim{hidden_dim}_epochs2000.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNN(input_dim=784, hidden_dim=hidden_dim, output_dim=10)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

print(f"Model loaded from {weights_path}")
print(f"Model architecture: input_dim=784, hidden_dim={hidden_dim}, output_dim=10")
print(f"Device: {device}")

# --------------------------
# Load MNIST test set
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset_full = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset_full, batch_size=256, shuffle=False)

loss_fn = nn.MSELoss()

# --------------------------
# Collect hidden activations
# --------------------------
all_hidden = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        _, hidden = model(data, return_hidden=True)
        all_hidden.append(hidden)

all_hidden = torch.cat(all_hidden, dim=0)

# --------------------------
# SVD of hidden activations
# --------------------------
U, S, Vh = torch.linalg.svd(all_hidden, full_matrices=False)
print("Singular values:")
print(S)

sv_path = f"/glade/derecho/scratch/tsatoperry/GAD/singular_values/mnist_hidden_dim{hidden_dim}_sv.pt"
torch.save(S.cpu(), sv_path)

print(f"Singular values saved to {sv_path}")