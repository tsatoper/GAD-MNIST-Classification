import torch
import math

import sys
sys.path.append('/glade/derecho/scratch/tsatoperry/GAD/MNIST')
from utilities import mnist_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

class RRF(torch.nn.Module):
    def __init__(self, input_dim, d_features, device):
        super().__init__()
        self.W = torch.randn(input_dim, d_features, device=device)
        self.b = torch.rand(d_features, device=device) * 2 * math.pi

    def forward(self, x):
        return torch.relu(x @ self.W + self.b)

n_samples = 100
train_loader = mnist_loader(train=True, n_samples=n_samples, seed=0)
test_loader  = mnist_loader(train=False)
y_train_onehot = torch.cat([
    torch.zeros(t.size(0), 10, device=device).scatter_(1, t.unsqueeze(1).to(device), 1)
    for _, t in train_loader], dim=0)  # (n, 10)

y_test_onehot = torch.cat([
    torch.zeros(t.size(0), 10, device=device).scatter_(1, t.unsqueeze(1).to(device), 1)
    for _, t in test_loader], dim=0)  # (p, 10)

X_train = torch.cat([x.view(x.size(0), -1).to(device) for x, _ in train_loader], dim=0)  # (n, 784)
X_test  = torch.cat([x.view(x.size(0), -1).to(device) for x, _ in test_loader],  dim=0)  # (p, 784)

d_features = 100
for d_features in [90, 100, 110, 1000]:
    rrf = RRF(input_dim=784, d_features=d_features, device=device)
    print(X_train[:5])
    M_tm = rrf(X_train) / math.sqrt(d_features)  # (n, d)
    print(M_tm)
    M_tm_dag = torch.linalg.pinv(M_tm)
    theta_m = M_tm_dag @ y_train_onehot # solve y = M * theta

    y_hat_train = M_tm @ theta_m
    train_mse = torch.mean(torch.abs(y_hat_train - y_train_onehot)**2)

    M_pm = rrf(X_test) / math.sqrt(d_features)
    y_hat_test  = M_pm @ theta_m #y = M * theta
    test_mse = torch.mean(torch.abs(y_hat_test - y_test_onehot)**2)

    print(f"d = {d_features}, ||M_tm+|| = {M_tm_dag.norm(p=2):.3f}, Train MSE = {train_mse.item():.3f}, Test MSE = {test_mse.item():.3f}")