import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time


class RandomReLUFeatures(nn.Module):
    """φ(x) = ReLU(W^T x + b),  He-initialized."""

    def __init__(self, input_dim, n_features, device='cpu'):
        super().__init__()
        self.W = torch.randn(input_dim, n_features, device=device) * np.sqrt(2.0 / input_dim)
        self.b = torch.randn(n_features, device=device) * 0.1

    def forward(self, X):
        return F.relu(X @ self.W + self.b)


class RecursiveFeatureMachineReLU:
    """
    RFM with Random ReLU Features — Algorithm 1.

    We store only M_sqrt (the Cholesky factor of M), never M itself.
    M_sqrt is all that's ever used: features and gradients both go through
    X @ M_sqrt^T, so recomputing M from M_sqrt would be wasteful.

    At the end of each iteration we build M_new = (1/n) Σ gg^T directly,
    then immediately factor it into the new M_sqrt — M is never stored.

    Regime selection for the linear solve:
      D >= n  →  DUAL:   (ΦΦᵀ + λ‖ΦΦᵀ‖I) α = y,  W = Φᵀα   (n×n)
      D <  n  →  PRIMAL: (ΦᵀΦ + λ‖ΦᵀΦ‖I) W = Φᵀy             (D×D)
    """

    def __init__(
        self,
        input_dim=784,
        n_features=1000,
        n_iterations=5,
        ridge=1e-2,
        n_grad_samples=500,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_dim = input_dim
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.ridge = ridge
        self.n_grad_samples = n_grad_samples
        self.device = device

        self.M_sqrt = None   # (d, d) — only matrix we store
        self.W_coef = None   # (D, C)
        self.relu_features = None

        self.sv_history = []
        self.train_loss_history = []
        self.test_loss_history = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _psd_sqrt(M):
        """Compute M^{1/2} via Cholesky; fall back to eigen if not PSD."""
        try:
            return torch.linalg.cholesky(M)
        except Exception:
            vals, vecs = torch.linalg.eigh(M)
            vals = torch.clamp(vals, min=1e-8)
            return vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T

    def _compute_features(self, X):
        """φ_M(x) = ReLU(W_relu^T  M_sqrt x + b)"""
        return self.relu_features(X @ self.M_sqrt.T)   # (n, D)

    def _solve(self, Phi, y):
        n, D = Phi.shape
        if D >= n:
            K     = Phi @ Phi.T
            scale = torch.linalg.matrix_norm(K, ord=2)
            reg   = K + self.ridge * scale * torch.eye(n, device=self.device)
            alpha = torch.linalg.solve(reg, y)          # (n, C)
            return Phi.T @ alpha                         # (D, C)
        else:
            A     = Phi.T @ Phi
            scale = torch.linalg.matrix_norm(A, ord=2)
            reg   = A + self.ridge * scale * torch.eye(D, device=self.device)
            return torch.linalg.solve(reg, Phi.T @ y)   # (D, C)

    def _compute_input_gradient(self, x, W_coef):
        """∇_x [φ_M(x) @ W_coef] summed over output classes."""
        x_t    = x @ self.M_sqrt.T
        active = (x_t @ self.relu_features.W + self.relu_features.b > 0).float()  # (D,)
        W_sum  = W_coef.sum(dim=1)                                                  # (D,)
        return self.M_sqrt.T @ (self.relu_features.W @ (active * W_sum))           # (d,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, X_test=None, y_test=None, verbose=True):
        X = X.to(self.device)
        y = y.to(self.device)
        if X_test is not None:
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)

        n, d = X.shape

        # M_0 = I  →  M_sqrt_0 = I
        self.M_sqrt = torch.eye(d, device=self.device)
        self.relu_features = RandomReLUFeatures(d, self.n_features, self.device)

        for t in range(self.n_iterations):
            t0 = time.time()

            # M_sqrt already set — use it directly
            Phi = self._compute_features(X)              # (n, D)

            sv = torch.linalg.svdvals(Phi)
            self.sv_history.append(sv.mean().item())

            self.W_coef = self._solve(Phi, y)            # (D, C)

            with torch.no_grad():
                train_loss = F.mse_loss(Phi @ self.W_coef, y).item()
                self.train_loss_history.append(train_loss)

                test_loss = None
                if X_test is not None:
                    Phi_te = self._compute_features(X_test)
                    test_loss = F.mse_loss(Phi_te @ self.W_coef, y_test).item()
                    self.test_loss_history.append(test_loss)

            # Build M_new = (1/n) Σ gg^T, normalize, then factor → new M_sqrt
            M_new = torch.zeros(d, d, device=self.device)
            idx = torch.randperm(n, device=self.device)[:self.n_grad_samples]
            for i in idx:
                g = self._compute_input_gradient(X[i], self.W_coef)
                M_new += torch.outer(g, g)
            M_new /= len(idx)

            tr = M_new.trace()
            if tr > 1e-10:
                M_new *= d / tr                          # keep scale ~ identity
            M_new += 1e-8 * torch.eye(d, device=self.device)

            # Factor once here; never store M again
            self.M_sqrt = self._psd_sqrt(M_new)

            if verbose:
                mode = "dual" if self.n_features >= n else "primal"
                msg = (f"Iter {t+1:>2}/{self.n_iterations} [{mode}] | "
                       f"Train MSE: {train_loss:.6f} | "
                       f"Mean SV: {sv.mean().item():.4f}")
                if test_loss is not None:
                    msg += f" | Test MSE: {test_loss:.6f}"
                msg += f" | {time.time()-t0:.1f}s"
                print(msg)

    def predict(self, X):
        X = X.to(self.device)
        return self._compute_features(X) @ self.W_coef


# ----------------------------------------------------------------------
# MNIST helpers
# ----------------------------------------------------------------------

def load_mnist_subset(n_train, n_test):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    te = datasets.MNIST('./data', train=False, download=True, transform=tf)

    def collect(ds, n):
        X, y = zip(*[(img.view(-1), lbl) for img, lbl in Subset(ds, range(n))])
        return torch.stack(X), torch.tensor(y)

    return collect(tr, n_train), collect(te, n_test)


def run(n_train=1000, n_test=1000, n_features=1000, n_iterations=5, ridge=1e-2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}  n_train={n_train}  D={n_features}  λ={ridge}")

    (X_tr, y_tr), (X_te, y_te) = load_mnist_subset(n_train, n_test)
    y_tr_oh = F.one_hot(y_tr, 10).float()
    y_te_oh = F.one_hot(y_te, 10).float()

    rfm = RecursiveFeatureMachineReLU(
        input_dim=784, n_features=n_features,
        n_iterations=n_iterations, ridge=ridge, device=device
    )
    rfm.fit(X_tr, y_tr_oh, X_test=X_te, y_test=y_te_oh)

    with torch.no_grad():
        tr_acc = (rfm.predict(X_tr).argmax(1).cpu() == y_tr).float().mean().item()
        te_acc = (rfm.predict(X_te).argmax(1).cpu() == y_te).float().mean().item()

    print(f"\nTrain Acc: {tr_acc*100:.2f}%   Test Acc: {te_acc*100:.2f}%")
    print(f"\n{'Iter':<6} {'Train MSE':<14} {'Test MSE':<14} {'Mean SV':<10}")
    print("-" * 50)
    for i in range(len(rfm.train_loss_history)):
        tl = rfm.train_loss_history[i]
        sl = rfm.test_loss_history[i] if i < len(rfm.test_loss_history) else float('nan')
        print(f"{i+1:<6} {tl:<14.6f} {sl:<14.6f} {rfm.sv_history[i]:<10.4f}")

    return rfm


if __name__ == "__main__":
    print("=" * 55)
    print("n = D = 1000")
    print("=" * 55)
    run(n_train=1000, n_test=1000, n_features=1000, n_iterations=5)

    print("\n" + "=" * 55)
    print("D >> n  (5000 features, 1000 train)")
    print("=" * 55)
    run(n_train=1000, n_test=1000, n_features=5000, n_iterations=5)

    print("\n" + "=" * 55)
    print("D << n  (500 features, 5000 train)")
    print("=" * 55)
    run(n_train=5000, n_test=1000, n_features=500, n_iterations=5)