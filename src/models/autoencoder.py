"""Autoencoder anomaly detector: learn to rebuild benign flows; high error = unusual.

Inputs are signed-log transformed (flow features span ~10 orders of magnitude),
then standardized on training benign only. Architecture 71 -> 32 -> 8 -> 32 -> 71.
Early stopping on a 10% holdout of the benign training flows. CPU, fixed seed.
"""

import numpy as np
import torch
from torch import nn

MAX_EPOCHS = 30
PATIENCE = 3
BATCH = 1024


def _signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


class Autoencoder:
    def __init__(self, seed=42, threads=4):
        self.seed, self.threads = seed, threads

    def fit(self, X_benign):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.RandomState(self.seed)
        Z = _signed_log(np.asarray(X_benign, dtype=np.float64))
        self.mean, self.std = Z.mean(axis=0), Z.std(axis=0) + 1e-6
        Z = ((Z - self.mean) / self.std).astype(np.float32)
        order = rng.permutation(len(Z))
        cut = len(Z) // 10
        val, train = torch.from_numpy(Z[order[:cut]]), torch.from_numpy(Z[order[cut:]])

        d = Z.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                                 nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best, best_state, waited = np.inf, None, 0
        self.epochs_run = 0
        for _ in range(MAX_EPOCHS):
            self.net.train()
            for i in torch.randperm(len(train)).split(BATCH):
                opt.zero_grad()
                loss = ((self.net(train[i]) - train[i]) ** 2).mean()
                loss.backward()
                opt.step()
            self.epochs_run += 1
            val_loss = float(self._errors(val).mean())
            if val_loss < best - 1e-4:
                best, waited = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_loss = best
        return self

    @torch.no_grad()
    def _errors(self, Z):
        self.net.eval()
        return torch.cat([((self.net(b) - b) ** 2).mean(dim=1) for b in Z.split(65536)]).numpy()

    def score(self, X):
        """Mean squared rebuild error per flow; higher = less like benign."""
        Z = ((_signed_log(np.asarray(X, dtype=np.float64)) - self.mean) / self.std).astype(np.float32)
        return self._errors(torch.from_numpy(Z))
