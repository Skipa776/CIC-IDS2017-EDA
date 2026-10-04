"""The evaluation contract: every model in every fold is scored by this module.

Rules it enforces (see the design doc):
- Thresholds come from VALIDATION benign scores only (threshold_on_benign).
  The test set is scored after thresholds are frozen; the actual test FPR is
  reported beside recall because a frozen threshold does not guarantee it.
- Average precision is always reported with its no-skill baseline (prevalence).
- ROC AUC is reported too. It ignores class balance, so on rare attacks it reads
  high even for a useless detector; read it beside AP and prevalence. Benign
  weights do not change it (benign is sampled uniformly), so it is unweighted.
- Per-family recall carries its flow count and a Wilson interval. The interval
  assumes independent flows, which attack flows are not, so it is optimistic.
"""

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.evaluate import threshold_on_benign

ROOT = Path(__file__).resolve().parent.parent.parent
BUDGETS = (0.001, 0.01, 0.05)
PRIMARY_BUDGET = 0.01
PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "torch"]


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def frozen_thresholds(val_benign_scores, budgets=BUDGETS):
    """Thresholds per alert budget, from validation benign scores only."""
    return {str(b): threshold_on_benign(val_benign_scores, b) for b in budgets}


def at_threshold(y, families, alert, weight):
    """Detection summary for one frozen threshold. `weight` undoes benign sampling."""
    benign, attack = y == 0, y == 1
    tp = float(weight[alert & attack].sum())
    fp = float(weight[alert & benign].sum())
    per_family = {}
    for fam in np.unique(families[attack]):
        mask = families == fam
        n, hits = int(mask.sum()), int(alert[mask].sum())
        low, high = wilson_interval(hits, n)
        per_family[str(fam)] = {"n": n, "recall": hits / n, "wilson_95": [low, high]}
    return {
        "fpr": float(alert[benign].mean()),
        "recall": float(alert[attack].mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "alerts_per_10k_flows": float(np.average(alert, weights=weight) * 1e4),
        "per_family": per_family,
    }


def evaluate(y_test, families_test, test_scores, thresholds, weight=None, ap=True):
    """Score one model on one test set at the frozen thresholds.

    `ap=False` for detectors that are an alert rule rather than one score (the hybrid).
    """
    y_test = np.asarray(y_test)
    weight = np.ones(len(y_test)) if weight is None else np.asarray(weight, dtype=float)
    out = {
        "n_test": int(len(y_test)),
        "n_attack": int(y_test.sum()),
        "prevalence": float(np.average(y_test, weights=weight)),
        "average_precision": (float(average_precision_score(y_test, test_scores, sample_weight=weight))
                              if ap else None),
        "roc_auc": float(roc_auc_score(y_test, test_scores)) if ap else None,
    }
    if isinstance(test_scores, dict):  # hybrid: precomputed alerts per budget
        out["budgets"] = {b: at_threshold(y_test, families_test, test_scores[b], weight) for b in thresholds}
    else:
        out["budgets"] = {b: at_threshold(y_test, families_test, test_scores >= t, weight)
                          for b, t in thresholds.items()}
    return out


def _sha256(path, cache={}):
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        cache[key] = digest.hexdigest()
    return cache[key]


def provenance(data_paths):
    """Git commit, data hashes and package versions for a results file."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    versions = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "data_sha256": {str(Path(p).relative_to(ROOT)): _sha256(p) for p in data_paths},
        "versions": versions,
    }
