#!/usr/bin/env python3
"""
Attack-behavior clustering mapped to MITRE ATT&CK techniques.

KMeans clusters attack flows from the v2 dataset, characterizes each cluster
(dominant attack label, purity, distinguishing features vs benign traffic),
and maps clusters to MITRE ATT&CK techniques via the static MITRE_MAPPINGS
table served by the API.

Usage:
    python scripts/attack_clustering.py

Output:
    - reports/attack_clusters.json
    - reports/figures/attack_cluster_composition.png
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.services.mitre_mapping import MITRE_MAPPINGS

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean_v2.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "attack_clusters.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "attack_cluster_composition.png"

MAX_PER_ATTACK = 5_000
BENIGN_REFERENCE = 50_000
K_RANGE = range(8, 17)
SILHOUETTE_SUBSAMPLE = 10_000
RANDOM_STATE = 42


def load_samples():
    print(f"Loading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    attacks = df[df["Label"] != "BENIGN"]
    attack_sample = attacks.groupby("Label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), MAX_PER_ATTACK), random_state=RANDOM_STATE)
    )
    benign_ref = df[df["Label"] == "BENIGN"].sample(
        BENIGN_REFERENCE, random_state=RANDOM_STATE
    )
    print(f"  {len(attack_sample):,} attack flows sampled "
          f"({attack_sample['Label'].nunique()} types), "
          f"{len(benign_ref):,} benign reference flows")
    return attack_sample, benign_ref, feature_cols


def select_k(X_scaled):
    """Pick k by silhouette score on a subsample."""
    rng = np.random.RandomState(RANDOM_STATE)
    sub = rng.choice(len(X_scaled), min(SILHOUETTE_SUBSAMPLE, len(X_scaled)), replace=False)
    best_k, best_score, scores = None, -1.0, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled[sub], labels[sub])
        scores[k] = round(float(score), 4)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k, scores


def characterize_clusters(attack_sample, benign_ref, feature_cols, cluster_labels):
    """Dominant label, purity, and distinguishing features per cluster."""
    benign_median = benign_ref[feature_cols].median()
    benign_std = benign_ref[feature_cols].std().replace(0, np.nan)

    clusters = []
    for c in sorted(np.unique(cluster_labels)):
        members = attack_sample[cluster_labels == c]
        counts = members["Label"].value_counts()
        dominant = counts.index[0]
        purity = float(counts.iloc[0] / len(members))

        # Features where the cluster deviates most from benign behavior
        deviation = ((members[feature_cols].median() - benign_median) / benign_std)
        top = deviation.abs().sort_values(ascending=False).head(5)
        signature = {
            feat: round(float(deviation[feat]), 2) for feat in top.index
        }

        mitre = MITRE_MAPPINGS.get(dominant, {})
        clusters.append({
            "cluster": int(c),
            "size": int(len(members)),
            "dominant_label": dominant,
            "purity": round(purity, 3),
            "label_distribution": {k: int(v) for k, v in counts.items()},
            "signature_features_z_vs_benign": signature,
            "mitre_technique_id": mitre.get("technique_id"),
            "mitre_technique_name": mitre.get("technique_name"),
            "mitre_tactic": mitre.get("tactic"),
        })
    return clusters


def save_figure(attack_sample, cluster_labels):
    comp = pd.crosstab(
        cluster_labels, attack_sample["Label"], normalize="index"
    )
    plt.figure(figsize=(12, 7))
    sns.heatmap(comp, cmap="viridis", annot=True, fmt=".2f", annot_kws={"size": 6})
    plt.xlabel("Attack type")
    plt.ylabel("Cluster")
    plt.title("Cluster composition (row-normalized share of attack types)")
    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=120)
    plt.close()
    print(f"Figure saved: {FIGURE_PATH}")


def main():
    attack_sample, benign_ref, feature_cols = load_samples()

    X_scaled = StandardScaler().fit_transform(attack_sample[feature_cols])

    print("\nSelecting k by silhouette score...")
    best_k, silhouette_scores = select_k(X_scaled)
    print(f"  chosen k={best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = km.fit_predict(X_scaled)

    print("\nCharacterizing clusters...")
    clusters = characterize_clusters(
        attack_sample, benign_ref, feature_cols, cluster_labels
    )
    for cl in clusters:
        print(f"  cluster {cl['cluster']:>2d}: n={cl['size']:>6,}  "
              f"{cl['dominant_label']:<28s} purity={cl['purity']:.2f}  "
              f"-> {cl['mitre_technique_id']}")

    techniques = sorted({
        cl["mitre_technique_id"] for cl in clusters if cl["mitre_technique_id"]
    })
    print(f"\nDistinct MITRE techniques covered by clusters: {len(techniques)}")
    print(f"  {techniques}")

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(DATA_PATH),
        "n_attack_flows_clustered": int(len(attack_sample)),
        "k": int(best_k),
        "silhouette_scores": silhouette_scores,
        "distinct_techniques": techniques,
        "clusters": clusters,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {REPORT_PATH}")

    save_figure(attack_sample, cluster_labels)


if __name__ == "__main__":
    main()
