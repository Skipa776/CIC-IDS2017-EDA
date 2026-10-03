"""Label-blind split controls for duplicate and related-flow contamination.

Behavior bins are a reproducible sensitivity analysis, not inferred session IDs.
Raw CSV order is preserved for block holdouts but is not assumed to be time order.
"""

import numpy as np
import pandas as pd

from src.features.engineering import FAST_FEATURES


def behavior_groups(df, bin_width=0.25):
    """Hash fixed behavior bins; no labels, provenance or learned statistics.

    Ports, flags and initial windows are exact. The other 20-model input
    features use signed log2(1+abs(x)) bins. Quarter-octave bins span about
    19% in 1+abs(x). Equal inputs always belong to the same global group,
    including across files. Nearby inputs across a bin boundary can differ.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    q = df[FAST_FEATURES].copy()
    for feature in FAST_FEATURES:
        if not any(token in feature for token in ["Port", "Flag", "Init_Win"]):
            values = q[feature].to_numpy(dtype=float)
            q[feature] = np.sign(values) * np.floor(np.log2(1 + np.abs(values)) / bin_width)
    return pd.util.hash_pandas_object(q, index=False).to_numpy()


def grouped_partitions(groups, seed):
    """Assign complete global profiles to 60/20/20 partitions without labels."""
    # SplitMix64 mixes sequential/profile hashes reproducibly. Integer wrap is
    # intentional, and assignment never depends on labels or requested scores.
    mixed = np.asarray(groups, dtype=np.uint64) ^ np.uint64(seed)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    bucket = mixed % np.uint64(100)
    return np.flatnonzero(bucket < 60), np.flatnonzero((bucket >= 60) & (bucket < 80)), np.flatnonzero(bucket >= 80)


def purged_file_blocks(files, raw_rows, groups, file_lengths=None):
    """Hold out the tail of EVERY file with gaps and global profile purging.

    Train [0,.58), validate [.62,.78), test [.82,1). Discard intervening
    gaps. Test keeps priority; remove its profiles from validation/training,
    then remove surviving validation profiles from training. No labels used.
    This measures file-order/block transfer, not verified chronological transfer.
    """
    files = np.asarray(files)
    raw_rows = np.asarray(raw_rows)
    fraction = np.empty(len(files), dtype=float)
    for source in np.unique(files):
        mask = files == source
        length = file_lengths[source] if file_lengths else int(raw_rows[mask].max()) + 1
        fraction[mask] = raw_rows[mask] / length
    train = np.flatnonzero(fraction < 0.58)
    val = np.flatnonzero((fraction >= 0.62) & (fraction < 0.78))
    test = np.flatnonzero(fraction >= 0.82)
    return purge_profiles(train, val, test, groups)


def purge_profiles(train, val, test, groups):
    """Retain test, remove its profiles from validation, then purge training."""
    val = val[~np.isin(groups[val], groups[test])]
    train = train[~np.isin(groups[train], np.r_[groups[val], groups[test]])]
    return train, val, test


def overlap_report(train, test, exact, profiles, y, labels):
    """Quantify test rows whose inputs/profiles were already in training."""
    seen_exact = np.isin(exact[test], exact[train])
    seen_profile = np.isin(profiles[test], profiles[train])
    return {
        "n_test": len(test),
        "exact_input_overlap_n": int(seen_exact.sum()),
        "exact_input_overlap_fraction": float(seen_exact.mean()),
        "behavior_profile_overlap_fraction": float(seen_profile.mean()),
        "attack_exact_overlap_n": int(seen_exact[y[test] == 1].sum()),
        "attack_profile_overlap_fraction": float(seen_profile[y[test] == 1].mean()),
        "per_label_profile_overlap": {str(label): float(seen_profile[labels[test] == label].mean())
                                      for label in np.unique(labels[test])},
    }
