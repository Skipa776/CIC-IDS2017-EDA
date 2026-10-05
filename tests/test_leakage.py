import numpy as np
import pandas as pd

from src.models.leakage import behavior_groups, grouped_partitions, purged_file_blocks


def test_behavior_groups_ignore_labels_and_file_names():
    from src.features.engineering import FAST_FEATURES

    df = pd.DataFrame(np.ones((100, len(FAST_FEATURES))), columns=FAST_FEATURES)
    df["Label"] = ["BENIGN", "Bot"] * 50
    df["Meta_source"] = ["file_a", "file_b"] * 50
    groups = behavior_groups(df)
    assert len(np.unique(groups)) == 1
    train, val, test = grouped_partitions(groups, 42)
    assert sorted([len(train), len(val), len(test)]) == [0, 0, 100]


def test_grouped_partitions_do_not_share_profiles():
    groups = np.repeat(np.arange(1000, dtype=np.uint64), 3)
    train, val, test = grouped_partitions(groups, 42)
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[val]) & set(groups[test])
    assert len(train) + len(val) + len(test) == len(groups)


def test_file_blocks_keep_test_tail_and_purge_shared_profiles():
    files = np.repeat(["a", "b"], 100)
    rows = np.tile(np.arange(100), 2)
    groups = np.arange(200, dtype=np.uint64)
    groups[0] = groups[90]  # a training row identical to a held-out profile
    groups[65] = groups[190]  # a validation row matching another file's test
    train, val, test = purged_file_blocks(files, rows, groups)
    assert 0 not in train
    assert 65 not in val
    assert 90 in test and 190 in test
    assert set(files[test]) == {"a", "b"}
    assert np.all(rows[test] >= 82)
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[val]) & set(groups[test])


def test_evaluation_cleaning_keeps_duplicates_and_conflicting_labels():
    from src.data.cleaning import CONSTANT_COLUMNS, DUPLICATE_COLUMNS, clean
    from src.features.engineering import FAST_FEATURES

    columns = list(dict.fromkeys(CONSTANT_COLUMNS + DUPLICATE_COLUMNS + FAST_FEATURES))
    raw = pd.DataFrame({feature: [1, 1, 1] for feature in columns})
    raw["Label"] = ["BENIGN", "Bot", "BENIGN"]
    kept, manifest = clean(raw, deduplicate=False)
    assert len(kept) == 3
    assert kept["Label"].tolist() == raw["Label"].tolist()
    assert manifest["rows_dropped_contradictory_labels"] == 0
    assert manifest["rows_dropped_duplicates"] == 0
    legacy, manifest = clean(raw)
    assert len(legacy) == 0
    assert manifest["rows_dropped_contradictory_labels"] == 2
