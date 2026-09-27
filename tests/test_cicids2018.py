import numpy as np
import pandas as pd

from src.data.cicids2018 import RENAME_2018_TO_2017, _prepare_chunk


def _raw_chunk(labels):
    features = [c for c in RENAME_2018_TO_2017 if c != "Label"]
    rows = [{c: "1" for c in features} | {"Label": label, "Timestamp": "x", "Protocol": "6"} for label in labels]
    return pd.DataFrame(rows)


def test_prepare_chunk_matches_2017_layout():
    chunk = _raw_chunk(["Benign"] * 1000 + ["Label", "SSH-Bruteforce", "Bot"])
    out = _prepare_chunk(chunk, np.random.RandomState(0))

    assert "Label" not in set(out["Label"])                      # repeated header line dropped
    assert {"SSH-Bruteforce", "Bot"} <= set(out["Label"])        # every attack kept
    assert 50 < (out["Label"] == "BENIGN").sum() < 150           # benign sampled at ~10%, renamed
    assert {"Timestamp", "Protocol", "Tot Fwd Pkts"}.isdisjoint(out.columns)
    assert {"Total Fwd Packets", "Fwd Header Length.1"} <= set(out.columns)
    assert out["Total Fwd Packets"].dtype.kind in "if"  # parsed as numbers, not strings
