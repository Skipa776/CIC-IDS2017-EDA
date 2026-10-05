"""Load CSE-CIC-IDS2018 in the same shape as the raw CIC-IDS2017 CSVs.

CSE-CIC-IDS2018 uses the same CICFlowMeter features as CIC-IDS2017 under
abbreviated names. After renaming, the frame goes through the unchanged
CIC-IDS2017 cleaning pipeline (src/data/cleaning.py), so both datasets are
cleaned identically.

Benign flows (about 13M) are sampled at BENIGN_SAMPLE_RATE while loading so
the data fits in memory; every attack flow is kept. Metrics that depend on
prevalence (PR-AUC) must weight benign rows by 1 / BENIGN_SAMPLE_RATE.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR_2018 = PROJECT_ROOT / "data" / "raw" / "CSE-CIC-IDS2018"
BENIGN_SAMPLE_RATE = 0.1

# One file (Thuesday-20-02-2018) also carries these; the others do not
DROP_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Protocol", "Timestamp"]

RENAME_2018_TO_2017 = {
    "Dst Port": "Destination Port",
    "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWE Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
    "Label": "Label",
}


def _prepare_chunk(chunk: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    chunk = chunk.drop(columns=DROP_COLUMNS, errors="ignore")
    chunk = chunk[chunk["Label"] != "Label"]  # header lines repeated inside some files
    keep = (chunk["Label"] != "Benign") | (rng.random_sample(len(chunk)) < BENIGN_SAMPLE_RATE)
    chunk = chunk[keep].rename(columns=RENAME_2018_TO_2017)
    features = [c for c in chunk.columns if c != "Label"]
    chunk[features] = chunk[features].apply(pd.to_numeric, errors="coerce")
    # CIC-IDS2017 has Fwd Header Length twice; recreate it so clean() runs unchanged
    chunk["Fwd Header Length.1"] = chunk["Fwd Header Length"]
    return chunk.assign(Label=chunk["Label"].replace("Benign", "BENIGN"))


def load_raw_2018(raw_dir: Path = RAW_DIR_2018, seed: int = 42) -> pd.DataFrame:
    """Load all CSE-CIC-IDS2018 CSVs with 2017 column names and a Meta_source column."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if len(csv_files) != 10:
        raise FileNotFoundError(f"Expected 10 CSE-CIC-IDS2018 CSVs in {raw_dir}, found {len(csv_files)}")
    rng = np.random.RandomState(seed)
    frames = []
    for path in csv_files:
        source = os.path.basename(path).replace("_TrafficForML_CICFlowMeter.csv", "")
        for chunk in pd.read_csv(path, chunksize=500_000, dtype=str):
            frames.append(_prepare_chunk(chunk, rng).assign(Meta_source=source))
    df = pd.concat(frames, ignore_index=True)
    missing = set(RENAME_2018_TO_2017.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing after rename: {sorted(missing)}")
    return df
