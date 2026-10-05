"""Builders for the three EDA notebooks (01a overview, 01b features, 01c data quality).

Imported by scripts/build_notebooks.py. EDA notebooks compute descriptive
statistics from the evaluation copy of CIC-IDS2017 (repeated flows and
conflicting labels kept, as in raw traffic); they never train or score a detector.
"""

import nbformat as nbf

from scripts.notebook_text import TEXT

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

LOAD = """import sys, warnings
sys.path.insert(0, "..")  # make src/ importable from notebooks/
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.metrics import roc_auc_score
from src.evaluation.folds import day_key
from src.models.cross_dataset import FAMILY
from src.reporting.plots import style, INK, MUTED
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

df = pd.read_parquet("../data/processed/cicids2017_eval.parquet")
features = df.select_dtypes(include=[np.number]).columns.tolist()
df["day"] = df["Meta_source"].map(lambda s: day_key(s, "2017"))
df["family"] = df["Label"].map(FAMILY)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ATTACK_FAMILIES = ["Brute force (FTP/SSH)", "DoS", "Heartbleed", "Web attack", "Infiltration", "Bot", "PortScan", "DDoS"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
BENIGN_C, ATTACK_C = "#2a78d6", "#eb6834"
print(f"CIC-IDS2017 evaluation copy: {len(df):,} flows, {len(features)} numeric features")
"""


def annotated_heatmap(var, value_fmt, title, cmap="BLUES", norm=None, figsize=(12, 5)):
    """Code that draws `var` (a DataFrame) as an annotated heatmap."""
    return f"""fig, ax = plt.subplots(figsize={figsize})
table = {var}
ax.imshow(table.values, cmap={cmap}, aspect="auto"{", norm=" + norm if norm else ""})
for (i, j), v in np.ndenumerate(table.values):
    if np.isfinite(v) and v != 0:
        ax.text(j, i, {value_fmt}, ha="center", va="center", fontsize=7.5,
                color="white" if v > table.values[np.isfinite(table.values)].max() * 0.55 else INK)
ax.set_xticks(range(table.shape[1])); ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=8.5)
ax.set_yticks(range(table.shape[0])); ax.set_yticklabels(table.index, fontsize=8.5)
ax.set_title({title!r}, loc="left", fontweight="bold", color=INK)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); plt.show()"""


def nb01a(write):
    write("01a_eda_overview", [
        md(TEXT["01a_intro"]),
        code(LOAD + """
attacks = df[df["Label"] != "BENIGN"]
days_per_family = attacks.groupby("family")["day"].nunique()
tcp = df.groupby("family")["Has_Init_Win_fwd"].mean()
print(f"Finding: all {len(days_per_family)} attack families occur on a single capture day, so a held-out day is a "
      f"held-out attack type. Every attack family completes a TCP handshake ({tcp.drop('Benign').min():.0%} of "
      f"flows show a TCP window) against {tcp['Benign']:.0%} of benign flows, and most attack families use one "
      f"destination port. Benign traffic's service mix is nearly identical across the five days.")"""),
        md("## Which attacks happen on which day"),
        code("""counts = df[df["Label"] != "BENIGN"].groupby(["family", "day"]).size().unstack(fill_value=0).reindex(columns=DAYS, fill_value=0)
counts = counts.loc[ATTACK_FAMILIES]
""" + annotated_heatmap("counts.replace(0, np.nan)", 'f"{int(v):,}"',
                        "Each attack family appears on exactly one day (CIC-IDS2017, attack flows)",
                        norm="LogNorm(vmin=1, vmax=counts.values.max())", figsize=(8, 4))),
        code("""per_day = df.groupby("day").agg(flows=("Label", "size"), attack_share=("Label", lambda s: (s != "BENIGN").mean())).reindex(DAYS)
per_day["attack_share"] = per_day["attack_share"].round(4)
per_day"""),
        md(TEXT["01a_services_md"]),
        code("""SERVICES = {53: "DNS (53)", 443: "HTTPS (443)", 80: "HTTP (80)", 8080: "HTTP alt (8080)", 21: "FTP (21)",
            22: "SSH (22)", 123: "NTP (123)", 137: "NetBIOS (137-139)", 138: "NetBIOS (137-139)",
            139: "NetBIOS (137-139)", 445: "SMB (445)", 444: "Port 444"}
def service(port):
    return SERVICES.get(port, "Other port < 1024" if port < 1024 else "Other port >= 1024")
df["service"] = df["Destination Port"].map(service)
columns = {f"Benign {d[:3]}": (df["Label"] == "BENIGN") & (df["day"] == d) for d in DAYS}
columns.update({f: df["family"] == f for f in ATTACK_FAMILIES})
order = ["DNS (53)", "HTTPS (443)", "HTTP (80)", "HTTP alt (8080)", "FTP (21)", "SSH (22)", "NTP (123)",
         "NetBIOS (137-139)", "SMB (445)", "Port 444", "Other port < 1024", "Other port >= 1024"]
service_mix = pd.DataFrame({c: df.loc[m, "service"].value_counts(normalize=True) for c, m in columns.items()}).reindex(order).fillna(0)
service_mix.loc["TCP window seen (any service)"] = [df.loc[m, "Has_Init_Win_fwd"].mean() for m in columns.values()]
""" + annotated_heatmap("service_mix", 'f"{v:.0%}" if v >= 0.005 else "<1%"',
                        "Share of flows by destination service: benign per day, then each attack family",
                        figsize=(13, 5.6))),
        md(TEXT["01a_body"]),
    ])


def nb01b(write):
    write("01b_eda_features", [
        md(TEXT["01b_intro"]),
        code(LOAD + """
X = df[features]
identical = []
for i, a in enumerate(features):
    for b in features[i + 1:]:
        if X[a].equals(X[b]):
            identical.append((a, b))
sample = df.sample(300_000, random_state=42)
rho = sample[features].rank().corr().fillna(0)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
dist = np.clip(1 - rho.abs().values, 0, None); np.fill_diagonal(dist, 0)
tree = linkage(squareform(dist, checks=False), "average")
groups = {t: len(set(fcluster(tree, 1 - t, "distance"))) for t in [0.99, 0.95, 0.9]}
q = X.quantile([0.5, 0.99]); zero_share = (X == 0).mean()
tails = pd.DataFrame({"zero share": zero_share, "median": q.loc[0.5], "99th pct": q.loc[0.99], "max": X.max()})
tails["max / 99th pct"] = tails["max"] / tails["99th pct"].replace(0, np.nan)
print(f"Finding: the {len(features)} features carry far less independent information than their count suggests. "
      f"{len(identical)} pairs of columns are identical in every flow, and grouping features whose rank correlation "
      f"is at least 0.95 leaves {groups[0.95]} groups. {int((zero_share > 0.5).sum())} features are zero in most "
      f"flows, and {int((tails['max / 99th pct'] > 10).sum())} have a maximum more than 10 times their 99th percentile.")"""),
        md(TEXT["01b_families_md"]),
        code("""def family_of(f):
    rules = [("Destination Port", "Flow metadata"), ("Flow Duration", "Flow metadata"), ("/s", "Rates"),
             ("IAT", "Inter-arrival times"), ("Flag", "TCP flags"), ("Init_Win", "TCP window"),
             ("Has_Init", "TCP window"), ("seg_size", "TCP window"), ("act_data", "TCP window"),
             ("Active", "Active / idle times"), ("Idle", "Active / idle times"), ("Subflow", "Subflow counts"),
             ("Header", "Header lengths"), ("Down/Up", "Ratios")]
    return next((fam for key, fam in rules if key in f), "Packet counts and sizes")
pd.Series(features).map(family_of).value_counts().rename_axis("feature family").to_frame("features")"""),
        md("## Heavy tails"),
        code("""tails.sort_values("max / 99th pct", ascending=False).head(10).round(1)"""),
        code("""benign = df[df["Label"] == "BENIGN"].sample(200_000, random_state=1)
attack = df[df["Label"] != "BENIGN"].sample(200_000, random_state=1)
show = ["Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
        "Flow Bytes/s", "Fwd IAT Min", "Init_Win_bytes_backward"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, f in zip(axes.flat, show):
    for frame, label, color in [(benign, "benign", BENIGN_C), (attack, "attack", ATTACK_C)]:
        v = np.sort(frame[f].to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, linewidth=2, label=label)
    ax.set_xscale("symlog", linthresh=1); ax.set_ylim(0, 1.01)
    ax.set_title(f, loc="left", fontsize=10, color=INK); style(ax); ax.grid(axis="y", color="#e4e3dc")
axes[0, 0].legend(frameon=False); axes[0, 0].set_ylabel("share of flows at or below x"); axes[1, 0].set_ylabel("share of flows at or below x")
fig.suptitle("Values span many orders of magnitude; attack and benign differ in shape, not just level (log-scaled x)",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout(); plt.show()"""),
        md(TEXT["01b_redundancy_md"]),
        code("""pd.DataFrame(identical, columns=["column", "identical in every flow to"])"""),
        code("""print("Independent groups of features at each rank-correlation threshold:", groups)
cl = fcluster(tree, 0.05, "distance")
big = pd.Series(features, index=cl).groupby(level=0).apply(list)
pd.DataFrame({"features with rank correlation >= 0.95": [", ".join(g) for g in big if len(g) >= 3]})"""),
        md(TEXT["01b_separation_md"]),
        code("""sep = {}
for fam in ATTACK_FAMILIES:
    rows = df[df["family"] == fam]
    rows = rows.sample(min(len(rows), 50_000), random_state=1)
    y = np.r_[np.zeros(len(benign)), np.ones(len(rows))]
    sep[fam] = {f: 2 * roc_auc_score(y, np.r_[benign[f].to_numpy(), rows[f].to_numpy()]) - 1 for f in features}
sep = pd.DataFrame(sep).T   # +1: attack always higher, -1: attack always lower, 0: no separation
top = sorted({f for fam in sep.index for f in sep.loc[fam].abs().nlargest(3).index})
diverging = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f2f1ec", "#d95926"])
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.imshow(sep[top].values, cmap=diverging, vmin=-1, vmax=1, aspect="auto")
for (i, j), v in np.ndenumerate(sep[top].values):
    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.7 else INK)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(sep))); ax.set_yticklabels(sep.index, fontsize=8.5)
ax.set_title("How well one feature separates each attack family from benign (orange: attack higher; blue: lower)",
             loc="left", fontweight="bold", color=INK, fontsize=10.5)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); plt.show()"""),
        code("""# Each family against benign on its most separating continuous feature
# (destination port is a category and yes/no indicators make a single step, so both are excluded)
not_continuous = ["Destination Port"] + [f for f in features if df[f].nunique() <= 2]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, fam in zip(axes.flat, ATTACK_FAMILIES):
    f = sep.loc[fam].drop(not_continuous).abs().idxmax()
    rows = df[df["family"] == fam]
    for frame, label, color in [(benign, "benign", BENIGN_C), (rows, fam, ATTACK_C)]:
        v = np.sort(frame[f].to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, linewidth=2, label=label)
    ax.set_xscale("symlog", linthresh=1); ax.set_ylim(0, 1.01)
    ax.set_title(f"{fam}\\n{f} (separation {sep.loc[fam, f]:+.2f})", loc="left", fontsize=9, color=INK)
    style(ax); ax.grid(axis="y", color="#e4e3dc")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("Each attack family against benign traffic on its single most separating feature (log-scaled x)",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout(); plt.show()"""),
        md(TEXT["01b_body"]),
    ])


def nb01c(write):
    write("01c_eda_data_quality", [
        md(TEXT["01c_intro"]),
        code(LOAD + """
df["vector"] = pd.util.hash_pandas_object(df[features], index=False).to_numpy()
df["exact_duplicate"] = df.duplicated(subset=["vector", "Label"], keep="first")
labels_per_vector = df.groupby("vector")["Label"].transform("nunique")
conflicts = df[labels_per_vector > 1]
empty = (df["Total Length of Fwd Packets"] == 0) & (df["Total Length of Bwd Packets"] == 0)
df["empty"] = empty
web = df[df["family"] == "Web attack"]
print(f"Finding: {df['exact_duplicate'].mean():.1%} of flows repeat an earlier flow exactly (up to "
      f"{df.groupby('Label')['exact_duplicate'].mean().max():.0%} for some attacks); {conflicts['vector'].nunique():,} "
      f"distinct flows carry contradictory labels; and {web['empty'].mean():.0%} of web-attack flows carry no "
      f"payload at all, so most 'web attacks' in the labels are not attack traffic.")"""),
        md("## Repeated flows and empty flows, by label"),
        code("""by_label = df.groupby("Label").agg(flows=("Label", "size"), exact_duplicate=("exact_duplicate", "mean"),
                                   no_payload=("empty", "mean")).sort_values("flows", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
order = by_label.index[::-1]
for ax, col, title in [(axes[0], "exact_duplicate", "Share of flows that repeat an earlier flow exactly"),
                       (axes[1], "no_payload", "Share of flows with zero payload bytes in both directions")]:
    ax.barh(order, by_label.loc[order, col], color=["#9ec5f4" if l == "BENIGN" else "#2a78d6" for l in order], height=0.65)
    for i, v in enumerate(by_label.loc[order, col]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.12); ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK); style(ax)
axes[0].tick_params(axis="y", labelsize=8.5)
fig.tight_layout(); plt.show()
by_label.round(3)"""),
        md(TEXT["01c_conflicts_md"]),
        code("""pairs = conflicts.groupby("vector").agg(labels=("Label", lambda s: " vs ".join(sorted(set(s)))),
                                        days=("day", lambda s: " / ".join(sorted(set(s), key=DAYS.index))),
                                        rows=("Label", "size"))
pairs["same day"] = ~pairs["days"].str.contains("/")
pairs.groupby(["labels", "same day"]).agg(distinct_flows=("rows", "size"), rows=("rows", "sum")).sort_values("rows", ascending=False)"""),
        code("""# One example: the same feature vector, labelled benign on one day and PortScan on another
example = pairs[pairs["labels"] == "BENIGN vs PortScan"].index[0]
cols = ["day", "Label", "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "SYN Flag Count", "RST Flag Count", "Init_Win_bytes_forward"]
conflicts.loc[conflicts["vector"] == example, cols].drop_duplicates(subset=["day", "Label"])"""),
        md(TEXT["01c_empty_md"]),
        code("""rows = []
for label in ["Web Attack - XSS", "Web Attack - Brute Force", "Web Attack - Sql Injection"]:
    w = df[df["Label"] == label]; e = w[w["empty"]]
    shape = e[["Total Fwd Packets", "Total Backward Packets"]].value_counts(normalize=True)
    rows.append({"label": label, "flows": len(w), "with payload": int((~w["empty"]).sum()),
                 "empty": len(e), "most common empty shape (fwd, bwd packets)": f"{shape.index[0]} in {shape.iloc[0]:.0%}",
                 "empty: median duration (s)": round(e["Flow Duration"].median() / 1e6, 1),
                 "empty: share with a FIN": round((e["FIN Flag Count"] > 0).mean(), 3),
                 "empty: server window 28,960": round((e["Init_Win_bytes_backward"] == 28960).mean(), 3),
                 "with payload: median forward bytes": w.loc[~w["empty"], "Total Length of Fwd Packets"].median()})
pd.DataFrame(rows).set_index("label").T"""),
        md(TEXT["01c_body"]),
    ])
