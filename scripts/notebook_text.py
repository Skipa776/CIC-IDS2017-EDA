"""Narrative cells for scripts/build_notebooks.py.

Written after reading the results files. Every number in a markdown cell here
must also be printed by a code cell in the same notebook, so a stale number is
visible next to the live one.
"""

TEXT = {}

TEXT["00_intro"] = """# 00 · Data and cleaning

**Question.** What is in the raw data, and what did cleaning keep and why?

The finding, its figure and the numbers come first; how they were produced follows."""

TEXT["00_body"] = """## What the cleaning does

The same code (`src/data/cleaning.py`) cleans both datasets. It strips column names, fixes the
garbled web-attack labels, drops 8 constant columns and one duplicate column, fills genuine
missing rates with 0, and drops rows whose rate columns are infinite or whose counts are negative
where negatives are impossible. Inter-arrival times can be negative in this data and are kept.

**The sentinel bug.** `Init_Win_bytes_forward/backward = -1` means "no TCP window was observed"
(UDP flows, flows without a handshake). The first cleaning pass treated it as corrupt and deleted
those rows: half the dataset, mostly ordinary benign traffic. The fix keeps them, adds
`Has_Init_Win_fwd/bwd` indicator columns, and sets the -1 to 0.

**Two copies of each dataset.**

- *Training copy* (`*_clean*.parquet`): exact duplicates and contradictory labels removed. Used by
  the older reports and the served model.
- *Evaluation copy* (`*_eval.parquet`, `python scripts/build_dataset.py --year YYYY --no-dedup`):
  keeps repeated flows and conflicting labels, because real traffic has both, and because removing
  contradictory labels across the whole dataset uses test labels before any split. All numbered
  notebooks 03-06 use the evaluation copy.

**CSE-CIC-IDS2018.** Its columns are renamed onto the 2017 names (`src/data/cicids2018.py`; all
78 columns map one to one). To fit in memory, 10% of benign flows are kept and every attack flow;
prevalence-dependent metrics weight benign rows by 10.

## Limits

- The first-pass row count comes from the old parquet file, which is kept only for this comparison.
- Cleaning cannot fix label errors in the source data (Engelen, Rimmer & Joosen, 2021)."""

TEXT["02_intro"] = """# 02 · Why random splits score near 1.0

**Question.** Why does a random train/test split give near-perfect scores, and do the
leakage checks pass?

A score of 1.0 is not impossible: it means every test attack outranked every benign flow. It is a
red flag that has to be explained before it is shown."""

TEXT["02_gates"] = """## The leakage gates

No number reaches the README until these hold. The table below the printout shows them for every
fold of the forward-in-time evaluation (notebooks 03-06).

| Gate | Pass rule |
| --- | --- |
| No exact twins | Report test rows whose model inputs exactly match a training row. Across days some twins are legitimate (identical benign DNS lookups, repeated flood packets); inside one capture they are leakage. |
| Feature choice inside training data | All 71 features are used; no selection step. The old 20-feature subset was picked using importance on all data, so it is not used here. |
| Shuffled-label control | A model trained on shuffled labels scores near the prevalence. |
| Shortcut check | Flag any fold where one feature alone comes within 0.05 average precision of the full model. |
| Thresholds from validation only | No threshold or model choice uses test labels; the actual test false positive rate is reported. |"""

TEXT["02_body"] = """## Interpretation

The near-perfect score is mostly not a code bug. Exact input twins between the old random split's
train and test sets were few (see the printout), the shuffled-label control sits at chance, and
grouping similar flows so they cannot straddle the split still leaves the score near 1.0.

The explanation is that the question is easy. Inside one lab capture, scripted attacks from a few
attacker machines are trivially separable from benign traffic. A random split asks "does the model
recognize attacks it has already seen, from the same capture?" That is a sanity check, not a
result. The forward-in-time notebooks (03-06) ask the question that matters for an intrusion
detector: can it flag the attacks of a day it has not seen?

## Limits

- Grouping similar flows uses coarse bins of 20 features; it is a proxy, not real session IDs.
- The random-row result uses the deduplicated training copy (`generalization_evaluation.json`);
  the other rows use the raw-derived population (`leakage_audit.json`).
- All of these captures have been looked at before, so none is a fresh, untouched test."""

# A computed finding: best detector whose actual test FPR stayed within budget, per fold.
FINDING_CODE = '''
def finding(folds, labels=None, caveats=None):
    for f in folds:
        d = summary["folds"][f]
        quotable = {n: v["budgets"]["0.01"]["recall"]["median"]
                    for n, v in d["detectors"].items() if not v["budget_violated"]["0.01"]}
        broke = [NAMES[n] for n, v in d["detectors"].items() if v["budget_violated"]["0.01"]]
        # ties go to the single model (a hybrid that gave one model the whole budget is that model)
        best = max(quotable, key=lambda n: (quotable[n], "+" not in n)) if quotable else None
        line = f"{(labels or {}).get(f, f)}: "
        if d["unseen_test_families"]:
            line += f"unseen families {', '.join(d['unseen_test_families'])}. "
        line += (f"Best detector within its false-alarm budget: {NAMES[best]}, recall {quotable[best]:.1%} "
                 f"(median of {len(d['seeds'])} seeds)" if best else "No detector stayed within its false-alarm budget")
        if broke:
            line += f". Over budget (recall not quotable): {', '.join(broke)}"
        if best and d["detectors"][best].get("shortcut_flag"):
            line += f". Shortcut flag: one feature ({', '.join(d['best_single_feature'])}) comes within 0.05 AP of it"
        if (caveats or {}).get(f):
            line += f". CAVEAT: {caveats[f]}"
        print(line + ".")
'''

TEXT["03_intro"] = """# 03 · Forward in time on CIC-IDS2017

**Question.** Can a model trained on earlier capture days catch the next day's attacks, all of
which are attack types it has never seen?

Three folds, five seeds each. Thresholds are frozen on the validation day's benign flows at a 1%
budget before the test day is scored. A recall counts only if the actual test false positive rate
stayed within 1.5 times the budget; otherwise it is marked "over budget"."""

TEXT["03_finding_code"] = FINDING_CODE + '''
finding(["2017_A", "2017_B", "2017_C"], {"2017_A": "Fold A (train Mon, test Wed)",
        "2017_B": "Fold B (train Mon-Tue, test Thu)", "2017_C": "Fold C (train Mon-Wed, test Fri)"},
        {"2017_B": "this recall rides on one victim-server TCP window value, not on attack behaviour "
                   "(see 'Explaining fold B')"})
'''

TEXT["03_explain"] = """## Explaining fold B before using it

Fold B is the one strong unseen-attack result: LightGBM, trained on Tuesday's FTP and SSH brute
force, catches most of Thursday's web attacks. The design required explaining it before using it.
`scripts/explain_forward_thursday.py` retrains the model without each of its top features."""

TEXT["03_body"] = """## What this shows

- **Fold B is an artifact, not transfer.** The recall rides on one feature,
  `Init_Win_bytes_backward`. Most Thursday web-attack flows carry the same backward TCP window
  value, the one printed above, which is likely the victim web server's default window; very few
  benign flows have it. Without that feature, recall falls to a few percent. The model learned
  "this server answered", not "this is an attack".
- **Fold C's DDoS detection is plausible transfer, with a caveat.** Trained on Wednesday's DoS,
  LightGBM flags most of Friday's DDoS, a closely related attack. One feature alone
  (`Fwd Packet Length Max`) comes close to the model, so the shortcut flag is raised. PortScan and
  Bot, which look nothing like Tuesday's or Wednesday's attacks, are almost never caught.
- **The anomaly models break their budgets.** Isolation Forest and the autoencoder are fitted to
  earlier days' benign traffic. On a new day, normal traffic shifts, and their "1%" thresholds let
  through 2-7% of benign flows. Their recall is therefore marked over budget.
- **Seeds matter for anomaly models.** The autoencoder's fold C recall ranges widely across
  seeds (see the table); LightGBM and logistic regression are stable.

## Limits

Five seeds measure fitting variation, not uncertainty about future traffic. Every day here has
been looked at before, so these are development results, not a fresh test."""

LABELS_2018 = {"2018_new_tool_dos": "Same family, new tool: DoS (16 Feb)",
               "2018_new_tool_ddos": "Same family, new tool: DDoS (21 Feb)",
               "2018_same_attack_web": "Same attack, later date: web (23 Feb)",
               "2018_same_attack_infiltration": "Same attack, later date: infiltration (1 Mar)",
               "2018_new_family_bot": "New family: Bot (2 Mar)"}

TEXT["04_intro"] = """# 04 · Forward in time on CSE-CIC-IDS2018

**Question.** Does detection hold up for the same attack on a later date, the same attack family
with a new tool, and a brand-new family?

2018 repeats some attack families on later days, so it can ask three different questions that
2017 cannot. Every fold trains on all days before the test day. Thresholds are frozen at a 1%
budget on validation benign flows: a 20% holdout of the training days for the "same attack" and
"new tool" folds, and the previous day for the "new family" fold."""

TEXT["04_finding_code"] = FINDING_CODE + f'''
LABELS_2018 = {LABELS_2018!r}
finding(list(LABELS_2018), LABELS_2018)
'''

TEXT["05_intro"] = """# 05 · Can a hybrid catch more unseen attacks?

**Question.** Does alerting when either a supervised model or an anomaly model fires catch more
attacks than either alone, by the success criteria fixed before any run?

The hybrid splits the 1% alert budget between the two models. The anomaly partner (Isolation Forest
or autoencoder) and the split are chosen on validation data only. The criteria, from the design doc:

1. Higher recall than the best single model on folds 2017_B, 2017_C and the 2018 new-family fold, in
   every seed. (Fold 2017_A has no supervised model, so no hybrid.)
2. Actual test false positive rate at most 1.5% on every fold and seed.
3. No attack family drops to zero that a single model caught."""

TEXT["05_finding_code"] = '''
crit = summary["success_criteria"]
passed = [h for h, r in crit.items() if r["overall_pass"]]
print("Finding: " + (f"{', '.join(passed)} met every criterion." if passed else
      "no hybrid met the success criteria; the README reports the best single model instead."))
for h, r in crit.items():
    c1 = r["criterion_1_beats_best_single"]["runs"]
    wins = sum(x["pass"] for x in c1)
    print(f"  {NAMES[h]}: beat the best single model in {wins} of {len(c1)} criterion runs; "
          f"worst actual FPR {r['criterion_2_fpr_at_most_1.5pct']['worst_fpr']:.2%}; "
          f"families lost: {len(r['criterion_3_no_family_lost']['families_lost'])}.")
'''

TEXT["06_intro"] = """# 06 · Does a 2017 model work on 2018?

**Question.** Trained on all of CIC-IDS2017, does a detector work on CSE-CIC-IDS2018, recorded a
year later on a different network with the same flow tool? This is forward in time and on a new
network at once.

Thresholds are frozen on a 20% holdout of 2017 benign traffic. The actual false positive rate on
2018 shows how far normal traffic moved."""

TEXT["06_finding_code"] = FINDING_CODE + '''
finding(["2017_to_2018"], {"2017_to_2018": "Train all 2017, test all 2018"})
'''

TEXT["06_why"] = """## Why it fails: the same column, measured differently

The flow tool changed between the two captures. Single-packet DNS queries are the same kind of
flow in both years, so their header-length features should match. They do not:"""

TEXT["04_body"] = """## What this shows

- **Same attack, later date: one clean success, one failure.** The web attacks of 22 Feb return on
  23 Feb, and LightGBM catches most of them within its budget, with no single-feature shortcut.
  This is the most trustworthy positive result in the project. Infiltration, which also repeats,
  is not caught: LightGBM's frozen threshold lets through 13-17% of benign flows on the new day,
  and logistic regression stays in budget but catches almost nothing.
- **Same family, new tool: caught, but through shortcuts.** Logistic regression catches the new DoS
  and DDoS tools within budget, but one feature alone (destination port for DoS; forward packet
  length for DDoS) reaches AP 0.999. On the DoS day, 23% of test attack flows are exact copies of
  flows from the training days: flood traffic repeats byte for byte. These are not evidence of
  learning what DoS is.
- **New family: nothing works.** Bot on 2 Mar is caught by no detector, at any seed.
- **Frozen thresholds drift.** On the DoS day, LightGBM's "1%" threshold flags 82-99% of benign
  flows. A threshold chosen on earlier days is not safe on a new day without recalibration.

## Limits

2018's benign flows are a 10% sample (prevalence-weighted). The web folds have only a few hundred
attack flows, so their per-family intervals are wide (see the results file)."""

TEXT["05_body"] = """## What this shows

The hybrid fails all three criteria.

- **The validation day points the wrong way.** On fold 2017_B the validation day (Wednesday, DoS)
  rewards the autoencoder, so the hybrid gives it the whole budget; the test day (Thursday, web)
  rewards LightGBM. Choosing settings on one unseen family does not carry over to another.
- **Anomaly models bring their drift with them.** Whenever the hybrid leans on Isolation Forest or
  the autoencoder, it inherits their broken false-alarm budgets on new days.
- **Families get lost.** Splitting the budget leaves each model too few alerts, so families that
  a single model caught fall to zero (web attacks on 2017_B, Bot on 2017_C).

The earlier informal result that suggested a gain (recall 0.675 on Wed+Thu) picked its budget
split after looking at the test days. With the split chosen on validation data only, the gain does
not appear. Per the design, the README reports the best single model instead.

## Limits

The hybrid is one simple combination rule. A learned combination might do better, but it would
need validation data that represents unseen attacks, which is the very thing this data cannot
supply."""

TEXT["06_body"] = """## What this shows

- **Within budget, only logistic regression is quotable, and it catches little.** LightGBM ranks
  2018 traffic better (higher AP) and catches DoS and brute force well, but its frozen threshold
  lets through 2.5-5.9% of 2018 benign flows: normal traffic moved between the two networks.
- **The flow tool changed what columns mean.** Single-packet DNS queries carry an 8-byte forward
  header in 2018 (the true UDP header size) and 20-40 bytes in 2017. Same column name, different
  measurement. `min_seg_size_forward` is also the single most separating feature on this fold.
- **Attack names hide different behaviour.** The archived diagnostics
  (`notebooks/archive/cross_year_diagnostics.ipynb`) show, for example, 2018 SSH brute-force flows
  about 32 times shorter than 2017's, and 2018 DoS Hulk flows with no reply packets.

## What would fix it

Not a model change. The data needs consistent feature extraction across captures (one flow tool
version, validated by checks like the DNS test above), and thresholds need recalibrating on recent
benign traffic from the network being monitored.

## Limits

Both datasets come from the same lab and flow tool, so this is a mild test of transfer. A dataset
from a different tool would need retraining on a different feature set."""

TEXT["01a_intro"] = """# 01a · EDA: what is in the capture

**Question.** What traffic does each capture day contain, which services does it use, and how
do attacks differ from benign traffic before looking at any flow statistic?

Data: the CIC-IDS2017 evaluation copy (`00_data_and_cleaning`), which keeps repeated flows and
conflicting labels as the raw capture has them."""

TEXT["01a_services_md"] = """## Services and protocol

The 2017 CSVs have no protocol column. Two proxies stand in for it: the destination port (the
service a flow talks to) and whether a TCP window was observed (`Has_Init_Win_fwd`), which is
true for TCP flows that completed a handshake and false for UDP, ICMP and handshake-less flows."""

TEXT["01a_body"] = """## What this shows

- **Day and attack are the same thing in 2017.** Each family sits on one day and Monday has none,
  so any split by day is also a split by attack type (`03_forward_2017`).
- **Benign traffic is stable across days.** DNS, HTTPS and HTTP dominate every day in nearly the
  same proportions. Day-to-day drift in benign traffic is small inside this one capture, which is
  why the bigger drift appears between years (`06_cross_year`).
- **Most attack families are one service.** Every DoS, DDoS and web attack goes to port 80;
  FTP-Patator to 21; SSH-Patator to 22; Bot mostly to 8080; Heartbleed and Infiltration to port
  444. PortScan is the exception, spread across ports by design. A model can therefore learn
  "port 22 means attack" on this data, which is a lab artifact, not attack behaviour.
- **Every attack family completes a TCP handshake; many benign flows do not.** Roughly 44% of
  benign flows are UDP or handshake-less (mostly DNS). A detector can lean on "is this TCP?",
  which separates nothing on a network where attacks also use UDP.

## Limits

Port is the destination port only; the source side and IP addresses are not in this export."""

TEXT["01b_intro"] = """# 01b · EDA: the features

**Question.** What do the 71 flow features look like, how much independent information do they
carry, and which of them separate each attack family from benign traffic?

All statistics are descriptive. Correlations and separation scores use fixed random samples
(seeds in the code) of 200,000-300,000 flows; the full data gives the same picture."""

TEXT["01b_families_md"] = """## Feature families

The features come from the CICFlowMeter tool: per-flow counts, sizes, timing and TCP-flag
statistics, computed separately for the forward (client to server) and backward directions."""

TEXT["01b_redundancy_md"] = """## Redundancy

Some columns are not just correlated but identical in every one of the 2.8M flows. Two of these
are suspicious rather than redundant: a SYN flag count equal to the forward PSH-flag count, and a
CWE flag count equal to the forward URG-flag count, are not plausible traffic. They point to
fields the flow tool mislabels or copies. The subflow pairs are identical because each flow here
has a single subflow.

Beyond exact copies, many features move together: timing statistics (total, mean, max of the
same inter-arrival times) and size statistics (total, mean, max of the same packets)."""

TEXT["01b_separation_md"] = """## What separates each attack family from benign traffic

For each family and each feature, the separation score is 2 × ROC AUC − 1 of that single feature
against a benign sample: +1 means the family's values are always higher than benign, -1 always
lower, 0 no separation. The heatmap shows each family's three most separating features. This is
description, not feature selection: no model uses these scores."""

TEXT["01b_body"] = """## What this shows

- **Far fewer signals than columns.** Four column pairs are identical, and at a rank correlation of
  0.95 the 71 features collapse into a few dozen groups. Models that report "71 features" are
  working with much less independent information.
- **Heavy tails everywhere.** Byte and packet counts span from zero to hundreds of millions, and a
  quarter of the features are zero in most flows. Scale-sensitive models (logistic regression,
  autoencoders) need log transforms; trees do not care.
- **Each family has its own fingerprint, often a lab one.** DoS and DDoS separate on packet sizes
  and timing; brute force on repeated small exchanges; web attacks and several others on the
  server's TCP window, which is a property of the victim machine rather than the attack (see
  `03_forward_2017`). Destination port separates almost every family, for the reason in `01a`.

## Limits

Single-feature separation ignores interactions, and it treats destination port as a number."""

TEXT["01c_intro"] = """# 01c · EDA: data quality

**Question.** How trustworthy are the rows and labels themselves?

Cleaning (`00_data_and_cleaning`) removed impossible values. This notebook looks at what cleaning
cannot fix: repeated flows, contradictory labels, and flows whose label does not match their
content."""

TEXT["01c_conflicts_md"] = """## Contradictory labels

The same feature vector should not be both benign and an attack. Below, every distinct vector
that carries more than one label, grouped by the labels and whether the conflict is within one
day or across days."""

TEXT["01c_empty_md"] = """## "Attacks" with no payload

An XSS or SQL injection attack has to send its payload. The table counts web-attack flows that
carry zero bytes in both directions and describes what they look like."""

TEXT["01c_body"] = """## What this shows

- **Most web-attack rows are not attacks.** The large majority of XSS and web brute-force flows
  carry no payload. They share one shape (three packets one way, one back, a few seconds, no FIN,
  answered by the victim server's 28,960-byte window). Only a few dozen XSS flows carry the
  actual attack. This is consistent with the "TCP appendix" defect described by Engelen, Rimmer &
  Joosen (2021): packets left over after a connection is closed are split into a new flow that
  inherits the attack label. It also explains `03_forward_2017`: the detector that "caught" 82%
  of unseen web attacks was recognizing these empty server replies.
- **Some attacks are mostly repeats.** SSH-Patator and PortScan rows are nearly half exact
  duplicates. Recall on those classes counts the same flow many times.
- **Labels contradict each other, across days and within them.** Most conflicting distinct flows
  are benign on one day and PortScan on Friday: short probe-like flows (the example above is one
  packet each way, 54 microseconds) that are indistinguishable from benign traffic in these
  features. By row count, benign vs DoS Hulk dominates, including conflicts within Wednesday itself. Removing them before a split (as the first cleaning did) uses test
  labels; the evaluation copy keeps them.

## Known defects from the literature

Engelen, Rimmer & Joosen (2021, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017
Case Study") report flow-construction errors in the CICFlowMeter tool, labelling errors, and
attempted attacks labelled as attacks. Findings above that match their description are stated as
consistent with it, not as independent confirmation.

## Consequences for evaluation

- Per-family recall for web attacks mostly measures empty connections, not attacks.
- Duplicates inflate both the training signal and the test counts for some families.
- Any cleaning that uses labels must happen inside the training split, never before it."""
