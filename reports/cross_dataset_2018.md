# Combining detectors, and testing on CSE-CIC-IDS2018

**Question.** A supervised model catches attack types it has seen. Isolation Forest can flag
unusual traffic it has never seen. Does combining them give a stronger detector? And does
anything trained on CIC-IDS2017 still work on a newer dataset from a different network?

**Script:** `scripts/combined_model.py`. Results: `reports/combined_model.json`.
Dataset build counts: `reports/dataset_2018_manifest.json`.

## Setup

**Detectors.**

| Detector | What it is |
|---|---|
| Logistic regression | 71 features, as in `cicids2017_eda.ipynb` |
| LightGBM | Layer 1, 20 features, as in `train_models.py` |
| Isolation Forest | 71 features, trained on benign flows only, 1,000 trees |
| Combined | Alerts if either the supervised model or Isolation Forest alerts |

**Alert budget.** Every detector may alert on at most 1% of benign flows. A combined detector
splits that 1% between its two models.

**Choosing the split without peeking.** The split is chosen by leave-one-day-out inside the
training data. Each training weekday with attacks is held out in turn, the models are fit on the
other days, and the split that catches the most held-out attacks wins. The test set is never used.

**Two kinds of thresholds.**
- *Deployed:* set on held-out benign flows from the training days. This is what you could do in
  practice.
- *Equal budget:* set on the test set's own benign flows, so every detector alerts on exactly 1% of
  test benign traffic. Optimistic, but a fair comparison between detectors.

**CSE-CIC-IDS2018.**
- Same institute and the same flow tool (CICFlowMeter), recorded a year later on a different
  network (AWS). The 2018 columns map one-to-one onto the 2017 features (`src/data/cicids2018.py`),
  and the data goes through the same cleaning code as 2017.
- To fit in memory, 10% of benign flows are sampled and every attack flow is kept. PR-AUC weights
  benign rows by 10 so it reflects the real attack share (10.3%).
- After cleaning: 2,514,216 rows. Cleaning removed 1,557,465 exact duplicates and 14,351
  contradictory-label rows. FTP-BruteForce (39 rows) and DoS-SlowHTTPTest (43 rows) almost
  disappear: nearly all of their flows were duplicates or had identical features under conflicting
  labels.

## Results

**Recall at a 1% false positive budget (equal budget):**

| Detector | 2017: train Mon+Tue+Fri, test Wed+Thu | 2018: train all of 2017, test 2018 |
|---|---:|---:|
| Logistic regression | 0.493 | 0.024 |
| LightGBM | 0.008 | 0.004 |
| Isolation Forest | 0.255 | 0.006 |
| Combined | same as logistic regression (split chose 100% supervised) | same as logistic regression |

**PR-AUC:**

| Detector | 2017 cross-day (no-skill 0.197) | 2018 (no-skill 0.103) |
|---|---:|---:|
| Logistic regression | 0.711 | 0.292 |
| LightGBM | 0.459 | 0.395 |
| Isolation Forest | 0.774 | 0.102 |

**Deployed thresholds drift.** Set on training-day benign traffic, the "1%" thresholds flagged
5.5% (logistic regression) and 6.1% (LightGBM) of benign flows on Wed+Thu, and 3.8% (LightGBM)
on 2018. Normal traffic changes between days and networks, so thresholds need recalibration on
recent benign traffic.

## What this shows

**1. The combination was not chosen.** Leave-one-day-out picked "100% supervised" on every held-out
day except Thursday, so the combined detector equals logistic regression. An earlier informal
test that picked the split after looking at the test days found a gain on Wed+Thu. That gain is
not reachable without peeking, so it should not be claimed.

**2. Isolation Forest is unstable at a 1% budget.** Rerunning it on the 2017 cross-day split with
different random seeds gave recall between 0.227 and 0.545 with 200 trees, and between 0.265 and
0.355 with 1,000 trees (a separate check, not saved in the repo). Single-run Isolation Forest
numbers at low alert budgets, including the ones in `notebooks/single_day_eda.ipynb`, should be
read with that spread in mind.

**3. Nothing transfers to 2018, including attack types the models were trained on.** Logistic
regression caught 2017's SSH brute force (same-day recall 0.996 in `single_day_eda.ipynb`), but
2018's SSH-Bruteforce at 0.000. Isolation Forest scored at no-skill (PR-AUC 0.102 vs. 0.103). The same attacks look
different on the new network (medians computed from the two cleaned parquets):

| Median | 2017 SSH-Patator | 2018 SSH-Bruteforce | 2017 DoS Hulk | 2018 DoS Hulk | 2017 benign | 2018 benign |
|---|---:|---:|---:|---:|---:|---:|
| Flow Duration (µs) | 12,029,788 | 374,609 | 86,419,094 | 50,949 | 40,277 | 149,420 |
| Total Backward Packets | 32 | 22 | 6 | 0 | 2 | 2 |
| Average Packet Size | 89.7 | 104.8 | 918.2 | 0.0 | 78.2 | 87.0 |
| Destination Port | 22 | 22 | 80 | 80 | 80 | 443 |

2018's SSH brute-force flows are about 32 times shorter. Its DoS Hulk flows get no reply packets.
Its benign traffic shifts from port 80 to 443. The models learned what attacks looked like on the
2017 network, with the 2017 tools, not what an attack is.

## Recommendation

- A detector trained on one network should not be trusted on another without retraining or at
  least recalibrating on that network's traffic.
- Choose thresholds on recent benign traffic from the network being monitored, and re-check them
  regularly.
- Report Isolation Forest results at low alert budgets as a range over seeds, and use many trees.
- The combination idea is still reasonable, but on this data it did not earn its place under an
  honest selection procedure. It would need more varied training days, or a better way to
  simulate unseen attacks, to show a real benefit.

## Limits

- One seed per detector in `combined_model.json` (Isolation Forest variance above).
- 2018 labels have known problems (for example the Infilteration class). Some of the failure may
  be label noise rather than model error.
- Both datasets were made by the same lab with the same flow tool, so this is a mild test of
  transfer. A dataset from a different tool (for example the NetFlow-based NF-UNSW-NB15-v2) would
  be a harder one, but would need retraining on a different feature set.
