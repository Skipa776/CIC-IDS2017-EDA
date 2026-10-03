# Evaluation evidence audit

The existing results establish strong within-capture discrimination and weak transfer of the
20-feature detector. They do not establish a generally effective intrusion detector, nor isolate
the causal effects of attack novelty, benign traffic drift or capture artifacts.

## Metric and run provenance

| Evidence | Data / model | Evaluation | Threshold source | Interpretation |
|---|---|---|---|---|
| `models/model_metadata.json` | CIC-IDS2017 clean v2; LightGBM, 20 features, seed 42 | Random 80/20; Friday holdout; Wed+Thu holdout | Fixed 0.5 | Stored model benchmark and day stress tests |
| `reports/crossday_threshold_analysis.json` | Same v2 data and 20-feature training recipe | Same original split definitions | Test labels / test ROC curve | Retrospective diagnostic, not a deployable threshold |
| `reports/eda_baseline_validation_v2.json` | v2; logistic regression, 71 features | Experiments use different populations; random baseline has 68% attacks | Default model decisions | Cleaning/protocol checks; not comparable to the natural-prevalence LightGBM random test |
| `reports/combined_model.json` | v2; logistic regression 71, LightGBM 20, Isolation Forest 71 | Mon+Tue+Fri to Wed+Thu; all 2017 to 2018 | Both training-pool held-out benign and test-benign thresholds | Distinguishes deployed threshold drift from retrospective equal-budget results |
| `reports/cross_year_train_2017_test_2018.json`, reverse-direction JSON | Harmonized 2017/2018 data; 71-feature models and feature/normalization variants | Cross-year transfer in both directions | Test-informed fixed-FPR diagnostics | Additional exploratory transfer evidence; not independent confirmation of a chosen fix |
| `reports/generalization_evaluation.json` | v2; five candidates, seeds 42/43/44 | Identical candidate partitions; day validation plus outer test; forward Thursday and random benchmark | Validation benign only; separately labeled retrospective results | Controlled exploratory comparison with realized test FPR, per-day/family counts and score drift |

All legacy fields named `pr_auc` above use `average_precision_score`, i.e. non-interpolated
average precision. For random ranking, attack prevalence is the conventional no-skill reference.
Differences in prevalence and attack mix affect AP; raw AP differences across test populations
are not a pure measure of model deterioration.

## Corrections to interpretation

- Friday AP 0.824 is above prevalence 0.358; Wed+Thu AP 0.466 is above prevalence 0.197.
  Poor recall at threshold 0.5 is distinct from poor ranking. Both matter for detection.
- The original 1% test-FPR diagnostic detects 13.76% of Friday attacks and 0.0271% of Wed+Thu
  attacks. The latter is small but not exactly zero: 53 DoS Hulk flows are detected.
- A test-selected threshold uses information that is unavailable before deployment. It cannot
  be described as a validated operational threshold. Frozen thresholds must report actual test FPR.
- Attack labels and capture days are tied together. Day holdouts also change benign traffic and
  capture conditions; they cannot separate these causes. The test labels themselves are not a
  general representation of arbitrary future or zero-day attacks.
- Exact deduplication and shuffled-label checks are useful controls, not proofs of independence
  or absence of leakage. Shuffling training targets is a negative control for corrupted-target
  learning; even a target-derived feature can collapse under that control. The subsequent raw-file
  audit in `leakage_audit.md` directly measures input/profile overlap and corrects global label filtering.
- The clean export retains `Meta_source`, not timestamps or session identifiers. Session grouping,
  within-day chronology and block-bootstrap intervals are not supportable from this export alone.
- Layer 2 macro F1 0.910 evaluates all true attack flows, bypassing Layer 1. It is not the gated
  service's end-to-end score.

## Why earlier reports differ

The original Wed+Thu 20-feature LightGBM AP is 0.4658; the combined-model report records a
nearby but distinct fit around 0.459. Its retrospective 1% recall also differs. These are separate
runs: the combined-model script uses a different training-row ordering and a different
test-benign quantile/strict-inequality threshold rule. No exact numerical equivalence is claimed.
The old reports do not retain partition hashes or complete library versions, so the precise
contribution of each implementation/environment difference cannot be reconstructed reliably.

The new comparison does not overwrite or retroactively relabel historical model results. Its
whole-day validation changes the training population: in the Wed+Thu stress test, reserving
Friday leaves only Mon+Tue for training. This sacrifices attack diversity as well as training
rows. New candidate differences are controlled within that protocol; comparisons with the old
Mon+Tue+Fri fit cannot be attributed solely to the model or threshold change.

## Promotion gate

Use 1% validation benign FPR as a research comparison point, with 0.1% and 5% diagnostics.
No minimum acceptable recall or SOC false-alert budget has been agreed, so no candidate is
certified or substituted for the API model. Before a deployment claim, specify minimum recall
and important-family requirements, select using development captures, then freeze the model
and threshold for independent future captures. Repeat known families across days to investigate
temporal transfer separately from unseen-family detection. Reusing these explored holdouts
would provide further development evidence, not an untouched confirmation set.
