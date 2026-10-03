# Leakage investigation and corrected holdouts

## Confirmed defects

- 1,854 legacy test flows have an exact 20-input twin in training (0.37%); 47 are attacks.
- 84.63% of legacy attack test flows share a fixed behavior profile with training. This is dependence under the declared proxy, not proof of shared sessions.
- Legacy cleaning removes contradictory labels globally before splitting. The new evaluation retains conflicting labels and repeated flows, grouping them across partitions.
- The feature-importance notebook fits on labels before its test split. The primary new model uses all 71 numeric inputs, avoiding that supervised feature selection.

## New protocols

Global label-blind behavior groups are assigned to train/validation/test (60/20/20 groups), including matching profiles across different files. Actual row proportions and prevalence may differ.

A separate purged block test takes the tail of every raw CSV, reserves intervening gaps, and removes test/validation profiles from earlier partitions. CSV order is not verified time order. Quarter-octave and eighth-octave profile splits assess grouping sensitivity; no rule is selected by test score.

Whole-day Friday, Wed+Thu and forward-Thursday tests are rerun on this raw-derived population. The day boundary is their separation unit: profiles are NOT purged across days, because similar behaviour on other days is what a transfer test measures. Profile overlap is still reported. Wed+Thu remains explicitly nonchronological.

Scalers and models fit on training only. Thresholds use validation benign flows only. Only training benign flows are capped. The original 20-feature model is a diagnostic; the 71-feature model is the primary baseline. All captures have already been explored, so these remain development evaluations.

## Results at the 1% validation FPR budget

| Protocol | Seed | Model | Test AP | Test prevalence | Actual test FPR | Recall |
|---|---:|---|---:|---:|---:|---:|
| grouped_quarter_octave | 42 | lgbm71 | 0.9996 | 0.1985 | 0.0123 | 0.9999 |
| grouped_quarter_octave | 42 | lgbm20 | 0.9994 | 0.1985 | 0.0136 | 0.9990 |
| grouped_eighth_octave | 42 | lgbm71 | 0.9994 | 0.2275 | 0.0096 | 0.9999 |
| grouped_eighth_octave | 42 | lgbm20 | 0.9992 | 0.2275 | 0.0090 | 0.9997 |
| purged_file_blocks | 42 | lgbm71 | 0.9908 | 0.0704 | 0.0103 | 0.9760 |
| purged_file_blocks | 42 | lgbm20 | 0.9869 | 0.0704 | 0.0099 | 0.9550 |
| test_friday | 42 | lgbm71 | 0.7715 | 0.4110 | 0.0125 | 0.4386 |
| test_friday | 42 | lgbm20 | 0.7694 | 0.4110 | 0.0090 | 0.2964 |
| test_wed_thu | 42 | lgbm71 | 0.5438 | 0.2208 | 0.0085 | 0.0132 |
| test_wed_thu | 42 | lgbm20 | 0.5267 | 0.2208 | 0.0117 | 0.0094 |
| forward_thursday | 42 | lgbm71 | 0.7622 | 0.0048 | 0.0146 | 0.8615 |
| forward_thursday | 42 | lgbm20 | 0.1976 | 0.0048 | 0.0124 | 0.8204 |
| grouped_quarter_octave | 43 | lgbm71 | 0.9996 | 0.2053 | 0.0119 | 0.9999 |
| grouped_quarter_octave | 43 | lgbm20 | 0.9994 | 0.2053 | 0.0043 | 0.9996 |
| grouped_eighth_octave | 43 | lgbm71 | 0.9995 | 0.1848 | 0.0089 | 0.9999 |
| grouped_eighth_octave | 43 | lgbm20 | 0.9991 | 0.1848 | 0.0132 | 0.9997 |
| purged_file_blocks | 43 | lgbm71 | 0.9908 | 0.0704 | 0.0093 | 0.9846 |
| purged_file_blocks | 43 | lgbm20 | 0.9905 | 0.0704 | 0.0104 | 0.9661 |
| test_friday | 43 | lgbm71 | 0.7707 | 0.4110 | 0.0122 | 0.2162 |
| test_friday | 43 | lgbm20 | 0.6131 | 0.4110 | 0.0102 | 0.0776 |
| test_wed_thu | 43 | lgbm71 | 0.2186 | 0.2208 | 0.0120 | 0.0099 |
| test_wed_thu | 43 | lgbm20 | 0.3044 | 0.2208 | 0.0113 | 0.0099 |
| forward_thursday | 43 | lgbm71 | 0.6507 | 0.0048 | 0.0119 | 0.8172 |
| forward_thursday | 43 | lgbm20 | 0.1929 | 0.0048 | 0.0128 | 0.8204 |
| grouped_quarter_octave | 44 | lgbm71 | 0.9997 | 0.1889 | 0.0128 | 0.9999 |
| grouped_quarter_octave | 44 | lgbm20 | 0.9995 | 0.1889 | 0.0115 | 0.9998 |
| grouped_eighth_octave | 44 | lgbm71 | 0.9997 | 0.1838 | 0.0083 | 0.9999 |
| grouped_eighth_octave | 44 | lgbm20 | 0.9994 | 0.1838 | 0.0092 | 0.9999 |
| purged_file_blocks | 44 | lgbm71 | 0.9930 | 0.0704 | 0.0117 | 0.9869 |
| purged_file_blocks | 44 | lgbm20 | 0.9880 | 0.0704 | 0.0101 | 0.9653 |
| test_friday | 44 | lgbm71 | 0.7883 | 0.4110 | 0.0131 | 0.3373 |
| test_friday | 44 | lgbm20 | 0.6859 | 0.4110 | 0.0100 | 0.0125 |
| test_wed_thu | 44 | lgbm71 | 0.2268 | 0.2208 | 0.0071 | 0.0096 |
| test_wed_thu | 44 | lgbm20 | 0.5944 | 0.2208 | 0.0118 | 0.0093 |
| forward_thursday | 44 | lgbm71 | 0.6684 | 0.0048 | 0.0082 | 0.8168 |
| forward_thursday | 44 | lgbm20 | 0.2314 | 0.0048 | 0.0121 | 0.8204 |

## Remaining limits

- Behavior bins do not identify real sessions; near neighbors across bin boundaries can remain.
- File-order blocks cannot be called chronological without timestamps.
- Same-file holdouts retain capture/campaign context; whole-day and cross-year tests remain essential.
- Already explored captures cannot provide a fresh confirmatory test.
- Seed variation is not a population confidence interval; no operational recall target is agreed.

Shuffled-training-label control on the quarter-octave grouped split: test AP 0.2065; prevalence 0.1985. This is a negative control for learning from corrupted training targets; near-baseline AP does not rule out target-derived features.

Source: [leakage_audit.json](leakage_audit.json). Historical metrics remain archived for auditability and are not promoted as generalization evidence.
