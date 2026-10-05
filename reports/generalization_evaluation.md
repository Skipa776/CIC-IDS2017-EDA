# Controlled day-transfer evaluation

The models do not establish reliable generalization across all tested days. This comparison separates ranking from detection at validation-frozen thresholds.

Source: [generalization_evaluation.json](generalization_evaluation.json). All ranges below are minimum–maximum over seeds, not confidence intervals.

## Protocol

Training-only benign cap: 200,000. Scalers fit on training only. Validation/test prevalence is unchanged. Thresholds use validation benign scores only; ties are handled so empirical validation FPR stays within budget.

Friday: train Mon–Wed, validate Thu, test Fri. Wed+Thu: train Mon+Tue, validate Fri, test Wed+Thu (nonchronological). Forward Thursday: train Mon+Tue, validate Wed, test Thu. Random benchmark: 64/16/20 train/validation/test, before training downsampling.

## All candidates at the 1% validation FPR budget

| Test | Candidate | Test AP | Actual test FPR | Test recall |
|---|---|---:|---:|---:|
| random_split | lgbm20 | 0.9997–0.9997 | 0.0087–0.0101 | 0.9997–0.9998 |
| random_split | lgbm71 | 0.9998–0.9998 | 0.0097–0.0103 | 0.9999–0.9999 |
| random_split | logreg20 | 0.8747–0.8755 | 0.0098–0.0100 | 0.5734–0.5772 |
| random_split | logreg71 | 0.9553–0.9573 | 0.0098–0.0099 | 0.7473–0.7520 |
| random_split | lgbm71_no_network | 0.9991–0.9991 | 0.0092–0.0099 | 0.9964–0.9968 |
| test_friday | lgbm20 | 0.7524–0.8684 | 0.0096–0.0099 | 0.3246–0.5954 |
| test_friday | lgbm71 | 0.7652–0.8179 | 0.0104–0.0108 | 0.2215–0.2760 |
| test_friday | logreg20 | 0.7272–0.7275 | 0.0122–0.0123 | 0.3863–0.3944 |
| test_friday | logreg71 | 0.8098–0.8121 | 0.0102–0.0104 | 0.3855–0.3865 |
| test_friday | lgbm71_no_network | 0.7762–0.8226 | 0.0055–0.0099 | 0.3220–0.3735 |
| test_wed_thu | lgbm20 | 0.3551–0.4273 | 0.0099–0.0119 | 0.0128–0.0128 |
| test_wed_thu | lgbm71 | 0.2062–0.3587 | 0.0028–0.0049 | 0.0125–0.0129 |
| test_wed_thu | logreg20 | 0.1241–0.1249 | 0.0259–0.0264 | 0.0017–0.0020 |
| test_wed_thu | logreg71 | 0.1196–0.1199 | 0.0113–0.0117 | 0.0045–0.0046 |
| test_wed_thu | lgbm71_no_network | 0.2078–0.2401 | 0.0087–0.0117 | 0.0030–0.0092 |
| forward_thursday | lgbm20 | 0.2593–0.2949 | 0.0102–0.0114 | 0.8343–0.8343 |
| forward_thursday | lgbm71 | 0.7005–0.7585 | 0.0055–0.0132 | 0.8330–0.8375 |
| forward_thursday | logreg20 | 0.0041–0.0043 | 0.0422–0.0425 | 0.0032–0.0046 |
| forward_thursday | logreg71 | 0.0235–0.0260 | 0.0092–0.0098 | 0.0688–0.0688 |
| forward_thursday | lgbm71_no_network | 0.0682–0.1581 | 0.0080–0.0085 | 0.0289–0.4029 |

## Selection without outer-test tuning

Candidates are selected by validation macro attack recall at the 1% validation budget. The selection itself can fail to transfer: validation and test contain different attack labels.

| Test | Seed | Validation-selected candidate | Actual test FPR | Test recall |
|---|---:|---|---:|---:|
| random_split | 42 | lgbm71 | 0.0099 | 0.9999 |
| test_friday | 42 | lgbm71 | 0.0108 | 0.2331 |
| test_wed_thu | 42 | logreg20 | 0.0259 | 0.0020 |
| forward_thursday | 42 | lgbm71_no_network | 0.0082 | 0.0289 |
| random_split | 43 | lgbm71 | 0.0097 | 0.9999 |
| test_friday | 43 | lgbm71 | 0.0107 | 0.2760 |
| test_wed_thu | 43 | logreg20 | 0.0264 | 0.0017 |
| forward_thursday | 43 | lgbm71 | 0.0078 | 0.8330 |
| random_split | 44 | lgbm71 | 0.0103 | 0.9999 |
| test_friday | 44 | lgbm71 | 0.0104 | 0.2215 |
| test_wed_thu | 44 | logreg20 | 0.0260 | 0.0018 |
| forward_thursday | 44 | lgbm71 | 0.0055 | 0.8375 |

## Scope of the conclusion

- Day and attack family change together; no causal isolation.
- Only file provenance is available, not session IDs or timestamps.
- Seed ranges measure fit/downsampling variation, not population confidence intervals.
- Outer holdouts were previously inspected; final confirmation needs fresh captures.
- No operational minimum recall has been agreed; no model is certified for deployment.

A candidate with better recall but a test FPR above 1% has exceeded the comparison budget. A higher AP alone does not remedy this. No production artifacts were replaced.
