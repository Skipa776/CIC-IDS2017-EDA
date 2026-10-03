#!/usr/bin/env python3
"""Summarize reports/results/<fold>.json into reports/results/summary.json.

Ranges are min-max over seeds (not confidence intervals). Also scores the
hybrid against the success criteria fixed in the design doc before any run:
  1. Higher recall than the best single model on every hybrid fold in
     CRITERION_FOLDS, in every seed.
     (Deviation, recorded: the doc said "two of three 2017 folds", but fold
     2017_A trains on benign-only Monday and has no supervised model, hence no
     hybrid. The criterion uses both 2017 folds that have one.)
  2. Actual test FPR at most 1.5% on every fold and seed.
  3. No attack family drops to zero recall that a single model caught.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results"
PRIMARY = "0.01"
CRITERION_FOLDS = ["2017_B", "2017_C", "2018_new_family_bot"]
MAX_FPR = 0.015
# A recall is only quotable if the actual test FPR stayed near its budget.
BUDGET_TOLERANCE = 1.5
SHORTCUT_MARGIN = 0.05  # a single feature within this AP of the model = shortcut flag


def span(values):
    values = [v for v in values if v is not None]
    return {"min": float(min(values)), "median": float(np.median(values)), "max": float(max(values))} if values else None


def canonical(data):
    """Hybrids pick their anomaly partner per seed on validation; key them as '<sup>+anomaly'."""
    for run in data["runs"]:
        for key in [k for k in run["detectors"] if "+" in k and not k.endswith("+anomaly")]:
            sup, partner = key.split("+")
            run["detectors"][f"{sup}+anomaly"] = run["detectors"].pop(key)
            run["detectors"][f"{sup}+anomaly"]["partner"] = partner
            choice = run.get("hybrid_choices", {}).pop(key, None)
            if choice is not None:
                run["hybrid_choices"][f"{sup}+anomaly"] = choice
    return data


def summarize_fold(data):
    runs = data["runs"]
    detectors = list(runs[0]["detectors"])
    out = {"question": data["fold"]["question"], "seeds": [r["seed"] for r in runs],
           "unseen_test_families": runs[0]["unseen_test_families"],
           "seen_test_families": runs[0]["seen_test_families"],
           "prevalence": runs[0]["detectors"][detectors[0]]["prevalence"],
           "n_attack": runs[0]["detectors"][detectors[0]]["n_attack"],
           "n_test": runs[0]["detectors"][detectors[0]]["n_test"],
           "gates": {k: span([r["gates"][k] for r in runs]) for k in
                     ["exact_twin_test_fraction", "exact_twin_attack_rows", "shuffled_label_ap", "best_single_feature_ap"]},
           "best_single_feature": sorted({data["features"][r["gates"]["best_single_feature_index"]] for r in runs}),
           "detectors": {}}
    for name in detectors:
        reps = [r["detectors"][name] for r in runs]
        families = sorted(reps[0]["budgets"][PRIMARY]["per_family"])
        out["detectors"][name] = {
            "average_precision": span([r["average_precision"] for r in reps]),
            "budgets": {b: {"recall": span([r["budgets"][b]["recall"] for r in reps]),
                            "fpr": span([r["budgets"][b]["fpr"] for r in reps])}
                        for b in reps[0]["budgets"]},
            "per_family_recall_at_1pct": {f: span([r["budgets"][PRIMARY]["per_family"][f]["recall"] for r in reps])
                                          for f in families},
            # recall at a budget is not quotable when the frozen threshold let through far more benign
            "budget_violated": {b: max(r["budgets"][b]["fpr"] for r in reps) > BUDGET_TOLERANCE * float(b)
                                for b in reps[0]["budgets"]},
        }
        aps = [r["average_precision"] for r in reps if r["average_precision"] is not None]
        if aps:
            out["detectors"][name]["shortcut_flag"] = bool(
                max(r["gates"]["best_single_feature_ap"] for r in runs) >= np.median(aps) - SHORTCUT_MARGIN)
    if "hybrid_choices" in runs[0] and runs[0]["hybrid_choices"]:
        out["hybrid_shares_chosen"] = {h: [r["hybrid_choices"][h]["share_supervised"] for r in runs]
                                       for h in runs[0]["hybrid_choices"]}
        out["hybrid_partner_chosen"] = {h: [r["hybrid_choices"][h]["anomaly"] for r in runs]
                                        for h in runs[0]["hybrid_choices"]}
    return out


def criteria(folds):
    report = {}
    for hybrid in ["lgbm+anomaly", "logreg+anomaly"]:
        c1, c2, c3, seen = [], [], [], False
        for name, data in folds.items():
            for run in data["runs"]:
                if hybrid not in run["detectors"]:
                    continue
                seen = True
                singles = {n: d for n, d in run["detectors"].items() if "+" not in n}
                h = run["detectors"][hybrid]["budgets"][PRIMARY]
                best_single = max(d["budgets"][PRIMARY]["recall"] for d in singles.values())
                if name in CRITERION_FOLDS:
                    c1.append({"fold": name, "seed": run["seed"], "hybrid": h["recall"],
                               "best_single": best_single, "pass": h["recall"] > best_single})
                c2.append({"fold": name, "seed": run["seed"], "fpr": h["fpr"], "pass": h["fpr"] <= MAX_FPR})
                for fam, v in h["per_family"].items():
                    caught = any(d["budgets"][PRIMARY]["per_family"][fam]["recall"] > 0 for d in singles.values())
                    if caught and v["recall"] == 0:
                        c3.append({"fold": name, "seed": run["seed"], "family": fam})
        if not seen:
            continue
        folds_covered = {r["fold"] for r in c1}
        report[hybrid] = {
            "criterion_1_beats_best_single": {"pass": bool(c1) and all(r["pass"] for r in c1)
                                              and folds_covered == set(CRITERION_FOLDS),
                                              "folds_covered": sorted(folds_covered), "runs": c1},
            "criterion_2_fpr_at_most_1.5pct": {"pass": all(r["pass"] for r in c2),
                                                "worst_fpr": max(r["fpr"] for r in c2)},
            "criterion_3_no_family_lost": {"pass": not c3, "families_lost": c3},
        }
        report[hybrid]["overall_pass"] = all(v["pass"] for k, v in report[hybrid].items() if k.startswith("criterion"))
    return report


def main():
    folds = {p.stem: canonical(json.loads(p.read_text())) for p in sorted(RESULTS.glob("20*.json"))}
    incomplete = [n for n, d in folds.items() if not d.get("complete")]
    summary = {
        "note": "Ranges are min-max over seeds, not confidence intervals. Thresholds frozen on validation benign.",
        "incomplete_folds": incomplete,
        "folds": {n: summarize_fold(d) for n, d in folds.items()},
    "rules": {"budget_tolerance": BUDGET_TOLERANCE, "shortcut_margin": SHORTCUT_MARGIN,
              "budget_violated": "max actual test FPR over seeds > tolerance x budget; recall then not quotable",
              "shortcut_flag": "best single feature AP within margin of the detector's median AP"},
        "success_criteria": criteria(folds),
        "criterion_folds": CRITERION_FOLDS,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, f in summary["folds"].items():
        row = ", ".join(f"{d} {v['budgets'][PRIMARY]['recall']['median']:.3f}"
                        f"{'!' if v['budget_violated'][PRIMARY] else ''}{'*' if v.get('shortcut_flag') else ''}"
                        for d, v in f["detectors"].items())
        print(f"{name:<32} {row}")
    for h, r in summary["success_criteria"].items():
        print(f"{h}: overall {'PASS' if r['overall_pass'] else 'FAIL'} "
              f"(c1 {r['criterion_1_beats_best_single']['pass']}, c2 {r['criterion_2_fpr_at_most_1.5pct']['pass']}, "
              f"c3 {r['criterion_3_no_family_lost']['pass']})")
    print("! = actual FPR broke the budget (recall not quotable); * = single-feature shortcut flag")
    if incomplete:
        print("Incomplete:", incomplete)


if __name__ == "__main__":
    main()
