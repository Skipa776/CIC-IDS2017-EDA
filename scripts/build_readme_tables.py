#!/usr/bin/env python3
"""Write the README's result tables from reports/results/summary.json.

Replaces the text between <!-- results:start --> and <!-- results:end -->, so
every number in those tables traces to a results file (design-doc gate 5).
A recall is shown only if the actual test FPR stayed within the budget rule;
otherwise the row says so.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SUMMARY = ROOT / "reports" / "results" / "summary.json"
EXPLAIN = ROOT / "reports" / "results" / "explain_2017_B.json"
NAMES = {"lgbm": "LightGBM", "logreg": "Logistic regression", "iforest": "Isolation Forest",
         "autoencoder": "Autoencoder"}
ROWS = [
    ("2017_A", "2017: train Mon (benign only), test Wed"),
    ("2017_B", "2017: train Mon-Tue, test Thu"),
    ("2017_C", "2017: train Mon-Wed, test Fri"),
    ("2018_same_attack_web", "2018: same attack, later date (web, 23 Feb)"),
    ("2018_same_attack_infiltration", "2018: same attack, later date (infiltration, 1 Mar)"),
    ("2018_new_tool_dos", "2018: same family, new tool (DoS, 16 Feb)"),
    ("2018_new_tool_ddos", "2018: same family, new tool (DDoS, 21 Feb)"),
    ("2018_new_family_bot", "2018: new family (Bot, 2 Mar)"),
    ("2017_to_2018", "Train all 2017, test all 2018"),
]


def name(det):
    if det.endswith("+anomaly"):
        return f"Hybrid: {NAMES[det.split('+')[0]]} + anomaly model"
    return NAMES[det]


def fmt(x):
    """Never round up to 1: above 0.99, truncate (not round) to 4 decimals, or 5 above 0.9999."""
    if x <= 0.99:
        return f"{x:.3f}"
    digits = 5 if x > 0.9999 else 4
    return f"{int(x * 10**digits) / 10**digits:.{digits}f}"


def roc_table(summary):
    dets = ["lgbm", "logreg", "iforest", "autoencoder"]
    lines = ["| Test | " + " | ".join(NAMES[d] for d in dets) + " |",
             "| --- |" + " ---: |" * len(dets)]
    for fold, label in ROWS:
        f = summary["folds"].get(fold)
        if f is None:
            continue
        cells = []
        for d in dets:
            auc = f["detectors"].get(d, {}).get("roc_auc")
            cells.append(f"{fmt(auc['median'])} ({fmt(auc['min'])}-{fmt(auc['max'])})" if auc else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def caveats():
    out = {}
    if EXPLAIN.exists():
        ex = json.loads(EXPLAIN.read_text())
        art = ex["window_artifact"]
        without = ex["ablations"]["without Init_Win_bytes_backward"]["per_label_recall"]
        out["2017_B"] = (f"Rides on one victim-server TCP window value ({art['most_common_web_attack_value']:,}, "
                         f"carried by {art['web_attack_share_with_value']:.0%} of web-attack flows); without it, "
                         f"web brute-force recall is {without['Web Attack - Brute Force']:.0%}")
    return out


def headline(summary):
    notes = caveats()
    lines = ["| Test | Attack families in test | Best detector within budget | Recall at 1% (median, seed range) "
             "| Actual FPR | AP / no-skill | ROC AUC | Notes |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for fold, label in ROWS:
        f = summary["folds"].get(fold)
        if f is None:
            lines.append(f"| {label} | | not run | | | | | |")
            continue
        fams = ", ".join(f["unseen_test_families"]) + (" (unseen)" if f["unseen_test_families"] else "")
        if f["seen_test_families"]:
            fams += ("; " if fams else "") + ", ".join(f["seen_test_families"]) + " (seen)"
        ok = {d: v for d, v in f["detectors"].items() if not v["budget_violated"]["0.01"]}
        flags = [notes[fold]] if fold in notes else []
        if not ok:
            worst = max(v["budgets"]["0.01"]["fpr"]["max"] for v in f["detectors"].values())
            lines.append(f"| {label} | {fams} | none: every detector broke the budget (worst actual FPR "
                         f"{worst:.1%}) | not quotable | | | | {'; '.join(flags)} |")
            continue
        # ties go to the single model (a hybrid that gave one model the whole budget is that model)
        best = max(ok, key=lambda d: (ok[d]["budgets"]["0.01"]["recall"]["median"], "+" not in d))
        v = ok[best]
        if v.get("shortcut_flag"):
            feats = f["best_single_feature"]
            which = feats[0] if len(feats) == 1 else ", ".join(feats[:-1]) + " or " + feats[-1]
            flags.insert(0, f"shortcut: one feature ({which}) alone comes within 0.05 AP")
        copies = f["gates"]["exact_twin_attack_rows"]["max"] / f["n_attack"]
        if copies >= 0.01:
            flags.append(f"{copies:.0%} of test attack flows are exact copies of training flows")
        r, fpr = v["budgets"]["0.01"]["recall"], v["budgets"]["0.01"]["fpr"]
        ap = v["average_precision"]
        ap_text = f"{fmt(ap['median']) if ap else '-'} / {f['prevalence']:.3f}"
        auc = v.get("roc_auc")
        lines.append(f"| {label} | {fams} | {name(best)} | {r['median']:.1%} ({r['min']:.1%}-{r['max']:.1%}) "
                     f"| {fpr['min']:.2%}-{fpr['max']:.2%} | {ap_text} | {fmt(auc['median']) if auc else '-'} "
                     f"| {'; '.join(flags)} |")
    return "\n".join(lines)


def hybrid(summary):
    lines = ["| Hybrid | Beats best single model (criterion runs won) | Worst actual FPR | Families lost | Verdict |",
             "| --- | ---: | ---: | ---: | --- |"]
    for h, r in summary["success_criteria"].items():
        c1 = r["criterion_1_beats_best_single"]["runs"]
        lines.append(f"| {name(h)} | {sum(x['pass'] for x in c1)} of {len(c1)} "
                     f"| {r['criterion_2_fpr_at_most_1.5pct']['worst_fpr']:.2%} "
                     f"| {len(r['criterion_3_no_family_lost']['families_lost'])} "
                     f"| {'pass' if r['overall_pass'] else 'fail'} |")
    return "\n".join(lines)


def main():
    summary = json.loads(SUMMARY.read_text())
    if summary["incomplete_folds"]:
        raise SystemExit(f"Incomplete folds, not writing README tables: {summary['incomplete_folds']}")
    block = ("<!-- results:start -->\n<!-- generated by scripts/build_readme_tables.py from "
             "reports/results/summary.json; do not edit by hand -->\n\n"
             + headline(summary) + "\n\n**Hybrid detector against the success criteria fixed before the runs:**\n\n"
             + hybrid(summary)
             + "\n\n**ROC AUC of every detector** (median, seed range; '-' = not trained on that test). "
               "ROC AUC ignores how rare attacks are, so it reads high even when a detector is useless at a "
               "1% alert budget; below 0.5 means attacks look more normal than benign traffic to the model. "
               "Read it beside AP and recall above. Example: LightGBM ranks 2018 Bot flows above most benign "
               "traffic (ROC AUC near 0.9) yet catches almost none even at a 5% budget, because Bot never "
               "outscores the top few percent of benign flows.\n\n"
             + roc_table(summary) + "\n\n<!-- results:end -->")
    text = README.read_text()
    new, n = re.subn(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, text, flags=re.S)
    if n != 1:
        raise SystemExit("README must contain exactly one results block")
    README.write_text(new)
    print("README results block updated")


if __name__ == "__main__":
    main()
