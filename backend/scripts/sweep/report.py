"""Phase 7: Markdown report combining the regression A/B, OFAT sweeps,
and the classifier-only sweep.

Reads JSON outputs from results/ and emits results/REPORT.md.

Run from backend/:
    python -m scripts.sweep.report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_RESULTS = Path(__file__).resolve().parent / "results"


def _load(name):
    p = _RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _ab_section() -> list[str]:
    pre = _load("pre_regression_agg.json")
    cur = _load("current_prod_agg.json")
    if not pre or not cur:
        return ["## A/B regression triage\n\n_(no data — run `run_regression.py` first)_\n"]

    lines = ["## A/B regression triage", ""]
    lines.append(f"Dataset: {pre['metrics']['n_visits']} clip-pinned visits "
                 f"with {pre['metrics']['total_gt_nabs']} GT NABs, "
                 f"{pre['metrics']['total_gt_real']} GT real-species labels.")
    lines.append("")
    lines.append("| Metric | PRE_REGRESSION | CURRENT_PROD | Δ |")
    lines.append("|---|---:|---:|---:|")

    def fmt_pct(x):
        return f"{100 * x:.1f}%"

    rows = [
        ("FP leak rate (↓ better)", "fp_leak_rate"),
        ("TP preservation (real, ↑ better)", "tp_preservation_real"),
        ("TP preservation (any bird, ↑ better)", "tp_preservation_any"),
        ("Classifier hit rate (↑ better)", "classifier_hit_rate"),
    ]
    for label, key in rows:
        p, c = pre["rates"][key], cur["rates"][key]
        delta = c - p
        sign = "+" if delta > 0 else ""
        lines.append(f"| {label} | {fmt_pct(p)} | {fmt_pct(c)} | {sign}{fmt_pct(delta)} |")

    lines.append("")
    # Headline interpretation
    fp_better = cur["rates"]["fp_leak_rate"] < pre["rates"]["fp_leak_rate"]
    tp_better = cur["rates"]["tp_preservation_any"] >= pre["rates"]["tp_preservation_any"]
    if fp_better and tp_better:
        verdict = ("**Headline:** Recent changes IMPROVED feed quality on this dataset — "
                   "CURRENT_PROD has lower FP leak with equal-or-better TP preservation. "
                   "The user's anecdotal regression is likely camera-side (CBR/bitrate/IR/exposure) "
                   "or content-side (time-of-year bird activity), NOT a config regression.")
    elif fp_better and not tp_better:
        verdict = ("**Headline:** Recent changes IMPROVED false-positive suppression but at a "
                   "real-bird-recall cost. Worth tuning the scene mask / classifier thresholds.")
    elif not fp_better and tp_better:
        verdict = ("**Headline:** Recent changes worsened FP suppression. Look at scene-mask "
                   "and classifier-rejection thresholds first.")
    else:
        verdict = ("**Headline:** Recent changes regressed on both axes. Significant rollback "
                   "is justified — start with the most-suspect knobs in the OFAT.")
    lines.append(verdict)
    lines.append("")

    # Per-visit detail table.
    lines.append("### Per-visit breakdown")
    lines.append("")
    lines.append("| Visit | GT NABs | GT real | PRE detections | CUR detections | PRE FP leak | CUR FP leak |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    pre_by_v = {d["visit_id"]: d for d in pre["per_visit"]}
    cur_by_v = {d["visit_id"]: d for d in cur["per_visit"]}
    for vid in sorted(set(pre_by_v) | set(cur_by_v)):
        p = pre_by_v.get(vid, {})
        c = cur_by_v.get(vid, {})
        lines.append(f"| {vid} | {p.get('n_gt_nabs', '-')} | {p.get('n_gt_real', '-')} | "
                     f"{p.get('n_replay_detections', '-')} | {c.get('n_replay_detections', '-')} | "
                     f"{p.get('fp_leaked', '-')} | {c.get('fp_leaked', '-')} |")
    return lines


def _ofat_section() -> list[str]:
    files = sorted(_RESULTS.glob("ofat_*.json"))
    if not files:
        return ["## OFAT optimization sweep\n\n_(no data — run `run_ofat.py`)_\n"]

    lines = ["## OFAT optimization sweep", ""]
    lines.append("For each knob, every other knob is held at its CURRENT_PROD value.")
    lines.append("")
    for f in files:
        data = json.loads(f.read_text())
        knob = data["knob"]
        cur = data["current_prod_value"]
        lines.append(f"### {knob}  (current value: `{cur}`)")
        lines.append("")
        lines.append("| value | FP leak | TP-real preserved | TP-any preserved | Cls hit rate | Novel dets |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        best_value = None
        best_score = -1
        for v in data["values"]:
            # Composite "score": tp_preservation_any minus fp_leak_rate.
            comp = v["tp_preservation_any"] - v["fp_leak_rate"]
            if comp > best_score:
                best_score = comp
                best_value = v["value"]
            mark = " ★" if v["value"] == cur else ""
            lines.append(f"| {v['value']}{mark} | {100*v['fp_leak_rate']:.1f}% | "
                         f"{100*v['tp_preservation_real']:.1f}% | "
                         f"{100*v['tp_preservation_any']:.1f}% | "
                         f"{100*v['classifier_hit_rate']:.1f}% | {v['novel_detections']} |")
        lines.append(f"\n**Best value (max TP-any − FP-leak):** `{best_value}`")
        if best_value != cur:
            lines.append(f"  → Differs from current production (`{cur}`). Consider updating.")
        lines.append("")
    return lines


def _classifier_section() -> list[str]:
    data = _load("classifier_in_range_sweep.json")
    if not data:
        return ["## Classifier IN_RANGE_THRESHOLD sweep\n\n_(no data — run `run_classifier_sweep.py`)_\n"]

    lines = ["## Classifier IN_RANGE_THRESHOLD sweep", ""]
    lines.append("Sweep run on every saved labeled crop (no clip replay needed).")
    lines.append("")
    lines.append("| threshold | NAB rejection ↑ | real top-1 ↑ | real top-5 | real rejection ↓ |")
    lines.append("|---:|---:|---:|---:|---:|")
    best = max(data, key=lambda r: r["nab_rejection_rate"] - r["real_rejection_rate"])
    for r in data:
        mark = " ★" if r is best else ""
        lines.append(f"| {r['threshold']:.3f}{mark} | "
                     f"{r['nab_rejected']}/{r['n_nab']} ({100*r['nab_rejection_rate']:.0f}%) | "
                     f"{100*r['real_top1_acc']:.1f}% | "
                     f"{100*r['real_top5_acc']:.1f}% | "
                     f"{r['real_rejected']}/{r['n_real']} ({100*r['real_rejection_rate']:.0f}%) |")
    lines.append(f"\n**Best (max NAB-rej − real-rej):** threshold = `{best['threshold']}`")
    lines.append("")
    return lines


def main():
    lines = ["# Pipeline tuning sweep — results", "",
             "_Generated by `scripts/sweep/report.py`._", ""]
    lines += _ab_section()
    lines += [""]
    lines += _ofat_section()
    lines += [""]
    lines += _classifier_section()
    out = _RESULTS / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
