"""Print a tight summary of whatever sweep results are sitting in results/.

Useful for in-progress checks without having to regenerate the full Markdown
report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_RESULTS = Path(__file__).resolve().parent / "results"


def _load(name):
    p = _RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    pre = _load("pre_regression_agg.json")
    cur = _load("current_prod_agg.json")
    if pre or cur:
        print("=" * 50)
        print("A/B regression triage")
        print("=" * 50)
        for label, d in [("PRE_REGRESSION", pre), ("CURRENT_PROD", cur)]:
            if not d:
                print(f"  {label}: (no data)")
                continue
            r = d["rates"]
            m = d["metrics"]
            print(f"  {label} (n={m['n_visits']}):"
                  f" FP-leak={100*r['fp_leak_rate']:.1f}% ({m['total_fp_leaked']}/{m['total_gt_nabs']}),"
                  f" novel={m['total_novel']}")
        print()

    ofat_files = sorted(_RESULTS.glob("ofat_*.json"))
    if not ofat_files:
        print("(no OFAT results yet)")
        return

    print("=" * 50)
    print("OFAT sweep")
    print("=" * 50)
    for f in ofat_files:
        d = json.loads(f.read_text())
        knob = d["knob"]
        cur_val = d["current_prod_value"]
        print(f"\n{knob}  (current: {cur_val!r})")
        print(f"  {'value':<22}{'FP leak':<14}{'TP-real':<14}{'novel':<10}{'cls-hit':<10}")
        for v in d["values"]:
            val_repr = repr(v["value"])[:20]
            mark = " ★" if v["value"] == cur_val else "  "
            print(f"  {val_repr:<22}{100*v['fp_leak_rate']:>5.1f}%({v['fp_leaked']:>2}/{v['gt_nabs']:<2})  "
                  f"{100*v['tp_preservation_real']:>5.1f}%({v['tps_preserved_real']:>2}/{v['gt_real']:<2})  "
                  f"{v['novel_detections']:<10}{100*v['classifier_hit_rate']:>5.1f}%{mark}")

    classifier = _load("classifier_in_range_sweep.json")
    if classifier:
        print()
        print("=" * 50)
        print("Classifier IN_RANGE_THRESHOLD sweep")
        print("=" * 50)
        print(f"  {'thresh':<10}{'NAB-rej':<14}{'real top-1':<14}{'real-rej':<14}")
        for r in classifier:
            print(f"  {r['threshold']:<10.3f}{100*r['nab_rejection_rate']:>4.0f}% ({r['nab_rejected']}/{r['n_nab']:<4})  "
                  f"{100*r['real_top1_acc']:>5.1f}% ({r['real_top1_correct']}/{max(1, r['n_real'] - r['real_rejected']):<4})  "
                  f"{100*r['real_rejection_rate']:>4.0f}% ({r['real_rejected']}/{r['n_real']})")


if __name__ == "__main__":
    sys.exit(main())
