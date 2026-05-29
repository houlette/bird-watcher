"""Score a ReplayResult against the user's labeled ground truth.

A produced detection "matches" a ground-truth detection if their bboxes
have IoU ≥ MATCH_IOU. We allow matches across frame indices because the
replay's track-best frame may differ from the original — what matters
for feed quality is whether YOLO+pipeline produced a detection at the
same SPATIAL location.

Three metrics per visit:

  * fp_leaked       : count of replay detections that match a known-NAB GT
                      detection. (Lower = better)
  * tps_preserved   : count of GT real-species detections that the replay
                      still produced a matching detection for. (Higher = better)
  * classifier_hits : of preserved real-species TPs, count where the
                      replay's predicted species matched the GT species.

Aggregated across visits → one row per config.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.sweep.replay import ReplayResult


MATCH_IOU = 0.30


# Sentinel labels that aren't "real species" but ARE valid ground truth.
NAB_LABEL = "Not a bird"
UNKNOWN_LABEL = "Unknown bird"


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class VisitMetrics:
    visit_id: int
    # Ground-truth counts in this visit
    n_gt_nabs: int = 0
    n_gt_real: int = 0
    n_gt_unknown: int = 0
    # Replay outcomes
    fp_leaked: int = 0           # GT-NAB locations the replay re-produced
    tps_preserved_real: int = 0  # GT real-species locations the replay preserved
    tps_preserved_unknown: int = 0
    classifier_hits: int = 0     # of preserved real-species, where species matched
    classifier_misses: int = 0   # preserved but wrong species
    classifier_rejected: int = 0 # preserved but species=None
    novel_detections: int = 0    # replay detections that matched no GT


@dataclass
class AggregateMetrics:
    config_label: str
    n_visits: int = 0
    total_gt_nabs: int = 0
    total_gt_real: int = 0
    total_gt_unknown: int = 0
    total_fp_leaked: int = 0
    total_tps_preserved_real: int = 0
    total_tps_preserved_unknown: int = 0
    total_classifier_hits: int = 0
    total_classifier_misses: int = 0
    total_classifier_rejected: int = 0
    total_novel: int = 0

    @property
    def fp_leak_rate(self) -> float:
        return self.total_fp_leaked / self.total_gt_nabs if self.total_gt_nabs else 0.0

    @property
    def tp_preservation_rate_real(self) -> float:
        return self.total_tps_preserved_real / self.total_gt_real if self.total_gt_real else 0.0

    @property
    def tp_preservation_rate_any(self) -> float:
        n_any = self.total_gt_real + self.total_gt_unknown
        preserved_any = self.total_tps_preserved_real + self.total_tps_preserved_unknown
        return preserved_any / n_any if n_any else 0.0

    @property
    def classifier_hit_rate(self) -> float:
        # Of preserved real-species TPs that the classifier didn't reject.
        denom = self.total_classifier_hits + self.total_classifier_misses
        return self.total_classifier_hits / denom if denom else 0.0


def score_visit(replay: ReplayResult, ground_truth_for_visit: list[dict]) -> VisitMetrics:
    """Compare one replay run against the GT bboxes for the same visit."""
    metrics = VisitMetrics(visit_id=replay.visit_id)

    # Tally GT counts and bucket them.
    nab_gts = []
    real_gts = []
    unknown_gts = []
    for gt in ground_truth_for_visit:
        if gt["ground_truth_species"] == NAB_LABEL:
            nab_gts.append(gt)
            metrics.n_gt_nabs += 1
        elif gt["ground_truth_species"] == UNKNOWN_LABEL:
            unknown_gts.append(gt)
            metrics.n_gt_unknown += 1
        else:
            real_gts.append(gt)
            metrics.n_gt_real += 1

    # For each replay detection, find best-matching GT (if any).
    matched_gt_ids: set[int] = set()
    for det in replay.detections:
        best_match = None
        best_iou = 0.0
        for gt in ground_truth_for_visit:
            iou = _iou(det.bbox, tuple(gt["bbox"]))
            if iou >= MATCH_IOU and iou > best_iou:
                best_match = gt
                best_iou = iou
        if best_match is None:
            metrics.novel_detections += 1
            continue
        matched_gt_ids.add(best_match["detection_id"])
        gt_species = best_match["ground_truth_species"]
        if gt_species == NAB_LABEL:
            metrics.fp_leaked += 1
        elif gt_species == UNKNOWN_LABEL:
            metrics.tps_preserved_unknown += 1
        else:
            metrics.tps_preserved_real += 1
            # Classifier scoring on real-species TPs.
            if det.species is None:
                metrics.classifier_rejected += 1
            elif det.species == gt_species:
                metrics.classifier_hits += 1
            else:
                metrics.classifier_misses += 1
    return metrics


def aggregate(metrics_per_visit: list[VisitMetrics], config_label: str) -> AggregateMetrics:
    agg = AggregateMetrics(config_label=config_label)
    for m in metrics_per_visit:
        agg.n_visits += 1
        agg.total_gt_nabs += m.n_gt_nabs
        agg.total_gt_real += m.n_gt_real
        agg.total_gt_unknown += m.n_gt_unknown
        agg.total_fp_leaked += m.fp_leaked
        agg.total_tps_preserved_real += m.tps_preserved_real
        agg.total_tps_preserved_unknown += m.tps_preserved_unknown
        agg.total_classifier_hits += m.classifier_hits
        agg.total_classifier_misses += m.classifier_misses
        agg.total_classifier_rejected += m.classifier_rejected
        agg.total_novel += m.novel_detections
    return agg


def format_report(agg: AggregateMetrics) -> str:
    lines = [
        f"=== Config: {agg.config_label} ===",
        f"  Visits scored:                  {agg.n_visits}",
        f"  GT NABs:                        {agg.total_gt_nabs}",
        f"  GT real species:                {agg.total_gt_real}",
        f"  GT unknown:                     {agg.total_gt_unknown}",
        "",
        f"  FP leaked:                      {agg.total_fp_leaked:>4} / {agg.total_gt_nabs}  ({100*agg.fp_leak_rate:.1f}%)  ↓ better",
        f"  Real-species TPs preserved:     {agg.total_tps_preserved_real:>4} / {agg.total_gt_real}  ({100*agg.tp_preservation_rate_real:.1f}%)  ↑ better",
        f"  Any-bird TPs preserved:         {agg.total_tps_preserved_real + agg.total_tps_preserved_unknown:>4} / {agg.total_gt_real + agg.total_gt_unknown}  ({100*agg.tp_preservation_rate_any:.1f}%)",
        f"  Classifier hit rate:            {agg.total_classifier_hits:>4} / {agg.total_classifier_hits + agg.total_classifier_misses}  ({100*agg.classifier_hit_rate:.1f}%)  ↑ better",
        f"  Classifier rejected (Unident.): {agg.total_classifier_rejected}",
        f"  Novel detections:               {agg.total_novel}  (no GT match — uncategorized)",
    ]
    return "\n".join(lines)
