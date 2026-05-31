"""SQLAlchemy ORM models."""
from datetime import date as _date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base
from db.utils import utcnow

# Reserved Species.common_name for "the user looked at this and it's not
# actually a bird" corrections (hummingbird feeders, wind spinners,
# leaves the classifier insisted were Ovenbirds, etc.). Used by:
#   - SpeciesPicker (renders this option pinned at the top)
#   - GET /api/detections (filters these out by default so the feed shows
#     only real birds)
#   - GET /api/species (excludes this from the picker's species list since
#     it's surfaced separately)
#   - The eventual retraining script (uses these as negative examples).
NOT_A_BIRD_LABEL = "Not a bird"

# Reserved Species.common_name for "this is definitely a bird but I can't
# tell the species" corrections. Distinct from NULL species_id (which means
# the classifier was unsure and the user hasn't weighed in yet) — UNKNOWN_BIRD
# is a positive human confirmation. Highest-value YOLO training data: a
# guaranteed-correct positive bounding box without species commitment.
# Unlike NOT_A_BIRD, these DO show in the feed (they're real bird sightings).
UNKNOWN_BIRD_LABEL = "Unknown bird"

# Sentinel labels that aren't real species. Used by /api/species to exclude
# them from the picker's species lists (they're surfaced separately) and by
# the eventual retraining script to handle them with their own logic.
SENTINEL_LABELS = frozenset({NOT_A_BIRD_LABEL, UNKNOWN_BIRD_LABEL})


class Species(Base):
    __tablename__ = "species"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    common_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String)
    nabirds_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_rare: Mapped[bool] = mapped_column(Integer, default=0)  # SQLite boolean
    # Family-level catch-all labels ("Sparrow", "Warbler"). These behave
    # like species rows in the DB (a Correction can point at one), but
    # downstream code treats them differently:
    #   - the species picker pins them in their own section,
    #   - the size prior calibration excludes them (wide variance),
    #   - classifier-accuracy stats give partial credit if the model's
    #     top-1 was any member species,
    #   - retraining pipelines treat them as soft labels.
    is_family: Mapped[bool | None] = mapped_column(Integer, nullable=True, default=0)


class Visit(Base):
    """A burst of frames containing one or more birds, treated as a single 'visit'."""

    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # Pipeline state. processed_at is set by the worker when the clip has been
    # extracted + detected + tracked. processing_error captures the last failure
    # so we can retry or surface bad clips.
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many YOLO detections the scene mask dropped during processing.
    # Tracked so the Stats page can surface the otherwise-invisible
    # "real bird at the hummingbird feeder got silently filtered" failure
    # mode. Nullable for rows written before this column existed.
    scene_mask_suppressed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(back_populates="visit", cascade="all, delete-orphan")


class Detection(Base):
    """One identified bird (per-track aggregate) within a visit."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), index=True)
    species_id: Mapped[int | None] = mapped_column(ForeignKey("species.id"), nullable=True, index=True)

    confidence: Mapped[float] = mapped_column(Float)
    # Raw YOLO confidence on the best (saved) detection. Distinct from
    # `confidence` above, which is the FUSED post-classifier+priors score
    # (collapsed to 0.0 when the classifier rejects). Captured so we can
    # study whether YOLO confidence alone separates real birds from false
    # positives — see scripts/analyze_yolo_confidence.py.
    # Nullable because rows inserted before this column existed don't have it.
    yolo_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Top-5 predictions before fusion: [{"species": "...", "p": 0.55}, ...]
    raw_predictions: Mapped[list] = mapped_column(JSON, default=list)
    # Whether Haikubox heard the same species within the correlation window
    audio_confirmed: Mapped[bool] = mapped_column(Integer, default=0)

    crop_path: Mapped[str] = mapped_column(String)
    bbox: Mapped[list] = mapped_column(JSON)  # [x, y, w, h] — best frame
    # Every per-frame YOLO bbox in the track that produced this detection,
    # not just the saved best one. Same `[x, y, w, h]` shape, in full-frame
    # 4K coords. Used by the size-prior calibration to compute a
    # track-median max(w, h) instead of trusting the single best frame's
    # noisy bbox. Nullable because rows inserted before this column existed
    # don't have it.
    track_bboxes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    track_id: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    visit: Mapped[Visit] = relationship(back_populates="detections")
    species: Mapped[Species | None] = relationship()


class HaikuboxDetection(Base):
    """Cached recent Haikubox audio detections for fusion / correlation."""

    __tablename__ = "haikubox_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    species_common_name: Mapped[str] = mapped_column(String, index=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    endpoint: Mapped[str] = mapped_column(Text, unique=True)
    p256dh: Mapped[str] = mapped_column(Text)
    auth: Mapped[str] = mapped_column(Text)
    # Push when this species has not been seen in the last N days.
    # Default 30: a typical Baltimore Oriole arriving in May after winter
    # absence pings; a House Sparrow showing up for the hundredth time doesn't.
    notify_window_days: Mapped[int] = mapped_column(Integer, default=30)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PipelineStatsDaily(Base):
    """One row per UTC date of pipeline funnel metrics.

    Written by the nightly job in pipeline.worker and re-used by the
    /api/stats endpoint. Today's row is recomputed on every read so the
    page is always current; yesterday-and-older are cached snapshots.

    The `payload` JSON column holds variable-shape bits the schema can't
    easily express (top species, per-species accuracy, hour-of-day
    histogram). New bonus charts should usually add a key to payload
    rather than a new column — keeps additive migrations rare.
    """

    __tablename__ = "pipeline_stats_daily"

    # UTC date (Visit.started_at's date). Used both as PK and as the
    # natural sort key; SQLite stores it as ISO text.
    date: Mapped[_date] = mapped_column(Date, primary_key=True)

    # Funnel stage counts (see backend/pipeline/stats.py docstring for the
    # exact GROUP BY definitions).
    clips_received: Mapped[int] = mapped_column(Integer, default=0)
    clips_daylight: Mapped[int] = mapped_column(Integer, default=0)
    clips_with_detections: Mapped[int] = mapped_column(Integer, default=0)
    detections_total: Mapped[int] = mapped_column(Integer, default=0)
    detections_labeled_by_classifier: Mapped[int] = mapped_column(Integer, default=0)
    detections_user_corrected: Mapped[int] = mapped_column(Integer, default=0)
    corrections_nab: Mapped[int] = mapped_column(Integer, default=0)
    corrections_unknown: Mapped[int] = mapped_column(Integer, default=0)
    corrections_real_species: Mapped[int] = mapped_column(Integer, default=0)
    # Of corrections to a real species (denominator excludes NAB / Unknown),
    # how many did the classifier's species_id already match? Used to compute
    # classifier accuracy.
    classifier_correct: Mapped[int] = mapped_column(Integer, default=0)
    classifier_eligible: Mapped[int] = mapped_column(Integer, default=0)
    # Operational signals.
    visits_with_processing_error: Mapped[int] = mapped_column(Integer, default=0)
    detections_audio_confirmed: Mapped[int] = mapped_column(Integer, default=0)
    # Aggregate count of YOLO detections the scene mask silently dropped
    # across all visits on this date. A spike here means lots of motion
    # is happening in NAB-clustered regions (sun glints, leaves, hummingbird
    # feeder) — and possibly real birds being filtered along with them.
    detections_scene_mask_suppressed: Mapped[int] = mapped_column(Integer, default=0)

    # Variable-shape extras: top species (list), per-species accuracy
    # (list), hour-of-day histogram, YOLO-confidence buckets.
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Correction(Base):
    """User correction of a species ID, used for active learning."""

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), index=True)
    correct_species_id: Mapped[int] = mapped_column(ForeignKey("species.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Where the correction came from. NULL = legacy / user-via-UI (the
    # default). `"llm-claude"` = generated by
    # scripts/llm_classify_unidentified.py. We need to distinguish these
    # because LLM-generated labels carry the model's biases — fine for
    # the size prior (sizes are objective) but suspect for any future
    # classifier fine-tune where we'd train on our own mistakes.
    source: Mapped[str | None] = mapped_column(String, nullable=True)
