"""SQLAlchemy ORM models."""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


class Species(Base):
    __tablename__ = "species"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    common_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String)
    nabirds_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_rare: Mapped[bool] = mapped_column(Integer, default=0)  # SQLite boolean


class Visit(Base):
    """A burst of frames containing one or more birds, treated as a single 'visit'."""

    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # Pipeline state. processed_at is set by the worker when the clip has been
    # extracted + detected + tracked. processing_error captures the last failure
    # so we can retry or surface bad clips.
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(back_populates="visit", cascade="all, delete-orphan")


class Detection(Base):
    """One identified bird (per-track aggregate) within a visit."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"), index=True)
    species_id: Mapped[int | None] = mapped_column(ForeignKey("species.id"), nullable=True, index=True)

    confidence: Mapped[float] = mapped_column(Float)
    # Top-5 predictions before fusion: [{"species": "...", "p": 0.55}, ...]
    raw_predictions: Mapped[list] = mapped_column(JSON, default=list)
    # Whether Haikubox heard the same species within the correlation window
    audio_confirmed: Mapped[bool] = mapped_column(Integer, default=0)

    crop_path: Mapped[str] = mapped_column(String)
    bbox: Mapped[list] = mapped_column(JSON)  # [x, y, w, h]
    track_id: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Correction(Base):
    """User correction of a species ID, used for active learning."""

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), index=True)
    correct_species_id: Mapped[int] = mapped_column(ForeignKey("species.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
