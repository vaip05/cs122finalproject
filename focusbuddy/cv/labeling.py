"""Emotion → focus mapping and temporal smoothing.

Owned by Author #2: Aradhana (Ara) Anandkumar.
Uses gowdaman engagement scores + FER expression + simple pose cues.
"""

from __future__ import annotations

import time
from typing import Optional

import config


def map_to_focus_label(
    face_detected: bool,
    emotion_label: str = "",
    confidence: float = 0.0,
    threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    *,
    understand_score: float = 0.0,
    not_understand_score: float = 0.0,
    fer_label: str = "",
    fer_confidence: float = 0.0,
    off_center: bool = False,
    face_too_small: bool = False,
) -> str:
    """
    Map signals → focused / distracted / away.

    gowdaman alone often stays on "understand"; FER + pose cues unlock distraction.
    """
    if not face_detected:
        return "away"

    fer = fer_label.strip().lower()
    distracted_fer = fer in config.FER_DISTRACTED_LABELS and fer_confidence >= 0.30
    focused_fer = fer in config.FER_FOCUSED_LABELS and fer_confidence >= 0.30

    # Strong distraction cues first.
    if not_understand_score >= config.NOT_UNDERSTAND_DISTRACT_SCORE:
        return "distracted"
    if distracted_fer:
        return "distracted"
    if off_center or face_too_small:
        return "distracted"

    # Focused only when engagement + calm expression agree.
    understands = understand_score >= threshold or (
        "understand" in emotion_label.lower()
        and "not" not in emotion_label.lower()
        and confidence >= threshold
    )
    if understands and focused_fer:
        return "focused"
    if understands and fer_confidence < 0.30:
        # Weak FER signal — trust engagement model.
        return "focused"

    # Face present but cues disagree / uncertain → distracted for study monitoring.
    return "distracted"


class LabelSmoother:
    """Accept a label change only after it holds briefly (faster for away/distracted)."""

    def __init__(
        self,
        smoothing_seconds: float = config.SMOOTHING_SECONDS,
        away_smoothing_seconds: float = config.AWAY_SMOOTHING_SECONDS,
        distracted_smoothing_seconds: float = config.DISTRACTED_SMOOTHING_SECONDS,
    ) -> None:
        self.smoothing_seconds = smoothing_seconds
        self.away_smoothing_seconds = away_smoothing_seconds
        self.distracted_smoothing_seconds = distracted_smoothing_seconds
        self.current_label = "away"
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None

    def _hold_for(self, label: str) -> float:
        if label == "away":
            return self.away_smoothing_seconds
        if label == "distracted":
            return self.distracted_smoothing_seconds
        return self.smoothing_seconds

    def update(self, raw_label: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        if raw_label == self.current_label:
            self._candidate = None
            self._candidate_since = None
            return self.current_label

        if self._candidate != raw_label:
            self._candidate = raw_label
            self._candidate_since = now
            return self.current_label

        assert self._candidate_since is not None
        if now - self._candidate_since >= self._hold_for(raw_label):
            self.current_label = raw_label
            self._candidate = None
            self._candidate_since = None
        return self.current_label

    def reset(self, label: str = "away") -> None:
        self.current_label = label
        self._candidate = None
        self._candidate_since = None
