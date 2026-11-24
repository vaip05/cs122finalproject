"""Hugging Face student-emotion + FER classifiers.

Owned by Author #1: Vaishnavi Panchal.

gowdaman/gowdaman-emotion-detection is binary (understand / not_understand) and in
practice often sticks on "understand". We also run the base FER model so distraction
(surprise/sad/fear/etc.) can surface during a study session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image

import config


@dataclass
class EmotionPrediction:
    engagement_label: str
    engagement_confidence: float
    understand_score: float
    not_understand_score: float
    fer_label: str
    fer_confidence: float
    scores: Dict[str, float]


class EmotionModel:
    """Engagement (gowdaman) + expression (FER) for face crops."""

    def __init__(
        self,
        model_id: str = config.HF_MODEL_ID,
        processor_id: str = config.HF_PROCESSOR_ID,
        fer_model_id: str = config.HF_FER_MODEL_ID,
    ) -> None:
        self.model_id = model_id
        self.processor_id = processor_id
        self.fer_model_id = fer_model_id
        self._engage_pipe = None
        self._fer_pipe = None
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        if self._engage_pipe is not None and self._fer_pipe is not None:
            return True
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification,
                pipeline,
            )

            processor = AutoImageProcessor.from_pretrained(self.processor_id)
            engage_model = AutoModelForImageClassification.from_pretrained(self.model_id)
            fer_model = AutoModelForImageClassification.from_pretrained(self.fer_model_id)

            self._engage_pipe = pipeline(
                "image-classification",
                model=engage_model,
                image_processor=processor,
                top_k=None,
            )
            self._fer_pipe = pipeline(
                "image-classification",
                model=fer_model,
                image_processor=processor,
                top_k=None,
            )
            self.load_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)
            self._engage_pipe = None
            self._fer_pipe = None
            return False

    @property
    def ready(self) -> bool:
        return self._engage_pipe is not None and self._fer_pipe is not None

    @staticmethod
    def _score_map(results) -> Dict[str, float]:
        if not results:
            return {}
        scores = results[0] if isinstance(results[0], list) else results
        return {str(item["label"]).lower(): float(item["score"]) for item in scores}

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        """Run both classifiers on a BGR face crop."""
        if not self.ready:
            return EmotionPrediction(
                engagement_label="unknown",
                engagement_confidence=0.0,
                understand_score=0.0,
                not_understand_score=0.0,
                fer_label="unknown",
                fer_confidence=0.0,
                scores={},
            )

        rgb = face_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)

        engage_scores = self._score_map(self._engage_pipe(image))
        fer_scores = self._score_map(self._fer_pipe(image))

        understand = engage_scores.get("understand", 0.0)
        not_understand = engage_scores.get("not_understand", 0.0)
        # Some checkpoints use spaced labels.
        if understand == 0.0 and not_understand == 0.0:
            understand = engage_scores.get("understand", 0.0)
            for key, value in engage_scores.items():
                if "not" in key and "understand" in key:
                    not_understand = value
                elif "understand" in key:
                    understand = max(understand, value)

        if engage_scores:
            engagement_label, engagement_confidence = max(
                engage_scores.items(), key=lambda kv: kv[1]
            )
        else:
            engagement_label, engagement_confidence = "unknown", 0.0

        if fer_scores:
            fer_label, fer_confidence = max(fer_scores.items(), key=lambda kv: kv[1])
        else:
            fer_label, fer_confidence = "unknown", 0.0

        return EmotionPrediction(
            engagement_label=engagement_label,
            engagement_confidence=engagement_confidence,
            understand_score=understand,
            not_understand_score=not_understand,
            fer_label=fer_label,
            fer_confidence=fer_confidence,
            scores={**engage_scores, **{f"fer_{k}": v for k, v in fer_scores.items()}},
        )
