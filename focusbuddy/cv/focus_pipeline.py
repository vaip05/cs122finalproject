"""Live focus sampler: webcam → face → HF model → labels → annotated preview.

Owned by Author #1: Vaishnavi Panchal.
Uses Ara's map_to_focus_label / LabelSmoother from labeling.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

import config
from focusbuddy.cv.camera import Camera
from focusbuddy.cv.emotion_model import EmotionModel, EmotionPrediction
from focusbuddy.cv.face_detector import BBox, FaceDetector
from focusbuddy.cv.labeling import LabelSmoother, map_to_focus_label

FocusCallback = Callable[[dict], None]


@dataclass
class FocusSample:
    raw_label: str
    smoothed_label: str
    confidence: float
    face_detected: bool
    timestamp: float
    bbox: Optional[BBox] = None
    fer_label: str = ""
    understand_score: float = 0.0
    not_understand_score: float = 0.0


class FocusPipeline:
    """Background loop: fast preview + periodic model inference."""

    def __init__(
        self,
        on_sample: Optional[FocusCallback] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self.camera = Camera()
        self.face_detector = FaceDetector()
        self.emotion_model = EmotionModel()
        self.smoother = LabelSmoother()
        self.on_sample = on_sample
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.load_confidence_threshold()
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()
        self._logging = False  # only write CSV samples while a session is running
        self.last_sample: Optional[FocusSample] = None
        self.status_message = "idle"
        self.model_ready = False

        self._preview_lock = threading.Lock()
        self._preview_rgb: Optional[np.ndarray] = None
        self._last_bbox: Optional[BBox] = None
        self._last_infer_time = 0.0
        self._baseline_face_area: Optional[float] = None
        self._cached_emotion = EmotionPrediction(
            engagement_label="unknown",
            engagement_confidence=0.0,
            understand_score=0.0,
            not_understand_score=0.0,
            fer_label="unknown",
            fer_confidence=0.0,
            scores={},
        )

    def prepare(self) -> bool:
        """Open camera, load model, and start preview loop."""
        cam_ok = self.camera.open()
        model_ok = self.emotion_model.load()
        self.model_ready = model_ok

        if not cam_ok and not model_ok:
            self.status_message = (
                f"Camera unavailable; model load failed: {self.emotion_model.load_error}"
            )
            return False
        if not cam_ok:
            self.status_message = "Camera unavailable"
            return False
        if not model_ok:
            self.status_message = (
                f"Model load failed ({self.emotion_model.load_error}). "
                "Tracking away vs present only."
            )
        else:
            self.status_message = "ready"

        # Always run preview once camera is open.
        self._stop.clear()
        self._paused.clear()
        if not (self._thread and self._thread.is_alive()):
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return True

    def start(self) -> None:
        """Begin logging focus events for a study session."""
        self.smoother.reset("away")
        self._baseline_face_area = None
        self._logging = True
        self._paused.clear()

    def pause(self) -> None:
        self._logging = False
        # Keep preview running; only stop logging.

    def resume(self) -> None:
        self._logging = True
        self._paused.clear()

    def stop(self) -> None:
        self._logging = False
        self._stop.set()
        self._paused.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self.camera.close()
        self.status_message = "stopped"

    def get_preview_rgb(self) -> Optional[np.ndarray]:
        with self._preview_lock:
            if self._preview_rgb is None:
                return None
            return self._preview_rgb.copy()

    def _box_color(self, label: str) -> Tuple[int, int, int]:
        if label == "focused":
            return config.BOX_COLOR_FOCUSED
        if label == "distracted":
            return config.BOX_COLOR_DISTRACTED
        return config.BOX_COLOR_AWAY

    def _annotate(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[BBox],
        label: str,
        confidence: float,
    ) -> np.ndarray:
        # Mirror first (selfie view), then draw box/text so labels stay readable.
        mirrored = cv2.flip(frame_bgr, 1)
        frame_w = frame_bgr.shape[1]

        if bbox is not None:
            x, y, w, h = bbox
            # Remap box into mirrored coordinates.
            mx = frame_w - x - w
            color = self._box_color(label)
            cv2.rectangle(mirrored, (mx, y), (mx + w, y + h), color, 3)
            caption = f"{label}  {confidence:.2f}" if label != "away" else "away"
            bar_w = max(w, 160)
            cv2.rectangle(
                mirrored,
                (mx, max(0, y - 28)),
                (mx + bar_w, y),
                color,
                -1,
            )
            cv2.putText(
                mirrored,
                caption,
                (mx + 6, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                mirrored,
                "No face detected - away",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                config.BOX_COLOR_AWAY,
                2,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)

    def _resize_preview(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        scale = min(config.PREVIEW_MAX_WIDTH / w, config.PREVIEW_MAX_HEIGHT / h, 1.0)
        if scale >= 0.999:
            return rgb
        return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _pose_flags(self, frame_bgr: np.ndarray, bbox: BBox) -> Tuple[bool, bool]:
        """Return (off_center, face_too_small) engagement proxies."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_bgr.shape[:2]
        face_cx = x + w / 2.0
        face_cy = y + h / 2.0
        norm_dx = abs(face_cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
        norm_dy = abs(face_cy - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
        off_center = max(norm_dx, norm_dy) >= config.OFF_CENTER_DISTRACT_RATIO

        area = float(w * h)
        if self._baseline_face_area is None:
            self._baseline_face_area = area
        face_too_small = area < self._baseline_face_area * config.SMALL_FACE_AREA_RATIO
        return off_center, face_too_small

    def classify_frame(self, run_model: bool) -> FocusSample:
        frame = self.camera.read()
        now = time.time()

        if frame is None:
            raw = "away"
            sample = FocusSample(
                raw_label=raw,
                smoothed_label=self.smoother.update(raw, now),
                confidence=0.0,
                face_detected=False,
                timestamp=now,
                bbox=None,
            )
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera unavailable",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            with self._preview_lock:
                self._preview_rgb = blank
            self.last_sample = sample
            return sample

        bbox = self.face_detector.detect_largest(frame)
        self._last_bbox = bbox
        pred = self._cached_emotion
        confidence = 0.0
        fer_label = pred.fer_label

        if bbox is None:
            raw = "away"
            face_detected = False
        else:
            face_detected = True
            off_center, face_too_small = self._pose_flags(frame, bbox)
            if run_model:
                crop = self.face_detector.crop_box(frame, bbox)
                if crop is not None and self.emotion_model.ready:
                    pred = self.emotion_model.predict(crop)
                    self._cached_emotion = pred
                elif not self.emotion_model.ready:
                    pred = EmotionPrediction(
                        engagement_label="understand",
                        engagement_confidence=1.0,
                        understand_score=1.0,
                        not_understand_score=0.0,
                        fer_label="neutral",
                        fer_confidence=1.0,
                        scores={},
                    )
                    self._cached_emotion = pred

            confidence = max(pred.engagement_confidence, pred.fer_confidence)
            fer_label = pred.fer_label
            raw = map_to_focus_label(
                face_detected=True,
                emotion_label=pred.engagement_label,
                confidence=pred.engagement_confidence,
                threshold=self.confidence_threshold,
                understand_score=pred.understand_score,
                not_understand_score=pred.not_understand_score,
                fer_label=pred.fer_label,
                fer_confidence=pred.fer_confidence,
                off_center=off_center,
                face_too_small=face_too_small,
            )

        if run_model:
            self._last_infer_time = now

        smoothed = self.smoother.update(raw, now)
        # UI: away immediately with no face; otherwise show smoothed focus state.
        display_label = "away" if not face_detected else smoothed
        display_confidence = 0.0 if not face_detected else confidence

        preview = self._resize_preview(
            self._annotate(frame, bbox, display_label, display_confidence)
        )
        with self._preview_lock:
            self._preview_rgb = preview

        sample = FocusSample(
            raw_label=raw,
            smoothed_label=smoothed,
            confidence=confidence if face_detected else 0.0,
            face_detected=face_detected,
            timestamp=now,
            bbox=bbox,
            fer_label=fer_label if face_detected else "",
            understand_score=pred.understand_score if face_detected else 0.0,
            not_understand_score=pred.not_understand_score if face_detected else 0.0,
        )
        self.last_sample = sample
        return sample

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = time.time()
                run_model = (now - self._last_infer_time) >= config.SAMPLE_INTERVAL_SECONDS
                sample = self.classify_frame(run_model=run_model)
                if self._logging and self.on_sample is not None and run_model:
                    self.on_sample(
                        {
                            "raw_label": sample.raw_label,
                            "smoothed_label": sample.smoothed_label,
                            "confidence": sample.confidence,
                            "face_detected": sample.face_detected,
                            "timestamp": sample.timestamp,
                            "camera_status": self.camera.status,
                            "bbox": sample.bbox,
                            "fer_label": sample.fer_label,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                self.status_message = f"pipeline error: {exc}"
            time.sleep(config.PREVIEW_INTERVAL_SECONDS)
