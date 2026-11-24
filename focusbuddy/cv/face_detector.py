"""Face detection and cropping for the emotion model.

Owned by Author #1: Vaishnavi Panchal.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

import config

BBox = Tuple[int, int, int, int]


class FaceDetector:
    """OpenCV Haar-cascade detector; returns largest face box/crop or None (away)."""

    def __init__(self, min_face_area: int = config.MIN_FACE_AREA) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.min_face_area = min_face_area

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        if int(w) * int(h) < self.min_face_area:
            return None
        return int(x), int(y), int(w), int(h)

    def crop_box(
        self,
        frame_bgr: np.ndarray,
        box: BBox,
        padding: float = 0.20,
    ) -> Optional[np.ndarray]:
        x, y, w, h = box
        height, width = frame_bgr.shape[:2]
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + w + pad_x)
        y1 = min(height, y + h + pad_y)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return crop

    def crop_face(self, frame_bgr: np.ndarray, padding: float = 0.20) -> Optional[np.ndarray]:
        box = self.detect_largest(frame_bgr)
        if box is None:
            return None
        return self.crop_box(frame_bgr, box, padding=padding)
