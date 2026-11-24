"""Webcam capture helpers.

Owned by Author #1: Vaishnavi Panchal.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

import config


class Camera:
    """Thread-safe webcam wrapper for the live study session loop."""

    def __init__(self, index: int = config.CAMERA_INDEX) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self.status = "closed"

    def open(self) -> bool:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self.status = "open"
                return True
            self._cap = cv2.VideoCapture(self.index)
            if not self._cap.isOpened():
                self.status = "unavailable"
                self._cap.release()
                self._cap = None
                return False
            self.status = "open"
            return True

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self.status = "unavailable"
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.status = "error"
                return None
            self.status = "open"
            return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.status = "closed"
