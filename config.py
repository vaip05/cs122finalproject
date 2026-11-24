"""FocusBuddy shared configuration.

Author #1 (Vaishnavi): paths, camera/model IDs, live-loop threshold defaults.
Author #2 (Ara): may extend calibration loading; smoothing seconds used by labeling.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CALIBRATION_PATH = DATA_DIR / "user_calibration.json"

STUDY_SESSIONS_CSV = DATA_DIR / "study_sessions.csv"
FOCUS_EVENTS_CSV = DATA_DIR / "focus_events.csv"
STUDENT_FEEDBACK_CSV = DATA_DIR / "student_feedback.csv"
EXTERNAL_STUDY_CSV = DATA_DIR / "external_study_data.csv"

HF_MODEL_ID = "gowdaman/gowdaman-emotion-detection"
# gowdaman's repo ships weights only; reuse the base ViT preprocessor.
HF_PROCESSOR_ID = "trpakov/vit-face-expression"
# Secondary 7-class FER model (same family) — gowdaman alone almost never emits distraction.
HF_FER_MODEL_ID = "trpakov/vit-face-expression"

# Live focus gating (Vaishnavi owns applying this in the sampler loop)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
# If not_understand probability reaches this, treat as distracted even if not top label.
NOT_UNDERSTAND_DISTRACT_SCORE = 0.30
# FER emotions that count as distracted study behavior.
FER_DISTRACTED_LABELS = ("angry", "disgust", "fear", "sad", "surprise")
FER_FOCUSED_LABELS = ("neutral", "happy")
# Face far from frame center → likely looking away from work → distracted.
OFF_CENTER_DISTRACT_RATIO = 0.28
# Face much smaller than when session started → leaned back / disengaged.
SMALL_FACE_AREA_RATIO = 0.45

# Used by Ara's smoother; exposed here so both sides share one value
SMOOTHING_SECONDS = 3.0
DISTRACTED_SMOOTHING_SECONDS = 1.5
# Flip to away much faster so leaving the desk doesn't keep saying "focused"
AWAY_SMOOTHING_SECONDS = 1.0

# Preview loop vs model inference cadence
PREVIEW_INTERVAL_SECONDS = 0.08  # ~12 fps UI preview
SAMPLE_INTERVAL_SECONDS = 1.5    # how often to run the HF model
MIN_FACE_AREA = 50 * 50
CAMERA_INDEX = 0
PREVIEW_MAX_WIDTH = 480
PREVIEW_MAX_HEIGHT = 360

# Bounding-box colors (BGR for OpenCV drawing)
BOX_COLOR_FOCUSED = (80, 180, 80)      # green
BOX_COLOR_DISTRACTED = (60, 140, 255)  # orange
BOX_COLOR_AWAY = (120, 120, 120)       # gray

EXTERNAL_DATA_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "college-majors/recent-grads.csv"
)

APP_NAME = "FocusBuddy"
PRODUCT_LABEL = "FocusGuard"
DISCLAIMER = (
    "Focus status is an approximate engagement proxy based on facial "
    "expression. It is not an objective measure of concentration."
)


def load_confidence_threshold() -> float:
    """Load threshold from calibration file if Ara/user saved one; else default."""
    import json

    if CALIBRATION_PATH.exists():
        try:
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            value = float(data.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
            return max(0.40, min(0.90, value))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return DEFAULT_CONFIDENCE_THRESHOLD
