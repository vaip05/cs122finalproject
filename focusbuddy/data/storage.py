"""CSV persistence for sessions, focus events, and feedback.

Owned by Author #1: Vaishnavi Panchal.
"""

from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import config

STUDY_SESSION_FIELDS = [
    "session_id",
    "date",
    "subject",
    "goal",
    "planned_duration_min",
    "start_time",
    "end_time",
    "goal_completed",
    "self_focus_rating",
]

FOCUS_EVENT_FIELDS = [
    "session_id",
    "timestamp",
    "raw_label",
    "smoothed_label",
    "confidence",
    "face_detected",
]

FEEDBACK_FIELDS = [
    "session_id",
    "timestamp",
    "faced_distractions",
    "distraction_notes",
    "self_focus_rating",
    "goal_completed",
]


def ensure_data_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_csv(config.STUDY_SESSIONS_CSV, STUDY_SESSION_FIELDS)
    _ensure_csv(config.FOCUS_EVENTS_CSV, FOCUS_EVENT_FIELDS)
    _ensure_csv(config.STUDENT_FEEDBACK_CSV, FEEDBACK_FIELDS)


def _ensure_csv(path: Path, fieldnames: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def append_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_data_dirs()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def create_session(subject: str, goal: str, planned_duration_min: float) -> dict[str, Any]:
    now = datetime.now()
    session = {
        "session_id": new_session_id(),
        "date": now.strftime("%Y-%m-%d"),
        "subject": subject.strip() or "Untitled",
        "goal": goal.strip() or "",
        "planned_duration_min": planned_duration_min,
        "start_time": now.isoformat(timespec="seconds"),
        "end_time": "",
        "goal_completed": "",
        "self_focus_rating": "",
    }
    append_row(config.STUDY_SESSIONS_CSV, STUDY_SESSION_FIELDS, session)
    return session


def finalize_session(
    session_id: str,
    goal_completed: bool,
    self_focus_rating: int | str,
) -> None:
    rows = _read_all(config.STUDY_SESSIONS_CSV, STUDY_SESSION_FIELDS)
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        if row["session_id"] == session_id:
            row["end_time"] = now
            row["goal_completed"] = "yes" if goal_completed else "no"
            row["self_focus_rating"] = str(self_focus_rating)
            break
    _write_all(config.STUDY_SESSIONS_CSV, STUDY_SESSION_FIELDS, rows)


def append_focus_event(
    session_id: str,
    raw_label: str,
    smoothed_label: str,
    confidence: float,
    face_detected: bool,
) -> None:
    append_row(
        config.FOCUS_EVENTS_CSV,
        FOCUS_EVENT_FIELDS,
        {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw_label": raw_label,
            "smoothed_label": smoothed_label,
            "confidence": f"{confidence:.4f}",
            "face_detected": "yes" if face_detected else "no",
        },
    )


def append_feedback(
    session_id: str,
    faced_distractions: bool,
    distraction_notes: str,
    self_focus_rating: int | str,
    goal_completed: bool,
) -> None:
    append_row(
        config.STUDENT_FEEDBACK_CSV,
        FEEDBACK_FIELDS,
        {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "faced_distractions": "yes" if faced_distractions else "no",
            "distraction_notes": distraction_notes.strip(),
            "self_focus_rating": str(self_focus_rating),
            "goal_completed": "yes" if goal_completed else "no",
        },
    )


def list_sessions() -> list[dict[str, str]]:
    ensure_data_dirs()
    return _read_all(config.STUDY_SESSIONS_CSV, STUDY_SESSION_FIELDS)


def _read_all(path: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: row.get(key, "") for key in fieldnames} for row in reader]


def _write_all(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
