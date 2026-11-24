"""Fetch and cache public study-habit comparison data.

Owned by Author #1: Vaishnavi Panchal.
Author #2 (Ara) consumes the resulting CSV for comparison charts/suggestions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

import config

_FALLBACK_ROWS = [
    {"source": "fallback", "metric": "avg_focus_percent", "value": 62.0, "group": "college"},
    {"source": "fallback", "metric": "avg_session_minutes", "value": 45.0, "group": "college"},
    {"source": "fallback", "metric": "avg_distractions_per_hour", "value": 4.5, "group": "college"},
    {"source": "fallback", "metric": "avg_focus_percent", "value": 55.0, "group": "high_school"},
    {"source": "fallback", "metric": "avg_session_minutes", "value": 30.0, "group": "high_school"},
    {"source": "fallback", "metric": "avg_distractions_per_hour", "value": 6.0, "group": "high_school"},
    {"source": "fallback", "metric": "avg_focus_percent", "value": 70.0, "group": "graduate"},
    {"source": "fallback", "metric": "avg_session_minutes", "value": 60.0, "group": "graduate"},
    {"source": "fallback", "metric": "avg_distractions_per_hour", "value": 3.2, "group": "graduate"},
]


def ensure_external_data(force_refresh: bool = False) -> Path:
    """Download/normalize public data into data/external_study_data.csv."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EXTERNAL_STUDY_CSV
    if path.exists() and path.stat().st_size > 0 and not force_refresh:
        return path

    try:
        response = requests.get(config.EXTERNAL_DATA_URL, timeout=20)
        response.raise_for_status()
        raw_path = config.DATA_DIR / "_raw_external.csv"
        raw_path.write_bytes(response.content)
        df = pd.read_csv(raw_path)
        _normalize_public_csv(df).to_csv(path, index=False)
        return path
    except Exception:
        pd.DataFrame(_FALLBACK_ROWS).to_csv(path, index=False)
        return path


def _normalize_public_csv(df: pd.DataFrame) -> pd.DataFrame:
    rows = list(_FALLBACK_ROWS)
    if "Unemployment_rate" in df.columns:
        mean_unemp = float(pd.to_numeric(df["Unemployment_rate"], errors="coerce").mean())
        proxy_focus = max(40.0, min(85.0, 100.0 - mean_unemp * 100.0))
        rows.append(
            {
                "source": "fivethirtyeight_recent_grads",
                "metric": "avg_focus_percent",
                "value": round(proxy_focus, 1),
                "group": "recent_grads_proxy",
            }
        )
    return pd.DataFrame(rows)


def load_external_benchmarks() -> pd.DataFrame:
    path = ensure_external_data()
    return pd.read_csv(path)
