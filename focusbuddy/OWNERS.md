# Module ownership

## Author #1 — Vaishnavi Panchal (implemented)

- `main.py`, `config.py`, `requirements.txt`
- `focusbuddy/data/storage.py` — session / focus event / feedback CSVs
- `focusbuddy/data/external.py` — fetch/cache external study data
- `focusbuddy/cv/camera.py` — webcam
- `focusbuddy/cv/face_detector.py` — face present vs away
- `focusbuddy/cv/emotion_model.py` — HF model inference
- `focusbuddy/cv/focus_pipeline.py` — live sampling loop + threshold gating
- `focusbuddy/ui/main_window.py` — start/pause/end session UI

## Author #2 — Aradhana (Ara) Anandkumar (to implement / extend)

- `focusbuddy/cv/labeling.py` — **stub in place**; own mapping + smoothing + calibration
- Results window (previous sessions, filters, metrics)
- Pandas stats, matplotlib charts, export
- External comparison charts + habit suggestions
- Demo validation of model label quality

### Integration contract

Vaishnavi's pipeline calls:

```python
from focusbuddy.cv.labeling import map_to_focus_label, LabelSmoother
```

Ara should keep those names/signatures stable while improving the logic.
