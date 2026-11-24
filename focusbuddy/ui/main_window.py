"""FocusBuddy UI: home page → live study session page.

Owned by Author #1: Vaishnavi Panchal.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

import config
from focusbuddy.cv.focus_pipeline import FocusPipeline
from focusbuddy.data import storage
from focusbuddy.data.external import ensure_external_data

FOCUS_COLORS = {
    "focused": "#2a9d8f",
    "distracted": "#e76f51",
    "away": "#6c757d",
    "paused": "#adb5bd",
    "—": "#6c757d",
}

BG = "#f4f6f8"
CARD = "#ffffff"
INK = "#101828"
MUTED = "#667085"
ACCENT = "#0f766e"


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(config.APP_NAME)
        self.geometry("1000x680")
        self.minsize(920, 620)
        self.configure(bg=BG)

        storage.ensure_data_dirs()

        self.session: Optional[dict] = None
        self.elapsed_seconds = 0
        self.running = False
        self.pipeline = FocusPipeline(on_sample=self._on_focus_sample)
        self._pipeline_ready = False
        self._timer_job: Optional[str] = None
        self._preview_job: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._current_ui_label = "away"
        self._stats = {
            "focused": 0,
            "distracted": 0,
            "away": 0,
            "distraction_bursts": 0,
            "prev_label": None,
        }
        self._planned_seconds = 0.0

        self._build_style()
        self.container = ttk.Frame(self, style="TFrame")
        self.container.pack(fill="both", expand=True)

        self.home_frame = ttk.Frame(self.container, style="TFrame")
        self.session_frame = ttk.Frame(self.container, style="TFrame")
        for frame in (self.home_frame, self.session_frame):
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_home()
        self._build_session()
        self.show_home()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabelframe", background=CARD)
        style.configure(
            "TLabelframe.Label",
            background=CARD,
            foreground=INK,
            font=("Helvetica", 11, "bold"),
        )
        style.configure("TLabel", background=CARD, foreground=INK, font=("Helvetica", 11))
        style.configure("Home.TLabel", background=BG, foreground=INK)
        style.configure("Brand.TLabel", background=BG, foreground=INK, font=("Helvetica", 42, "bold"))
        style.configure("Tagline.TLabel", background=BG, foreground=MUTED, font=("Helvetica", 13))
        style.configure("Timer.TLabel", background=CARD, font=("Helvetica", 30, "bold"))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 11), padding=8)
        style.configure("Accent.TButton", font=("Helvetica", 12, "bold"), padding=10)
        style.configure("TEntry", padding=4)

    # ── Home page ──────────────────────────────────────────────────────────

    def _build_home(self) -> None:
        wrap = ttk.Frame(self.home_frame, style="TFrame")
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(wrap, text=config.APP_NAME, style="Brand.TLabel").pack(anchor="center")
        ttk.Label(
            wrap,
            text="Your study buddy for focused sessions",
            style="Tagline.TLabel",
        ).pack(anchor="center", pady=(4, 22))

        card = ttk.LabelFrame(wrap, text="Create a study session", padding=18)
        card.pack(fill="x")

        ttk.Label(card, text="Subject").grid(row=0, column=0, sticky="w", pady=6)
        self.subject_var = tk.StringVar()
        ttk.Entry(card, textvariable=self.subject_var, width=36).grid(
            row=0, column=1, sticky="ew", pady=6, padx=(12, 0)
        )

        ttk.Label(card, text="Study goal").grid(row=1, column=0, sticky="w", pady=6)
        self.goal_var = tk.StringVar()
        ttk.Entry(card, textvariable=self.goal_var, width=36).grid(
            row=1, column=1, sticky="ew", pady=6, padx=(12, 0)
        )

        ttk.Label(card, text="Duration (min)").grid(row=2, column=0, sticky="w", pady=6)
        self.duration_var = tk.StringVar(value="25")
        ttk.Entry(card, textvariable=self.duration_var, width=10).grid(
            row=2, column=1, sticky="w", pady=6, padx=(12, 0)
        )
        card.columnconfigure(1, weight=1)

        ttk.Button(
            card,
            text="Start study session",
            style="Accent.TButton",
            command=self.begin_session_from_home,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 4))

        links = ttk.Frame(wrap, style="TFrame")
        links.pack(fill="x", pady=(14, 0))
        ttk.Button(
            links, text="Previous sessions", command=self.open_results_placeholder
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            links, text="Fetch external data", command=self.fetch_external_data
        ).pack(side="left")

        ttk.Label(
            wrap,
            text=config.DISCLAIMER,
            style="Tagline.TLabel",
            wraplength=520,
            justify="center",
        ).pack(pady=(18, 0))

    def show_home(self) -> None:
        self._stop_preview_loop()
        self.home_frame.tkraise()

    def show_session(self) -> None:
        self.session_frame.tkraise()
        self._schedule_preview()

    # ── Session page ───────────────────────────────────────────────────────

    def _build_session(self) -> None:
        outer = ttk.Frame(self.session_frame, style="TFrame")
        outer.pack(fill="both", expand=True, padx=16, pady=12)

        top = ttk.Frame(outer, style="TFrame")
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(
            top,
            text=config.APP_NAME,
            style="Home.TLabel",
            font=("Helvetica", 18, "bold"),
        ).pack(side="left")
        self.session_meta_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.session_meta_var, style="Tagline.TLabel").pack(
            side="left", padx=(12, 0)
        )

        body = ttk.Frame(outer, style="TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_card = ttk.LabelFrame(body, text="Live camera", padding=10)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.video_label = tk.Label(
            preview_card,
            text="Starting camera…",
            bg="#111827",
            fg="#e5e7eb",
            width=60,
            height=18,
            font=("Helvetica", 12),
        )
        self.video_label.pack(fill="both", expand=True)

        legend = ttk.Frame(preview_card)
        legend.pack(fill="x", pady=(8, 0))
        for name, color in (("focused", "#2a9d8f"), ("distracted", "#e76f51"), ("away", "#6c757d")):
            swatch = tk.Canvas(legend, width=12, height=12, highlightthickness=0, bg=CARD)
            swatch.create_rectangle(0, 0, 12, 12, fill=color, outline=color)
            swatch.pack(side="left", padx=(0, 4))
            ttk.Label(legend, text=name + "   ", style="Muted.TLabel").pack(side="left")

        right = ttk.Frame(body, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        status = ttk.LabelFrame(right, text="Live status", padding=10)
        status.pack(fill="x")

        self.timer_var = tk.StringVar(value="00:00:00")
        self.focus_var = tk.StringVar(value="—")
        self.raw_var = tk.StringVar(value="Raw: —")
        self.webcam_var = tk.StringVar(value="Webcam: —")
        self.buddy_var = tk.StringVar(value="Buddy: preparing…")
        self.confidence_var = tk.StringVar(value="Confidence: —")

        ttk.Label(status, textvariable=self.timer_var, style="Timer.TLabel").pack(anchor="w")
        self.focus_label = tk.Label(
            status,
            textvariable=self.focus_var,
            font=("Helvetica", 18, "bold"),
            bg=CARD,
            fg=FOCUS_COLORS["—"],
        )
        self.focus_label.pack(anchor="w", pady=(4, 2))
        ttk.Label(status, textvariable=self.confidence_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(status, textvariable=self.raw_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(status, textvariable=self.webcam_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(status, textvariable=self.buddy_var).pack(anchor="w", pady=(6, 0))

        stats_card = ttk.LabelFrame(right, text="Session stats", padding=10)
        stats_card.pack(fill="both", expand=True, pady=(10, 0))

        self.stat_focus_pct_var = tk.StringVar(value="Focus: 0%")
        self.stat_focus_time_var = tk.StringVar(value="Focused time: 0s")
        self.stat_distract_time_var = tk.StringVar(value="Distracted time: 0s")
        self.stat_away_time_var = tk.StringVar(value="Away time: 0s")
        self.stat_bursts_var = tk.StringVar(value="Distraction bursts: 0")
        self.stat_remaining_var = tk.StringVar(value="Time remaining: —")

        self.stat_focus_pct_label = tk.Label(
            stats_card,
            textvariable=self.stat_focus_pct_var,
            font=("Helvetica", 20, "bold"),
            bg=CARD,
            fg=ACCENT,
        )
        self.stat_focus_pct_label.pack(anchor="w", pady=(0, 8))
        for var in (
            self.stat_focus_time_var,
            self.stat_distract_time_var,
            self.stat_away_time_var,
            self.stat_bursts_var,
            self.stat_remaining_var,
        ):
            ttk.Label(stats_card, textvariable=var).pack(anchor="w", pady=2)

        controls = ttk.Frame(right, style="TFrame")
        controls.pack(fill="x", pady=(12, 0))
        self.pause_btn = ttk.Button(controls, text="Pause", command=self.pause_session)
        self.resume_btn = ttk.Button(
            controls, text="Resume", command=self.resume_session, state="disabled"
        )
        self.end_btn = ttk.Button(controls, text="End session", command=self.end_session)
        self.pause_btn.pack(fill="x", pady=3)
        self.resume_btn.pack(fill="x", pady=3)
        self.end_btn.pack(fill="x", pady=3)

    # ── Navigation / session lifecycle ─────────────────────────────────────

    def begin_session_from_home(self) -> None:
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(config.APP_NAME, "Enter a positive duration in minutes.")
            return

        subject = self.subject_var.get().strip() or "Untitled"
        goal = self.goal_var.get().strip()
        self.session_meta_var.set(f"{subject} · {duration:g} min")
        self._planned_seconds = duration * 60.0
        self._reset_live_stats()
        self.show_session()
        self.buddy_var.set("Buddy: loading camera & model…")
        self.webcam_var.set("Webcam: starting…")
        self.update_idletasks()

        if not self._ensure_pipeline():
            messagebox.showerror(
                config.APP_NAME,
                "Could not open the camera. Return home and check permissions.",
            )
            self.show_home()
            return

        self.session = storage.create_session(
            subject=subject,
            goal=goal,
            planned_duration_min=duration,
        )
        self.elapsed_seconds = 0
        self.pipeline.confidence_threshold = config.load_confidence_threshold()
        self.pipeline.start()
        self.running = True
        self.pause_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.end_btn.configure(state="normal")
        self.buddy_var.set("Buddy: present — studying with you")
        self._refresh_stats_labels()
        self._tick_timer()

    def _ensure_pipeline(self) -> bool:
        if self._pipeline_ready and self.pipeline.camera.status == "open":
            return True
        ok = self.pipeline.prepare()
        self._pipeline_ready = bool(ok and self.pipeline.camera.status == "open")
        if self.pipeline.camera.status == "open":
            self.webcam_var.set("Webcam: ready")
        else:
            self.webcam_var.set(f"Webcam: {self.pipeline.camera.status}")
        if not self.pipeline.model_ready:
            messagebox.showwarning(
                config.APP_NAME,
                "Emotion model did not load. Face presence can still be tracked.\n\n"
                f"Details: {self.pipeline.emotion_model.load_error}",
            )
        return self._pipeline_ready

    def pause_session(self) -> None:
        if not self.session or not self.running:
            return
        self.running = False
        self.pipeline.pause()
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None
        self.pause_btn.configure(state="disabled")
        self.resume_btn.configure(state="normal")
        self.buddy_var.set("Buddy: paused — resume when ready")
        self.focus_var.set("PAUSED")
        self.focus_label.configure(fg=FOCUS_COLORS["paused"])

    def resume_session(self) -> None:
        if not self.session or self.running:
            return
        self.pipeline.resume()
        self.running = True
        self.pause_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.buddy_var.set("Buddy: present — studying with you")
        self._tick_timer()

    def end_session(self) -> None:
        if not self.session:
            self.show_home()
            return

        self.running = False
        self.pipeline.pause()
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None

        feedback = self._collect_feedback()
        if feedback is None:
            # Cancelled end → stay on session page paused
            self.pause_btn.configure(state="disabled")
            self.resume_btn.configure(state="normal")
            self.focus_var.set("PAUSED")
            self.focus_label.configure(fg=FOCUS_COLORS["paused"])
            return

        summary = (
            f"focus={self._focus_percent():.0f}%; "
            f"bursts={self._stats['distraction_bursts']}; "
            f"away={self._format_seconds(self._stats['away'])}"
        )
        storage.finalize_session(
            self.session["session_id"],
            goal_completed=feedback["goal_completed"],
            self_focus_rating=feedback["self_focus_rating"],
        )
        storage.append_feedback(
            session_id=self.session["session_id"],
            faced_distractions=feedback["faced_distractions"],
            distraction_notes=summary,
            self_focus_rating=feedback["self_focus_rating"],
            goal_completed=feedback["goal_completed"],
        )

        messagebox.showinfo(
            config.APP_NAME,
            f"Session saved.\n\nFocus {self._focus_percent():.0f}% · "
            f"{self._stats['distraction_bursts']} distraction burst(s).",
        )
        self._reset_session_state()
        self.show_home()

    def _reset_live_stats(self) -> None:
        self._stats = {
            "focused": 0,
            "distracted": 0,
            "away": 0,
            "distraction_bursts": 0,
            "prev_label": None,
        }
        self._current_ui_label = "away"
        self._refresh_stats_labels()

    def _reset_session_state(self) -> None:
        self.session = None
        self.elapsed_seconds = 0
        self.running = False
        self.timer_var.set("00:00:00")
        self.focus_var.set("—")
        self.focus_label.configure(fg=FOCUS_COLORS["—"])
        self.confidence_var.set("Confidence: —")
        self.raw_var.set("Raw: —")
        self.buddy_var.set("Buddy: waiting to start")
        self.pause_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.video_label.configure(image="", text="Camera idle")
        self._photo = None
        self._reset_live_stats()

    @staticmethod
    def _format_seconds(total: int) -> str:
        total = int(total)
        minutes, seconds = divmod(total, 60)
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _focus_percent(self) -> float:
        tracked = self._stats["focused"] + self._stats["distracted"] + self._stats["away"]
        if tracked <= 0:
            return 0.0
        return 100.0 * self._stats["focused"] / tracked

    def _refresh_stats_labels(self) -> None:
        pct = self._focus_percent()
        self.stat_focus_pct_var.set(f"Focus: {pct:.0f}%")
        self.stat_focus_time_var.set(
            f"Focused time: {self._format_seconds(self._stats['focused'])}"
        )
        self.stat_distract_time_var.set(
            f"Distracted time: {self._format_seconds(self._stats['distracted'])}"
        )
        self.stat_away_time_var.set(
            f"Away time: {self._format_seconds(self._stats['away'])}"
        )
        self.stat_bursts_var.set(
            f"Distraction bursts: {self._stats['distraction_bursts']}"
        )
        if self._planned_seconds > 0:
            remaining = max(0, int(self._planned_seconds - self.elapsed_seconds))
            self.stat_remaining_var.set(
                f"Time remaining: {self._format_seconds(remaining)}"
            )
        else:
            self.stat_remaining_var.set("Time remaining: —")

    def _accumulate_second(self, label: str) -> None:
        if label not in ("focused", "distracted", "away"):
            return
        prev = self._stats["prev_label"]
        if label == "distracted" and prev != "distracted":
            self._stats["distraction_bursts"] += 1
        self._stats[label] += 1
        self._stats["prev_label"] = label
        self._refresh_stats_labels()

    def _collect_feedback(self) -> Optional[dict]:
        goal_completed = messagebox.askyesno(
            config.APP_NAME, "Did you complete your study goal?"
        )
        rating = simpledialog.askinteger(
            config.APP_NAME,
            "Self-reported focus (1–5):",
            minvalue=1,
            maxvalue=5,
            parent=self,
        )
        if rating is None:
            return None
        faced = messagebox.askyesno(
            config.APP_NAME, "Did you face distractions this session?"
        )
        return {
            "goal_completed": bool(goal_completed),
            "self_focus_rating": int(rating),
            "faced_distractions": bool(faced),
            "distraction_notes": "",
        }

    # ── Preview / timer ────────────────────────────────────────────────────

    def _schedule_preview(self) -> None:
        self._stop_preview_loop()
        self._update_preview()
        self._preview_job = self.after(80, self._schedule_preview)

    def _stop_preview_loop(self) -> None:
        if self._preview_job:
            try:
                self.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None

    def _update_preview(self) -> None:
        frame = self.pipeline.get_preview_rgb()
        if frame is not None:
            image = Image.fromarray(frame.astype(np.uint8))
            self._photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self._photo, text="")

        sample = self.pipeline.last_sample
        if sample is None or not self.session:
            return

        # While paused, keep the PAUSED badge and still refresh the camera feed.
        if not self.running:
            self.focus_var.set("PAUSED")
            self.focus_label.configure(fg=FOCUS_COLORS["paused"])
            self.webcam_var.set(f"Webcam: {self.pipeline.camera.status}")
            return

        if not sample.face_detected:
            ui_label = "away"
        elif sample.raw_label == "distracted":
            ui_label = "distracted"
        else:
            ui_label = sample.smoothed_label

        self._current_ui_label = ui_label
        self.focus_var.set(ui_label.upper())
        self.focus_label.configure(fg=FOCUS_COLORS.get(ui_label, FOCUS_COLORS["—"]))
        fer = getattr(sample, "fer_label", "") or "—"
        self.confidence_var.set(f"Confidence: {sample.confidence:.2f} · expression: {fer}")
        self.raw_var.set(
            f"Raw: {sample.raw_label} · face={'yes' if sample.face_detected else 'no'}"
        )
        self.webcam_var.set(f"Webcam: {self.pipeline.camera.status}")
        if ui_label == "focused":
            self.buddy_var.set("Buddy: present — you look engaged")
        elif ui_label == "distracted":
            self.buddy_var.set("Buddy: still here — gentle nudge to refocus")
        else:
            self.buddy_var.set("Buddy: waiting for you to return")

    def _tick_timer(self) -> None:
        if not self.running:
            return
        hours, rem = divmod(self.elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self.timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self._accumulate_second(self._current_ui_label)

        if self._planned_seconds and self.elapsed_seconds >= self._planned_seconds:
            messagebox.showinfo(
                config.APP_NAME,
                "Planned duration reached. Pause or end when you're ready.",
            )
            self.pause_session()
            return

        self.elapsed_seconds += 1
        self._timer_job = self.after(1000, self._tick_timer)

    def _on_focus_sample(self, sample: dict) -> None:
        if not self.running or not self.session:
            return
        storage.append_focus_event(
            session_id=self.session["session_id"],
            raw_label=sample["raw_label"],
            smoothed_label=sample["smoothed_label"],
            confidence=sample["confidence"],
            face_detected=bool(sample["face_detected"]),
        )

    def open_results_placeholder(self) -> None:
        messagebox.showinfo(
            config.APP_NAME,
            "Previous sessions / analysis UI is Author #2 (Ara)'s portion.\n\n"
            "Session data is already being saved under data/ for that work.",
        )

    def fetch_external_data(self) -> None:
        try:
            path = ensure_external_data(force_refresh=True)
            messagebox.showinfo(
                config.APP_NAME,
                f"External study data ready at:\n{path}",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(config.APP_NAME, f"Could not fetch external data:\n{exc}")

    def _on_close(self) -> None:
        self._stop_preview_loop()
        if self._timer_job:
            try:
                self.after_cancel(self._timer_job)
            except Exception:
                pass
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.destroy()
