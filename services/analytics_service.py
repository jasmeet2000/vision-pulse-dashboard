"""
VisionPulse Dashboard — Analytics Service

Pure-Python service that accumulates real-time detection statistics
using fixed-size rolling buffers (``collections.deque``).

Responsibilities:
    - Accept per-frame updates (FPS, object count, inference time)
    - Compute rolling averages and aggregates
    - Provide a snapshot dict for the analytics panel
    - Track per-class detection counts

Design:
    No Qt dependency — this is a plain data service.
    Thread-safety is not required because the worker calls ``update()``
    sequentially and the UI reads via ``get_snapshot()`` on signal.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

from config.settings import AppSettings

_settings = AppSettings().analytics


@dataclass
class AnalyticsService:
    """
    Rolling statistics accumulator for the detection pipeline.

    Keeps the last ``max_samples`` values for each metric and
    provides computed aggregates.
    """

    max_samples: int = _settings.max_samples

    # Rolling buffers
    _fps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_settings.max_samples))
    _object_count_history: Deque[int] = field(default_factory=lambda: deque(maxlen=_settings.max_samples))
    _inference_time_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_settings.max_samples))

    # Per-class counters for the current session
    _class_counts: Dict[str, int] = field(default_factory=dict)

    # Latest values
    _current_fps: float = 0.0
    _current_object_count: int = 0
    _current_inference_ms: float = 0.0
    _current_classes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        fps: float,
        object_count: int,
        inference_ms: float,
        class_names: List[str],
    ) -> None:
        """
        Record a single frame's metrics.

        Parameters
        ----------
        fps:
            Instantaneous frames per second.
        object_count:
            Number of objects detected in this frame.
        inference_ms:
            Model inference time in milliseconds.
        class_names:
            List of class names detected in this frame.
        """
        self._current_fps = fps
        self._current_object_count = object_count
        self._current_inference_ms = inference_ms
        self._current_classes = class_names

        self._fps_history.append(fps)
        self._object_count_history.append(object_count)
        self._inference_time_history.append(inference_ms)

        self._class_counts.clear()
        for name in class_names:
            self._class_counts[name] = self._class_counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def average_fps(self) -> float:
        """Rolling average FPS over the history buffer."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    @property
    def average_inference_ms(self) -> float:
        """Rolling average inference time in milliseconds."""
        if not self._inference_time_history:
            return 0.0
        return sum(self._inference_time_history) / len(self._inference_time_history)

    @property
    def object_count_series(self) -> List[int]:
        """List of recent object counts for charting."""
        return list(self._object_count_history)

    @property
    def fps_series(self) -> List[float]:
        """List of recent FPS values for charting."""
        return list(self._fps_history)

    # ------------------------------------------------------------------
    # Snapshot for the UI
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """
        Return a dict of current analytics suitable for emitting
        via a Qt signal.

        Keys:
            current_fps, average_fps, inference_ms, object_count,
            class_names, class_counts, object_count_series, fps_series
        """
        return {
            "current_fps": round(self._current_fps, 1),
            "average_fps": round(self.average_fps, 1),
            "inference_ms": round(self._current_inference_ms, 1),
            "object_count": self._current_object_count,
            "class_names": self._current_classes,
            "class_counts": dict(self._class_counts),
            "object_count_series": self.object_count_series,
            "fps_series": self.fps_series,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._fps_history.clear()
        self._object_count_history.clear()
        self._inference_time_history.clear()
        self._class_counts.clear()
        self._current_fps = 0.0
        self._current_object_count = 0
        self._current_inference_ms = 0.0
        self._current_classes = []
