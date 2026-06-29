"""
VisionPulse Dashboard — Application Settings

Centralized configuration using frozen dataclasses.
Single source of truth for all tunables: model paths, UI defaults,
performance targets, and logging configuration.

Design:
    - Immutable defaults via frozen dataclass
    - Mutable runtime config via standard dataclass
    - All paths resolved relative to project root using pathlib
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Project root — resolved once, reused everywhere
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSettings:
    """Configuration for the AI detection model."""

    weights_path: str = "yolov8n.pt"
    default_confidence: float = 0.25
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    device: str = ""  # "" lets Ultralytics auto-select (CUDA → CPU fallback)


@dataclass(frozen=True)
class VideoSettings:
    """Configuration for video capture and display."""

    default_camera_id: int = 0
    target_fps: int = 30
    frame_skip: int = 0  # 0 = process every frame
    supported_formats: Tuple[str, ...] = (
        "*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.flv", "*.webm",
    )


@dataclass(frozen=True)
class AnalyticsSettings:
    """Configuration for the analytics accumulator."""

    max_samples: int = 100  # Rolling buffer size for charts
    fps_smoothing_window: int = 30  # Frames over which to average FPS


@dataclass(frozen=True)
class UISettings:
    """Configuration for the desktop interface."""

    window_title: str = "VisionPulse Dashboard"
    window_min_width: int = 1280
    window_min_height: int = 720
    default_width: int = 1400
    default_height: int = 800
    left_panel_width: int = 260
    right_panel_width: int = 320
    font_family: str = "Segoe UI"
    font_size: int = 10


@dataclass(frozen=True)
class LoggingSettings:
    """Configuration for the logging subsystem."""

    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    log_file: str = "visionpulse.log"
    log_level: str = "DEBUG"
    max_bytes: int = 5 * 1024 * 1024  # 5 MB per log file
    backup_count: int = 3


@dataclass(frozen=True)
class AppSettings:
    """
    Top-level aggregator of all configuration sections.

    Usage::

        from config.settings import AppSettings
        settings = AppSettings()
        print(settings.model.weights_path)
    """

    model: ModelSettings = field(default_factory=ModelSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    analytics: AnalyticsSettings = field(default_factory=AnalyticsSettings)
    ui: UISettings = field(default_factory=UISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    style_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "styles" / "dark_theme.qss"
    )
