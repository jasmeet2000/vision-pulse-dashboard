"""
VisionPulse Dashboard — Control Panel (Left Panel)

Provides user controls for the application:
    - Start Webcam button
    - Load Video button
    - Stop Stream button
    - Confidence threshold slider with live value display
    - Placeholder buttons for future features

Emits Qt signals for all user actions — the ``DashboardWindow``
connects these to the ``VideoWorker``.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppSettings

_settings = AppSettings()


class ControlPanel(QWidget):
    """
    Left sidebar containing stream controls and settings.

    Signals
    -------
    start_webcam_clicked()
        Emitted when the user clicks "Start Webcam".
    load_video_clicked()
        Emitted when the user clicks "Load Video".
    stop_clicked()
        Emitted when the user clicks "Stop Stream".
    confidence_changed(float)
        Emitted when the confidence slider value changes.
    """

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    start_webcam_clicked = Signal()
    load_video_clicked = Signal()
    stop_clicked = Signal()
    confidence_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the control panel layout."""
        self.setFixedWidth(_settings.ui.left_panel_width)

        # Main container frame (for QSS styling)
        container = QFrame()
        container.setObjectName("panel_left")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(10)

        # ---- Title ----
        title = QLabel("Controls")
        title.setObjectName("lbl_section_title")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        layout.addWidget(self._create_separator())

        # ---- Stream Controls ----
        self.btn_start_webcam = QPushButton("Start Webcam")
        self.btn_start_webcam.setObjectName("btn_start_webcam")
        self.btn_start_webcam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_webcam.setToolTip("Open the default camera and begin detection")
        layout.addWidget(self.btn_start_webcam)

        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.setObjectName("btn_load_video")
        self.btn_load_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_video.setToolTip("Load a video file for detection")
        layout.addWidget(self.btn_load_video)

        self.btn_stop = QPushButton("Stop Stream")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setToolTip("Stop the current video stream")
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        layout.addSpacing(8)
        layout.addWidget(self._create_separator())
        layout.addSpacing(4)

        # ---- Confidence Slider ----
        conf_title = QLabel("Confidence Threshold")
        conf_title.setObjectName("lbl_panel_title")
        layout.addWidget(conf_title)

        # Slider
        self.slider_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_confidence.setRange(0, 100)
        self.slider_confidence.setValue(int(_settings.model.default_confidence * 100))
        self.slider_confidence.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_confidence.setTickInterval(10)
        layout.addWidget(self.slider_confidence)

        # Value display
        self.lbl_confidence_value = QLabel(
            f"{_settings.model.default_confidence:.2f}"
        )
        self.lbl_confidence_value.setObjectName("lbl_confidence_value")
        self.lbl_confidence_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_confidence_value)

        layout.addSpacing(8)
        layout.addWidget(self._create_separator())
        layout.addSpacing(4)

        # ---- Future Features (Placeholders) ----
        future_title = QLabel("Coming Soon")
        future_title.setObjectName("lbl_panel_title")
        layout.addWidget(future_title)

        placeholder_names = ["Record", "Screenshot", "Settings"]
        for name in placeholder_names:
            btn = QPushButton(name)
            btn.setObjectName("btn_placeholder")
            btn.setEnabled(False)
            btn.setToolTip("Feature coming in a future release")
            layout.addWidget(btn)

        # Push everything to the top
        layout.addStretch()

        # ---- Status indicator ----
        self.lbl_status = QLabel("● Idle")
        self.lbl_status.setObjectName("lbl_stat_label")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_separator() -> QFrame:
        """Create a horizontal separator line."""
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        return sep

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire internal widget signals to panel-level signals."""
        self.btn_start_webcam.clicked.connect(self.start_webcam_clicked.emit)
        self.btn_load_video.clicked.connect(self.load_video_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.slider_confidence.valueChanged.connect(self._on_confidence_changed)

    def _on_confidence_changed(self, value: int) -> None:
        """Convert integer slider value (0–100) to float (0.0–1.0)."""
        conf = value / 100.0
        self.lbl_confidence_value.setText(f"{conf:.2f}")
        self.confidence_changed.emit(conf)

    # ------------------------------------------------------------------
    # Public state management
    # ------------------------------------------------------------------

    def set_streaming_state(self, is_streaming: bool) -> None:
        """
        Update button enabled states based on streaming status.

        Parameters
        ----------
        is_streaming:
            ``True`` when a stream is active, ``False`` when idle.
        """
        self.btn_start_webcam.setEnabled(not is_streaming)
        self.btn_load_video.setEnabled(not is_streaming)
        self.btn_stop.setEnabled(is_streaming)

        if is_streaming:
            self.lbl_status.setText("● Streaming")
            self.lbl_status.setStyleSheet("color: #3fb950;")
        else:
            self.lbl_status.setText("● Idle")
            self.lbl_status.setStyleSheet("color: #8b949e;")
