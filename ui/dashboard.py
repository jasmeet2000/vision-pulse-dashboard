"""
VisionPulse Dashboard — Main Dashboard Window

The ``DashboardWindow`` is the top-level ``QMainWindow`` that:
    - Assembles the three-panel layout (control, video, analytics)
    - Owns the ``VideoWorker`` lifecycle
    - Connects all signals and slots
    - Manages the YOLO detector instance (loaded once)
    - Handles error dialogs and cleanup on shutdown

This is the only module that bridges the UI, worker, and model layers.
"""

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QWidget,
)

from config.settings import AppSettings
from models.detector import YoloDetector
from ui.analytics_panel import AnalyticsPanel
from ui.control_panel import ControlPanel
from ui.video_panel import VideoPanel
from utils.logger import get_logger
from workers.video_worker import VideoWorker

logger = get_logger(__name__)
_settings = AppSettings()


class DashboardWindow(QMainWindow):
    """
    Main application window.

    Owns:
        - The YOLO detector (singleton, loaded once)
        - The VideoWorker thread
        - All three UI panels
    """

    def __init__(self) -> None:
        super().__init__()
        self._detector: Optional[YoloDetector] = None
        self._worker: Optional[VideoWorker] = None
        self._is_streaming: bool = False

        self._init_window()
        self._init_panels()
        self._connect_signals()

        # Defer model loading so the window appears instantly
        QTimer.singleShot(100, self._load_model)

        logger.info("Dashboard window initialized")

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _init_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle(_settings.ui.window_title)
        self.setMinimumSize(
            _settings.ui.window_min_width,
            _settings.ui.window_min_height,
        )
        self.resize(_settings.ui.default_width, _settings.ui.default_height)

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def _init_panels(self) -> None:
        """Create and arrange the three-panel layout."""
        # Panels
        self._control_panel = ControlPanel()
        self._video_panel = VideoPanel()
        self._analytics_panel = AnalyticsPanel()

        # Splitter for responsive resizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._control_panel)
        splitter.addWidget(self._video_panel)
        splitter.addWidget(self._analytics_panel)

        # Set proportional sizes — center panel gets the most space
        splitter.setStretchFactor(0, 0)  # Left: fixed width
        splitter.setStretchFactor(1, 1)  # Center: stretches
        splitter.setStretchFactor(2, 0)  # Right: fixed width

        # Force initial sizes so the center panel expands and pushes the right panel
        splitter.setSizes([260, 2000, 320])

        # Central widget
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire control panel signals to handler methods."""
        self._control_panel.start_webcam_clicked.connect(self._on_start_webcam)
        self._control_panel.load_video_clicked.connect(self._on_load_video)
        self._control_panel.stop_clicked.connect(self._on_stop)
        self._control_panel.confidence_changed.connect(self._on_confidence_changed)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the YOLO model asynchronously after the window is shown.

        Displays a loading state and shows an error dialog on failure.
        """
        self._control_panel.lbl_status.setText("● Loading model…")
        self._control_panel.lbl_status.setStyleSheet("color: #d29922;")
        self._control_panel.btn_start_webcam.setEnabled(False)
        self._control_panel.btn_load_video.setEnabled(False)
        QApplication.processEvents()  # Force UI update before blocking load

        try:
            self._detector = YoloDetector(
                weights_path=_settings.model.weights_path,
                confidence=_settings.model.default_confidence,
                device=_settings.model.device,
            )
            logger.info("Model loaded: %s", self._detector.model_name)

            self._control_panel.lbl_status.setText("● Ready")
            self._control_panel.lbl_status.setStyleSheet("color: #3fb950;")
            self._control_panel.btn_start_webcam.setEnabled(True)
            self._control_panel.btn_load_video.setEnabled(True)

        except Exception as exc:
            logger.exception("Failed to load detection model")
            self._control_panel.lbl_status.setText("● Model Error")
            self._control_panel.lbl_status.setStyleSheet("color: #f85149;")

            QMessageBox.critical(
                self,
                "Model Loading Error",
                f"Failed to load the detection model.\n\n"
                f"Error: {exc}\n\n"
                f"Please ensure 'ultralytics' is installed and an internet "
                f"connection is available for the initial weight download.",
            )

    # ------------------------------------------------------------------
    # Stream control handlers
    # ------------------------------------------------------------------

    def _on_start_webcam(self) -> None:
        """Start the webcam stream."""
        if self._detector is None:
            QMessageBox.warning(self, "Not Ready", "The detection model is still loading.")
            return

        logger.info("Starting webcam stream")
        self._start_worker()
        self._worker.start_camera(_settings.video.default_camera_id)

    def _on_load_video(self) -> None:
        """Open a file dialog and start video playback."""
        if self._detector is None:
            QMessageBox.warning(self, "Not Ready", "The detection model is still loading.")
            return

        # Build filter string from supported formats
        formats = " ".join(_settings.video.supported_formats)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            f"Video Files ({formats});;All Files (*)",
        )

        if not path:
            return  # User cancelled

        logger.info("Loading video: %s", path)
        self._start_worker()
        self._worker.start_video(path)

    def _on_stop(self) -> None:
        """Stop the current stream."""
        logger.info("Stopping stream")
        self._stop_worker()

    def _on_confidence_changed(self, value: float) -> None:
        """Update the detector's confidence threshold in real time."""
        if self._detector is not None:
            self._detector.confidence = value

        self._analytics_panel.update_confidence_display(value)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Create a fresh worker, connect its signals, and update UI state."""
        # Stop any existing worker first
        self._stop_worker()

        self._worker = VideoWorker(self._detector)

        # Connect worker signals to UI slots
        self._worker.frame_ready.connect(self._video_panel.update_frame)
        self._worker.stats_ready.connect(self._analytics_panel.update_stats)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)

        self._is_streaming = True
        self._control_panel.set_streaming_state(True)
        self._analytics_panel.reset()

    def _stop_worker(self) -> None:
        """Stop and clean up the worker thread."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()

        self._worker = None
        self._is_streaming = False
        self._control_panel.set_streaming_state(False)
        self._video_panel.clear()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_worker_error(self, message: str) -> None:
        """Display an error dialog when the worker encounters an issue."""
        logger.error("Worker error: %s", message)
        QMessageBox.warning(self, "Stream Error", message)
        self._stop_worker()

    def _on_worker_finished(self) -> None:
        """Handle clean worker shutdown (e.g., video file ended)."""
        logger.info("Worker finished")
        if self._is_streaming:
            self._is_streaming = False
            self._control_panel.set_streaming_state(False)

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """
        Ensure the worker thread is stopped before the window closes.

        This prevents dangling threads and OpenCV resource leaks.
        """
        logger.info("Application closing — shutting down worker")
        self._stop_worker()
        event.accept()
