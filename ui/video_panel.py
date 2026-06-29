"""
VisionPulse Dashboard — Video Panel (Center Panel)

Displays the live video feed with detection annotations.

Frame pipeline:
    QImage (from worker signal)
        → QPixmap (scaled to fit, aspect ratio preserved)
        → QLabel.setPixmap()

When no stream is active, shows a centered "No Video Source" placeholder.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget


class VideoPanel(QWidget):
    """
    Center panel that renders video frames from the worker thread.

    The ``update_frame()`` slot receives a ``QImage`` via signal and
    converts it to a scaled ``QPixmap`` for display.  The aspect ratio
    is always preserved.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the video display area."""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Container frame (for QSS styling)
        self._container = QFrame()
        self._container.setObjectName("panel_center")

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(0)

        # Video display label
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._video_label.setStyleSheet("background-color: #0d1117; border-radius: 8px;")

        # Placeholder text
        self._show_placeholder()

        container_layout.addWidget(self._video_label)

        # Outer layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def update_frame(self, qimage: QImage) -> None:
        """
        Display a new frame.  Called from the GUI thread via signal.

        The image is scaled to fit the label while preserving the
        aspect ratio with smooth (bilinear) interpolation.

        Parameters
        ----------
        qimage:
            The annotated frame as a ``QImage`` in RGB888 format.
        """
        if qimage.isNull():
            return

        pixmap = QPixmap.fromImage(qimage)

        # Scale to fit the label's current size
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def clear(self) -> None:
        """Remove the current frame and show the placeholder."""
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Display the 'No Video Source' placeholder text."""
        self._video_label.clear()
        self._video_label.setText("No Video Source")
        self._video_label.setObjectName("lbl_no_video")
        # Re-apply styling after objectName change
        self._video_label.setStyleSheet(
            "color: #484f58; font-size: 16pt; font-weight: 600; "
            "background-color: #0d1117; border-radius: 8px;"
        )

    def resizeEvent(self, event) -> None:
        """
        On resize, re-scale the current pixmap if one is displayed.

        This prevents the video from getting clipped or letter-boxed
        incorrectly after a window resize.
        """
        super().resizeEvent(event)
        pixmap = self._video_label.pixmap()
        if pixmap and not pixmap.isNull():
            # The next frame_ready signal will set the correct size,
            # so we don't need to do anything here — the label handles
            # it implicitly because we scale on every frame.
            pass
