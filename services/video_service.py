"""
VisionPulse Dashboard — Video File Service

Encapsulates video-file I/O via OpenCV's ``VideoCapture``.

Responsibilities:
    - Open / close a video file by path
    - Read frames sequentially
    - Report metadata (FPS, frame count, duration)
    - Handle corrupt or missing files gracefully

This service is consumed by ``VideoWorker``; it has no Qt dependency.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoService:
    """
    Manages playback of a single video file.

    Parameters
    ----------
    path:
        Filesystem path to the video file.  Can be set later via ``open()``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path: Optional[str] = path
        self._capture: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, path: str) -> bool:
        """
        Open a video file for sequential reading.

        Parameters
        ----------
        path:
            Absolute or relative path to the video file.

        Returns
        -------
        bool
            ``True`` if the file was opened successfully.
        """
        self.release()

        resolved = Path(path).resolve()
        if not resolved.is_file():
            logger.error("Video file not found: %s", resolved)
            return False

        logger.info("Opening video file: %s", resolved)
        self._path = str(resolved)
        self._capture = cv2.VideoCapture(self._path)

        if not self._capture.isOpened():
            logger.error("Failed to open video file: %s", self._path)
            self._capture = None
            return False

        logger.info(
            "Video opened — %d frames, %.1f FPS, %.0f×%.0f",
            self.get_frame_count(),
            self.get_fps(),
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        return True

    def release(self) -> None:
        """Release the video capture if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Video file released: %s", self._path)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            ``(True, frame)`` on success, ``(False, None)`` at EOF or error.
        """
        if self._capture is None or not self._capture.isOpened():
            return False, None

        ret, frame = self._capture.read()
        if not ret:
            logger.info("Video playback ended: %s", self._path)
            return False, None

        return True, frame

    # ------------------------------------------------------------------
    # State / metadata queries
    # ------------------------------------------------------------------

    def is_opened(self) -> bool:
        """Return ``True`` if a video file is currently open."""
        return self._capture is not None and self._capture.isOpened()

    def get_fps(self) -> float:
        """Return the native FPS of the video file, or ``0.0``."""
        if self._capture is None:
            return 0.0
        return self._capture.get(cv2.CAP_PROP_FPS) or 0.0

    def get_frame_count(self) -> int:
        """Return the total frame count, or ``0``."""
        if self._capture is None:
            return 0
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_duration_seconds(self) -> float:
        """Return approximate video duration in seconds."""
        fps = self.get_fps()
        if fps <= 0:
            return 0.0
        return self.get_frame_count() / fps

    @property
    def path(self) -> Optional[str]:
        """Currently loaded file path."""
        return self._path
