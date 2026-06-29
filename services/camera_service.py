"""
VisionPulse Dashboard — Camera Service

Encapsulates webcam I/O via OpenCV's ``VideoCapture``.

Responsibilities:
    - Open / close a webcam device by integer ID
    - Read frames
    - Report camera state
    - Handle missing-camera and disconnection errors gracefully

This service is consumed by ``VideoWorker``; it has no Qt dependency.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class CameraService:
    """
    Manages a single webcam capture device.

    Parameters
    ----------
    device_id:
        Integer index of the camera (default ``0`` = first camera).
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id: int = device_id
        self._capture: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, device_id: Optional[int] = None) -> bool:
        """
        Open the camera device.

        Parameters
        ----------
        device_id:
            Override the default device ID for this session.

        Returns
        -------
        bool
            ``True`` if the camera was opened successfully.
        """
        target_ids = [device_id] if device_id is not None else [0, 1, 2, 3, 4]
        self.release()  # Close any existing capture first

        for d_id in target_ids:
            self._device_id = d_id
            
            logger.info("Trying camera device %d (CAP_DSHOW) …", self._device_id)
            self._capture = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)
    
            if self._capture.isOpened():
                break
                
            logger.warning("Device %d: CAP_DSHOW failed. Trying default backend…", self._device_id)
            self._capture.release()
            
            self._capture = cv2.VideoCapture(self._device_id)
            if self._capture.isOpened():
                break
                
            self._capture.release()
            self._capture = None

        if self._capture is None or not self._capture.isOpened():
            logger.error("Failed to open ANY camera device from %s", target_ids)
            self._capture = None
            return False

        logger.info(
            "Camera %d opened — resolution: %.0f×%.0f",
            self._device_id,
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        return True

    def release(self) -> None:
        """Release the camera device if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera device %d released", self._device_id)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            ``(True, frame)`` on success, ``(False, None)`` on failure.
        """
        if self._capture is None or not self._capture.isOpened():
            return False, None

        ret, frame = self._capture.read()
        if not ret:
            logger.warning("Camera read failed (device %d)", self._device_id)
            return False, None

        return True, frame

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_opened(self) -> bool:
        """Return ``True`` if the camera device is currently open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def device_id(self) -> int:
        """Currently configured device ID."""
        return self._device_id

    def get_resolution(self) -> Tuple[int, int]:
        """Return ``(width, height)`` of the camera, or ``(0, 0)``."""
        if self._capture is None:
            return 0, 0
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
