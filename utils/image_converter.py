"""
VisionPulse Dashboard — Image Conversion Utilities

Pure, stateless functions that handle the OpenCV ↔ Qt image pipeline:

    OpenCV BGR ndarray
        ↓  bgr_to_rgb()
    RGB ndarray
        ↓  frame_to_qimage()
    QImage
        ↓  qimage_to_qpixmap()
    QPixmap  → ready for QLabel.setPixmap()

Keeping these conversions in one place avoids scattered format logic
and makes it trivial to unit-test the pipeline.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to RGB color space.

    Parameters
    ----------
    frame:
        A (H, W, 3) uint8 ndarray in BGR order.

    Returns
    -------
    np.ndarray
        The same frame in RGB order (contiguous memory).
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """
    Convert an RGB numpy frame to a ``QImage``.

    The returned ``QImage`` **copies** the pixel data so the original
    ndarray can be safely reused or freed.

    Parameters
    ----------
    frame:
        A (H, W, 3) uint8 ndarray in **RGB** order.

    Returns
    -------
    QImage
        A ``QImage`` in ``Format_RGB888`` format.
    """
    # Ensure contiguous memory layout — required by QImage constructor
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)

    height, width, channels = frame.shape
    bytes_per_line = channels * width

    # QImage constructor takes a *copy* when we pass bytes(frame.data)
    # but the faster path is to pass the buffer and then .copy()
    qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return qimage.copy()  # Detach from numpy memory


def qimage_to_qpixmap(
    qimage: QImage,
    target_size: Optional[Tuple[int, int]] = None,
) -> QPixmap:
    """
    Convert a ``QImage`` to a ``QPixmap``, optionally scaling to fit.

    Parameters
    ----------
    qimage:
        The source image.
    target_size:
        If provided, a ``(width, height)`` tuple.  The pixmap is scaled
        to fit within this bounding box while preserving aspect ratio.

    Returns
    -------
    QPixmap
        A display-ready pixmap.
    """
    pixmap = QPixmap.fromImage(qimage)

    if target_size is not None:
        pixmap = pixmap.scaled(
            QSize(*target_size),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    return pixmap


def draw_detections(
    frame: np.ndarray,
    boxes: list,
    confidences: list,
    class_names: list,
    color: Tuple[int, int, int] = (0, 255, 127),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and confidence scores on a frame.

    Parameters
    ----------
    frame:
        BGR ndarray to annotate (modified in place).
    boxes:
        List of ``[x1, y1, x2, y2]`` bounding boxes (pixel coords).
    confidences:
        Confidence score for each detection.
    class_names:
        Human-readable class name for each detection.
    color:
        BGR color for boxes and text background.
    thickness:
        Line thickness for bounding boxes.
    font_scale:
        Font scale for labels.

    Returns
    -------
    np.ndarray
        The annotated frame (same reference as input).
    """
    for box, conf, name in zip(boxes, confidences, class_names):
        x1, y1, x2, y2 = map(int, box)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label with background
        label = f"{name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Filled rectangle behind text for readability
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text on colored background
            1,
            cv2.LINE_AA,
        )

    return frame
