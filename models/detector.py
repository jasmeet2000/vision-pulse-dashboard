"""
VisionPulse Dashboard — Detection Models

Defines the AI detection abstraction layer using the Strategy Pattern:

    BaseDetector (ABC)
        └── YoloDetector  — Ultralytics YOLOv8 implementation

Design rationale:
    Any future model (ONNX Runtime, TensorRT, face-recognition, pose
    estimation) only needs to subclass ``BaseDetector`` and implement
    ``detect()``.  The ``VideoWorker`` and UI layers remain untouched.

Thread-safety:
    ``confidence`` is a simple float attribute.  The worker thread reads
    it each frame, and the GUI thread writes it via the slider.  Because
    Python's GIL guarantees atomic float assignment on CPython, no lock
    is required for this single-attribute pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Detection result value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionResult:
    """
    Immutable container for a single frame's detection output.

    Attributes
    ----------
    boxes:
        List of ``[x1, y1, x2, y2]`` bounding boxes in pixel coordinates.
    confidences:
        Confidence score for each detection, range [0, 1].
    class_ids:
        Integer class ID for each detection.
    class_names:
        Human-readable class name for each detection.
    inference_time_ms:
        Time spent on model inference in milliseconds.
    """

    boxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Number of detections in this result."""
        return len(self.boxes)


# ---------------------------------------------------------------------------
# Abstract base class — Strategy interface
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Abstract interface for object detection models.

    Subclasses must implement:
        - ``detect(frame)``  → ``DetectionResult``
        - ``model_name``     → ``str``
    """

    def __init__(self, confidence: float = 0.25) -> None:
        self._confidence: float = confidence

    @property
    def confidence(self) -> float:
        """Current confidence threshold."""
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """Update confidence threshold (clamped to [0, 1])."""
        self._confidence = max(0.0, min(1.0, value))

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the loaded model."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame:
            (H, W, 3) uint8 ndarray in BGR color space.

        Returns
        -------
        DetectionResult
            Detections found in this frame.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementation — YOLOv8
# ---------------------------------------------------------------------------

class YoloDetector(BaseDetector):
    """
    YOLOv8 object detector powered by the Ultralytics library.

    The model weights are loaded **once** in ``__init__`` and reused for
    every subsequent ``detect()`` call.  The confidence threshold can be
    changed at any time via the ``confidence`` property.

    Parameters
    ----------
    weights_path:
        Path or name of the YOLO weights file (e.g. ``"yolov8n.pt"``).
        Ultralytics will auto-download if not present locally.
    confidence:
        Initial confidence threshold.
    device:
        Device string (``"cuda"``, ``"cpu"``, or ``""`` for auto).
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "",
    ) -> None:
        super().__init__(confidence)

        self._weights_path = weights_path
        self._device = device
        self._model: Optional[object] = None  # Lazy type to avoid import at module level
        self._name: str = ""

        self._load_model()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load YOLOv8 weights.  Raises on failure with a clear message."""
        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO model: %s", self._weights_path)
            self._model = YOLO(self._weights_path)
            self._name = f"YOLOv8 ({self._weights_path})"

            # Warm-up inference — forces weight loading and JIT compilation
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)

            logger.info(
                "Model loaded successfully — device: %s",
                self._device or "auto",
            )
        except ImportError:
            logger.critical("Ultralytics package not installed. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.critical("Failed to load YOLO model '%s': %s", self._weights_path, exc)
            raise RuntimeError(
                f"Could not load model '{self._weights_path}'. "
                f"Ensure the weights file exists or can be downloaded.\n"
                f"Error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._name

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLOv8 inference on a BGR frame.

        Returns a ``DetectionResult`` populated with boxes, confidences,
        class IDs, class names, and timing information.
        """
        import time

        if self._model is None:
            return DetectionResult()

        start = time.perf_counter()

        # Ultralytics predict — verbose=False suppresses console spam
        results = self._model(
            frame,
            conf=self._confidence,
            device=self._device or None,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Parse first result (single-image batch)
        result = results[0] if results else None
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(inference_time_ms=elapsed_ms)

        boxes_tensor = result.boxes.xyxy.cpu().numpy()
        confs_tensor = result.boxes.conf.cpu().numpy()
        cls_tensor = result.boxes.cls.cpu().numpy().astype(int)

        # Map class IDs → human names
        names_map = result.names  # dict[int, str]
        class_names = [names_map.get(int(c), f"class_{c}") for c in cls_tensor]

        return DetectionResult(
            boxes=boxes_tensor.tolist(),
            confidences=confs_tensor.tolist(),
            class_ids=cls_tensor.tolist(),
            class_names=class_names,
            inference_time_ms=elapsed_ms,
        )
