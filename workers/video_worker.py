"""
VisionPulse Dashboard — Video Worker Thread

The ``VideoWorker`` is the heart of the application's multithreading
architecture.  It runs the entire capture → detect → emit loop on a
dedicated ``QThread``, ensuring the GUI thread is never blocked.

Signal contract:
    frame_ready(QImage)   — annotated frame ready for display
    stats_ready(dict)     — analytics snapshot for the right panel
    error_occurred(str)   — user-friendly error message
    finished()            — worker has shut down cleanly

Shutdown protocol:
    1. GUI calls ``stop()``
    2. ``stop()`` sets ``requestInterruption()``
    3. Run loop checks ``isInterruptionRequested()`` each iteration
    4. Loop exits → capture released → ``finished`` emitted

No UI widget is ever touched from this thread.
"""

import time
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from config.settings import AppSettings
from models.detector import BaseDetector, DetectionResult
from services.analytics_service import AnalyticsService
from services.camera_service import CameraService
from services.video_service import VideoService
from utils.image_converter import bgr_to_rgb, draw_detections, frame_to_qimage
from utils.logger import get_logger

logger = get_logger(__name__)

_settings = AppSettings()


class SourceType(Enum):
    """Enumeration of supported video source types."""
    CAMERA = auto()
    FILE = auto()


class VideoWorker(QThread):
    """
    Dedicated worker thread for video capture and AI inference.

    This thread owns its own ``CameraService`` / ``VideoService`` and
    ``AnalyticsService`` instances.  The ``BaseDetector`` is injected
    from outside so the model is loaded once and shared.

    Parameters
    ----------
    detector:
        A pre-loaded detection model implementing ``BaseDetector``.
    parent:
        Optional Qt parent object.
    """

    # ------------------------------------------------------------------
    # Qt Signals — all communication with the GUI goes through these
    # ------------------------------------------------------------------
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        detector: BaseDetector,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._camera_service = CameraService()
        self._video_service = VideoService()
        self._analytics = AnalyticsService()

        self._source_type: Optional[SourceType] = None
        self._video_path: Optional[str] = None

        # FPS tracking
        self._target_fps: int = _settings.video.target_fps
        self._frame_interval: float = 1.0 / self._target_fps

    # ------------------------------------------------------------------
    # Public API — called from the GUI thread
    # ------------------------------------------------------------------

    def start_camera(self, device_id: int = 0) -> None:
        """Configure the worker to read from a webcam, then start."""
        self._source_type = SourceType.CAMERA
        self._camera_service = CameraService(device_id)
        self._analytics.reset()
        self.start()

    def start_video(self, path: str) -> None:
        """Configure the worker to read from a video file, then start."""
        self._source_type = SourceType.FILE
        self._video_path = path
        self._analytics.reset()
        self.start()

    def stop(self) -> None:
        """Request a graceful shutdown of the worker loop."""
        logger.info("Worker stop requested")
        self.requestInterruption()
        if not self.wait(5000):  # 5-second timeout
            logger.warning("Worker did not stop within timeout — terminating")
            self.terminate()
            self.wait()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main worker loop.  Runs entirely on the worker thread.

        Sequence per iteration:
            1. Read frame from source
            2. Run detection
            3. Draw annotations
            4. Convert to QImage
            5. Emit frame + stats signals
            6. Pace to target FPS
        """
        logger.info("Worker thread started — source: %s", self._source_type)

        try:
            if not self._open_source():
                return

            self._processing_loop()

        except Exception as exc:
            logger.exception("Unexpected error in worker thread")
            self.error_occurred.emit(f"Unexpected error: {exc}")

        finally:
            self._release_source()
            logger.info("Worker thread finished")
            self.finished.emit()

    # ------------------------------------------------------------------
    # Internal — source management
    # ------------------------------------------------------------------

    def _open_source(self) -> bool:
        """Open the configured video source.  Emits error on failure."""
        if self._source_type == SourceType.CAMERA:
            if not self._camera_service.open():
                self.error_occurred.emit(
                    "Could not open webcam.\n\n"
                    "Please check that a camera is connected and not in use "
                    "by another application."
                )
                return False
            return True

        elif self._source_type == SourceType.FILE:
            if self._video_path is None:
                self.error_occurred.emit("No video file path specified.")
                return False
            if not self._video_service.open(self._video_path):
                self.error_occurred.emit(
                    f"Could not open video file:\n{self._video_path}\n\n"
                    "The file may be corrupt or in an unsupported format."
                )
                return False
            return True

        else:
            self.error_occurred.emit("No video source configured.")
            return False

    def _release_source(self) -> None:
        """Release whichever source is active."""
        self._camera_service.release()
        self._video_service.release()

    def _read_frame(self):
        """Read a frame from the active source."""
        if self._source_type == SourceType.CAMERA:
            return self._camera_service.read()
        elif self._source_type == SourceType.FILE:
            return self._video_service.read()
        return False, None

    # ------------------------------------------------------------------
    # Internal — main processing loop
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """
        Core capture → detect → emit loop.

        Uses ``isInterruptionRequested()`` for clean shutdown and
        ``time.perf_counter()`` for accurate FPS pacing.
        """
        prev_time = time.perf_counter()
        start_time = time.perf_counter()
        frame_idx = 0

        while not self.isInterruptionRequested():
            loop_start = time.perf_counter()

            # -- Frame skipping for video files to maintain real-time playback --
            if self._source_type == SourceType.FILE:
                elapsed_real = time.perf_counter() - start_time
                target_frame_idx = int(elapsed_real * self._target_fps)
                
                frames_to_skip = max(0, target_frame_idx - frame_idx)
                
                hit_eof = False
                for _ in range(frames_to_skip):
                    ok, _ = self._read_frame()
                    if not ok:
                        hit_eof = True
                        break
                    frame_idx += 1
                    
                if hit_eof:
                    logger.info("Video file playback complete")
                    break

            # -- 1. Read frame ------------------------------------------
            ok, frame = self._read_frame()
            if not ok:
                if self._source_type == SourceType.FILE:
                    logger.info("Video file playback complete")
                else:
                    self.error_occurred.emit("Camera feed lost.")
                break
                
            if self._source_type == SourceType.FILE:
                frame_idx += 1

            # -- 2. Run detection ---------------------------------------
            result: DetectionResult = self._detector.detect(frame)

            # -- 3. Draw annotations on the BGR frame -------------------
            if result.count > 0:
                draw_detections(
                    frame,
                    result.boxes,
                    result.confidences,
                    result.class_names,
                )

            # -- 4. Convert BGR → RGB → QImage -------------------------
            rgb_frame = bgr_to_rgb(frame)
            qimage = frame_to_qimage(rgb_frame)

            # -- 5. Calculate FPS ---------------------------------------
            current_time = time.perf_counter()
            delta = current_time - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            prev_time = current_time

            # -- 6. Update analytics ------------------------------------
            self._analytics.update(
                fps=fps,
                object_count=result.count,
                inference_ms=result.inference_time_ms,
                class_names=result.class_names,
            )

            # -- 7. Emit signals ----------------------------------------
            self.frame_ready.emit(qimage)
            self.stats_ready.emit(self._analytics.get_snapshot())

            # -- 8. Pace to target FPS ----------------------------------
            elapsed = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))
