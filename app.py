"""
VisionPulse Dashboard — Application Entry Point

Bootstraps the Qt application:
    1. Creates the QApplication instance
    2. Loads the dark theme stylesheet
    3. Instantiates the DashboardWindow
    4. Enters the Qt event loop

Usage::

    python app.py
"""

import sys
from pathlib import Path

# CRITICAL: Import PySide6 before pyqtgraph to ensure binding detection
import PySide6  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from config.settings import AppSettings
from utils.logger import get_logger


def load_stylesheet(app: QApplication, style_path: Path) -> None:
    """
    Load a QSS stylesheet from disk and apply it to the application.

    Parameters
    ----------
    app:
        The Qt application instance.
    style_path:
        Path to the ``.qss`` file.
    """
    logger = get_logger(__name__)

    if not style_path.is_file():
        logger.warning("Stylesheet not found: %s — using default theme", style_path)
        return

    try:
        stylesheet = style_path.read_text(encoding="utf-8")
        app.setStyleSheet(stylesheet)
        logger.info("Stylesheet loaded: %s", style_path)
    except Exception as exc:
        logger.error("Failed to load stylesheet: %s", exc)


def main() -> int:
    """
    Application entry point.

    Returns
    -------
    int
        Exit code from the Qt event loop.
    """
    settings = AppSettings()
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("VisionPulse Dashboard — Starting")
    logger.info("=" * 60)

    # ---- Qt Application -----------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName(settings.ui.window_title)
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VisionPulse")

    # High-DPI support (Qt6 handles this automatically, but be explicit)
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Default font
    font = QFont(settings.ui.font_family, settings.ui.font_size)
    app.setFont(font)

    # ---- Stylesheet ----------------------------------------------------
    load_stylesheet(app, settings.style_path)

    # ---- Main Window ---------------------------------------------------
    from ui.dashboard import DashboardWindow

    window = DashboardWindow()
    window.show()

    logger.info("Application window shown — entering event loop")

    # ---- Event Loop ----------------------------------------------------
    exit_code = app.exec()

    logger.info("Application exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
