"""
VisionPulse Dashboard — Analytics Panel (Right Panel)

Displays real-time detection statistics and a rolling object-count chart.

Stat cards:
    - Current FPS
    - Average FPS
    - Inference Time (ms)
    - Detected Objects
    - Current Confidence

Chart:
    - pyqtgraph PlotWidget showing the last 100 object-count samples
    - Updates continuously via the ``update_stats()`` slot

The panel receives data exclusively through the ``stats_ready(dict)``
signal emitted by the ``VideoWorker``.
"""

from typing import List, Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from config.settings import AppSettings

_settings = AppSettings()


class StatCard(QFrame):
    """
    A styled card displaying a single metric with label and value.

    Parameters
    ----------
    label:
        Description text (e.g. "Current FPS").
    initial_value:
        Starting display value.
    """

    def __init__(
        self,
        label: str,
        initial_value: str = "—",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("stat_card")
        self._init_ui(label, initial_value)

    def _init_ui(self, label: str, initial_value: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self._value_label = QLabel(initial_value)
        self._value_label.setObjectName("lbl_stat_value")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._desc_label = QLabel(label)
        self._desc_label.setObjectName("lbl_stat_label")
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._value_label)
        layout.addWidget(self._desc_label)

    def set_value(self, value: str) -> None:
        """Update the displayed value."""
        self._value_label.setText(value)


class AnalyticsPanel(QWidget):
    """
    Right sidebar displaying live detection analytics and a rolling chart.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the analytics panel layout."""
        self.setFixedWidth(_settings.ui.right_panel_width)

        # Container frame
        container = QFrame()
        container.setObjectName("panel_right")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(10)

        # ---- Title ----
        title = QLabel("Analytics")
        title.setObjectName("lbl_section_title")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        layout.addWidget(self._create_separator())

        # ---- Stat cards in a 2-column grid ----
        grid = QGridLayout()
        grid.setSpacing(8)

        self.card_fps = StatCard("Current FPS")
        self.card_avg_fps = StatCard("Average FPS")
        self.card_inference = StatCard("Inference (ms)")
        self.card_objects = StatCard("Objects")
        self.card_confidence = StatCard("Confidence", f"{_settings.model.default_confidence:.2f}")

        grid.addWidget(self.card_fps, 0, 0)
        grid.addWidget(self.card_avg_fps, 0, 1)
        grid.addWidget(self.card_inference, 1, 0)
        grid.addWidget(self.card_objects, 1, 1)
        grid.addWidget(self.card_confidence, 2, 0, 1, 2)  # Span both columns

        layout.addLayout(grid)

        layout.addSpacing(4)
        layout.addWidget(self._create_separator())
        layout.addSpacing(4)

        # ---- Detected classes ----
        classes_title = QLabel("Detected Classes")
        classes_title.setObjectName("lbl_panel_title")
        layout.addWidget(classes_title)

        self.lbl_classes = QLabel("—")
        self.lbl_classes.setObjectName("lbl_stat_label")
        self.lbl_classes.setWordWrap(True)
        self.lbl_classes.setMinimumHeight(40)
        layout.addWidget(self.lbl_classes)

        layout.addSpacing(4)
        layout.addWidget(self._create_separator())
        layout.addSpacing(4)

        # ---- Rolling chart ----
        chart_title = QLabel("Object Count (Rolling)")
        chart_title.setObjectName("lbl_panel_title")
        layout.addWidget(chart_title)

        self._chart = self._create_chart()
        layout.addWidget(self._chart)

        layout.addStretch()

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

    def _create_chart(self) -> pg.PlotWidget:
        """
        Create and configure the pyqtgraph rolling chart.

        Returns a ``PlotWidget`` styled to match the dark theme.
        """
        chart = pg.PlotWidget()
        chart.setMinimumHeight(180)
        chart.setMaximumHeight(220)

        # Dark-theme styling
        chart.setBackground("#1c2333")
        chart.getAxis("bottom").setPen(pg.mkPen("#8b949e", width=1))
        chart.getAxis("left").setPen(pg.mkPen("#8b949e", width=1))
        chart.getAxis("bottom").setTextPen(pg.mkPen("#8b949e"))
        chart.getAxis("left").setTextPen(pg.mkPen("#8b949e"))
        chart.showGrid(x=True, y=True, alpha=0.15)
        chart.setLabel("bottom", "Samples")
        chart.setLabel("left", "Count")
        chart.setXRange(0, _settings.analytics.max_samples)
        chart.setMouseEnabled(x=False, y=False)  # Disable zoom/pan
        chart.hideButtons()

        # Plot line — will be updated each frame
        self._plot_line = chart.plot(
            pen=pg.mkPen(color="#58a6ff", width=2),
            fillLevel=0,
            fillBrush=pg.mkBrush(88, 166, 255, 40),  # Translucent fill
        )

        return chart

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
    # Public slot
    # ------------------------------------------------------------------

    def update_stats(self, stats: dict) -> None:
        """
        Update all stat cards and the chart with fresh data.

        Parameters
        ----------
        stats:
            Dictionary from ``AnalyticsService.get_snapshot()``.
            Expected keys: current_fps, average_fps, inference_ms,
            object_count, class_names, class_counts, object_count_series.
        """
        self.card_fps.set_value(f"{stats.get('current_fps', 0):.1f}")
        self.card_avg_fps.set_value(f"{stats.get('average_fps', 0):.1f}")
        self.card_inference.set_value(f"{stats.get('inference_ms', 0):.1f}")
        self.card_objects.set_value(str(stats.get("object_count", 0)))

        # Detected classes — show unique names with counts
        class_counts: dict = stats.get("class_counts", {})
        if class_counts:
            # Sort by count descending, show top 10
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            classes_text = "  •  ".join(f"{name}: {count}" for name, count in sorted_classes)
            self.lbl_classes.setText(classes_text)
        else:
            self.lbl_classes.setText("—")

        # Rolling chart update
        series: List[int] = stats.get("object_count_series", [])
        if series:
            self._plot_line.setData(series)

    def update_confidence_display(self, value: float) -> None:
        """Update the confidence stat card when the slider changes."""
        self.card_confidence.set_value(f"{value:.2f}")

    def reset(self) -> None:
        """Clear all stats and the chart."""
        self.card_fps.set_value("—")
        self.card_avg_fps.set_value("—")
        self.card_inference.set_value("—")
        self.card_objects.set_value("—")
        self.lbl_classes.setText("—")
        self._plot_line.setData([])
