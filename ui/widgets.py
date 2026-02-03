"""Custom widgets for Video Montage Linker UI."""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent
from PyQt6.QtWidgets import QWidget


class TrimSlider(QWidget):
    """A slider widget with two draggable handles for trimming sequences.

    Allows setting in/out points for a sequence by dragging left and right handles.
    Gray areas indicate trimmed regions, colored area indicates included images.
    """

    trimChanged = pyqtSignal(int, int, str)  # Emits (trim_start, trim_end, 'left' or 'right')

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the trim slider.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._total = 0
        self._trim_start = 0
        self._trim_end = 0
        self._current_pos = 0
        self._dragging: Optional[str] = None  # 'left', 'right', or None
        self._handle_width = 10
        self._track_height = 20
        self._enabled = True

        self.setMinimumHeight(40)
        self.setMinimumWidth(100)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setMouseTracking(True)

    def setRange(self, total: int) -> None:
        """Set the total number of items in the sequence.

        Args:
            total: Total number of items.
        """
        self._total = max(0, total)
        # Clamp trim values to valid range
        self._trim_start = min(self._trim_start, max(0, self._total - 1))
        self._trim_end = min(self._trim_end, max(0, self._total - 1 - self._trim_start))
        self.update()

    def setTrimStart(self, value: int) -> None:
        """Set the trim start value.

        Args:
            value: Number of items to trim from start.
        """
        max_start = max(0, self._total - 1 - self._trim_end)
        self._trim_start = max(0, min(value, max_start))
        self.update()

    def setTrimEnd(self, value: int) -> None:
        """Set the trim end value.

        Args:
            value: Number of items to trim from end.
        """
        max_end = max(0, self._total - 1 - self._trim_start)
        self._trim_end = max(0, min(value, max_end))
        self.update()

    def setCurrentPosition(self, pos: int) -> None:
        """Set the current position indicator.

        Args:
            pos: Current position index.
        """
        self._current_pos = max(0, min(pos, self._total - 1)) if self._total > 0 else 0
        self.update()

    def trimStart(self) -> int:
        """Get the trim start value."""
        return self._trim_start

    def trimEnd(self) -> int:
        """Get the trim end value."""
        return self._trim_end

    def total(self) -> int:
        """Get the total number of items."""
        return self._total

    def includedRange(self) -> tuple[int, int]:
        """Get the range of included items (after trimming).

        Returns:
            Tuple of (first_included_index, last_included_index).
            Returns (-1, -1) if no items are included.
        """
        if self._total == 0:
            return (-1, -1)
        first = self._trim_start
        last = self._total - 1 - self._trim_end
        if first > last:
            return (-1, -1)
        return (first, last)

    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable the widget."""
        self._enabled = enabled
        self.update()

    def _track_rect(self) -> QRect:
        """Get the rectangle for the slider track."""
        margin = self._handle_width
        return QRect(
            margin,
            (self.height() - self._track_height) // 2,
            self.width() - 2 * margin,
            self._track_height
        )

    def _value_to_x(self, value: int) -> int:
        """Convert a value to an x coordinate."""
        track = self._track_rect()
        if self._total <= 1:
            return track.left()
        ratio = value / (self._total - 1)
        return int(track.left() + ratio * track.width())

    def _x_to_value(self, x: int) -> int:
        """Convert an x coordinate to a value."""
        track = self._track_rect()
        if track.width() == 0 or self._total <= 1:
            return 0
        ratio = (x - track.left()) / track.width()
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * (self._total - 1)))

    def _left_handle_rect(self) -> QRect:
        """Get the rectangle for the left (trim start) handle."""
        x = self._value_to_x(self._trim_start)
        return QRect(
            x - self._handle_width // 2,
            (self.height() - self._track_height - 10) // 2,
            self._handle_width,
            self._track_height + 10
        )

    def _right_handle_rect(self) -> QRect:
        """Get the rectangle for the right (trim end) handle."""
        x = self._value_to_x(self._total - 1 - self._trim_end) if self._total > 0 else 0
        return QRect(
            x - self._handle_width // 2,
            (self.height() - self._track_height - 10) // 2,
            self._handle_width,
            self._track_height + 10
        )

    def paintEvent(self, event) -> None:
        """Paint the trim slider."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track = self._track_rect()

        # Colors
        bg_color = QColor(60, 60, 60)
        trimmed_color = QColor(80, 80, 80)
        included_color = QColor(52, 152, 219) if self._enabled else QColor(100, 100, 100)
        handle_color = QColor(200, 200, 200) if self._enabled else QColor(120, 120, 120)
        position_color = QColor(255, 255, 255)

        # Draw background track
        painter.fillRect(track, bg_color)

        if self._total > 0:
            # Draw trimmed regions (darker)
            left_trim_x = self._value_to_x(self._trim_start)
            right_trim_x = self._value_to_x(self._total - 1 - self._trim_end)

            # Left trimmed region
            if self._trim_start > 0:
                left_rect = QRect(track.left(), track.top(),
                                  left_trim_x - track.left(), track.height())
                painter.fillRect(left_rect, trimmed_color)

            # Right trimmed region
            if self._trim_end > 0:
                right_rect = QRect(right_trim_x, track.top(),
                                   track.right() - right_trim_x, track.height())
                painter.fillRect(right_rect, trimmed_color)

            # Draw included region
            if left_trim_x < right_trim_x:
                included_rect = QRect(left_trim_x, track.top(),
                                      right_trim_x - left_trim_x, track.height())
                painter.fillRect(included_rect, included_color)

            # Draw current position indicator
            if self._trim_start <= self._current_pos <= (self._total - 1 - self._trim_end):
                pos_x = self._value_to_x(self._current_pos)
                painter.setPen(QPen(position_color, 2))
                painter.drawLine(pos_x, track.top() - 2, pos_x, track.bottom() + 2)

            # Draw handles
            painter.setBrush(QBrush(handle_color))
            painter.setPen(QPen(Qt.GlobalColor.black, 1))

            # Left handle
            left_handle = self._left_handle_rect()
            painter.drawRect(left_handle)

            # Right handle
            right_handle = self._right_handle_rect()
            painter.drawRect(right_handle)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press to start dragging handles."""
        if not self._enabled or self._total == 0:
            return

        pos = event.pos()

        # Check if clicking on handles (check right first since it may overlap)
        right_rect = self._right_handle_rect()
        left_rect = self._left_handle_rect()

        # Expand hit area slightly for easier grabbing
        expand = 5
        left_expanded = left_rect.adjusted(-expand, -expand, expand, expand)
        right_expanded = right_rect.adjusted(-expand, -expand, expand, expand)

        if right_expanded.contains(pos):
            self._dragging = 'right'
        elif left_expanded.contains(pos):
            self._dragging = 'left'
        else:
            self._dragging = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move to drag handles."""
        if not self._enabled:
            return

        pos = event.pos()

        # Update cursor based on position
        if self._dragging:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            left_rect = self._left_handle_rect()
            right_rect = self._right_handle_rect()
            expand = 5
            left_expanded = left_rect.adjusted(-expand, -expand, expand, expand)
            right_expanded = right_rect.adjusted(-expand, -expand, expand, expand)

            if left_expanded.contains(pos) or right_expanded.contains(pos):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._dragging and self._total > 0:
            value = self._x_to_value(pos.x())

            if self._dragging == 'left':
                # Left handle: set trim_start, clamped to not exceed right
                max_start = self._total - 1 - self._trim_end
                new_start = max(0, min(value, max_start))
                if new_start != self._trim_start:
                    self._trim_start = new_start
                    self.update()
                    self.trimChanged.emit(self._trim_start, self._trim_end, 'left')

            elif self._dragging == 'right':
                # Right handle: set trim_end based on position
                # value is the index position, trim_end is count from end
                max_val = self._total - 1 - self._trim_start
                clamped_value = max(self._trim_start, min(value, self._total - 1))
                new_end = self._total - 1 - clamped_value
                if new_end != self._trim_end:
                    self._trim_end = max(0, new_end)
                    self.update()
                    self.trimChanged.emit(self._trim_start, self._trim_end, 'right')

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to stop dragging."""
        self._dragging = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
