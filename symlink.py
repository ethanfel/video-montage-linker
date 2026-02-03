#!/usr/bin/env python3
"""Video Montage Linker - Create sequenced symlinks for image files.

Supports both GUI and CLI modes for creating numbered symlinks from one or more
source directories into a single destination directory.
"""

# --- Imports ---
import argparse
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, QEvent, QPoint, pyqtSignal, QRect
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QColor, QBrush, QPen, QMouseEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QHBoxLayout,
    QMessageBox,
    QListWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QComboBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QKeyEvent

# --- Configuration ---
SUPPORTED_EXTENSIONS = ('.png', '.webp', '.jpg', '.jpeg')
VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v')
DB_PATH = Path.home() / '.config' / 'video-montage-linker' / 'symlinks.db'


# --- Custom Widgets ---
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


# --- Exceptions ---
class SymlinkError(Exception):
    """Base exception for symlink operations."""


class PathValidationError(SymlinkError):
    """Error validating file paths."""


class SourceNotFoundError(PathValidationError):
    """Source directory does not exist."""


class DestinationError(PathValidationError):
    """Error with destination directory."""


class CleanupError(SymlinkError):
    """Error during cleanup of existing symlinks."""


class DatabaseError(SymlinkError):
    """Error with database operations."""


# --- Data Classes ---
@dataclass
class LinkResult:
    """Result of a symlink creation operation."""

    source_path: Path
    link_path: Path
    sequence_number: int
    success: bool
    error: Optional[str] = None


@dataclass
class SymlinkRecord:
    """Database record of a created symlink."""

    id: int
    session_id: int
    source_path: str
    link_path: str
    original_filename: str
    sequence_number: int
    created_at: datetime


@dataclass
class SessionRecord:
    """Database record of a symlink session."""

    id: int
    created_at: datetime
    destination: str
    link_count: int = 0


# --- Database ---
class DatabaseManager:
    """Manages SQLite database for tracking symlink sessions and links."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS symlink_sessions (
                    id INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    destination TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS symlinks (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    link_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sequence_trim_settings (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    source_folder TEXT NOT NULL,
                    trim_start INTEGER DEFAULT 0,
                    trim_end INTEGER DEFAULT 0,
                    UNIQUE(session_id, source_folder)
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_session(self, destination: str) -> int:
        """Create a new linking session.

        Args:
            destination: The destination directory path.

        Returns:
            The ID of the created session.

        Raises:
            DatabaseError: If session creation fails.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO symlink_sessions (destination) VALUES (?)",
                    (destination,)
                )
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create session: {e}") from e

    def record_symlink(
        self,
        session_id: int,
        source: str,
        link: str,
        filename: str,
        seq: int
    ) -> int:
        """Record a created symlink.

        Args:
            session_id: The session this symlink belongs to.
            source: Full path to the source file.
            link: Full path to the created symlink.
            filename: Original filename.
            seq: Sequence number in the destination.

        Returns:
            The ID of the created record.

        Raises:
            DatabaseError: If recording fails.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """INSERT INTO symlinks
                       (session_id, source_path, link_path, original_filename, sequence_number)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, source, link, filename, seq)
                )
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to record symlink: {e}") from e

    def get_sessions(self) -> list[SessionRecord]:
        """List all sessions with link counts.

        Returns:
            List of session records.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.id, s.created_at, s.destination, COUNT(l.id) as link_count
                FROM symlink_sessions s
                LEFT JOIN symlinks l ON s.id = l.session_id
                GROUP BY s.id
                ORDER BY s.created_at DESC
            """).fetchall()

        return [
            SessionRecord(
                id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                destination=row[2],
                link_count=row[3]
            )
            for row in rows
        ]

    def get_symlinks_by_session(self, session_id: int) -> list[SymlinkRecord]:
        """Get all symlinks for a session.

        Args:
            session_id: The session ID to query.

        Returns:
            List of symlink records.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, session_id, source_path, link_path,
                          original_filename, sequence_number, created_at
                   FROM symlinks WHERE session_id = ?
                   ORDER BY sequence_number""",
                (session_id,)
            ).fetchall()

        return [
            SymlinkRecord(
                id=row[0],
                session_id=row[1],
                source_path=row[2],
                link_path=row[3],
                original_filename=row[4],
                sequence_number=row[5],
                created_at=datetime.fromisoformat(row[6])
            )
            for row in rows
        ]

    def get_symlinks_by_destination(self, dest: str) -> list[SymlinkRecord]:
        """Get all symlinks for a destination directory.

        Args:
            dest: The destination directory path.

        Returns:
            List of symlink records.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT l.id, l.session_id, l.source_path, l.link_path,
                          l.original_filename, l.sequence_number, l.created_at
                   FROM symlinks l
                   JOIN symlink_sessions s ON l.session_id = s.id
                   WHERE s.destination = ?
                   ORDER BY l.sequence_number""",
                (dest,)
            ).fetchall()

        return [
            SymlinkRecord(
                id=row[0],
                session_id=row[1],
                source_path=row[2],
                link_path=row[3],
                original_filename=row[4],
                sequence_number=row[5],
                created_at=datetime.fromisoformat(row[6])
            )
            for row in rows
        ]

    def delete_session(self, session_id: int) -> None:
        """Delete a session and all its symlink records.

        Args:
            session_id: The session ID to delete.

        Raises:
            DatabaseError: If deletion fails.
        """
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM symlinks WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM symlink_sessions WHERE id = ?", (session_id,))
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete session: {e}") from e

    def get_sessions_by_destination(self, dest: str) -> list[SessionRecord]:
        """Get all sessions for a destination directory.

        Args:
            dest: The destination directory path.

        Returns:
            List of session records.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.id, s.created_at, s.destination, COUNT(l.id) as link_count
                FROM symlink_sessions s
                LEFT JOIN symlinks l ON s.id = l.session_id
                WHERE s.destination = ?
                GROUP BY s.id
                ORDER BY s.created_at DESC
            """, (dest,)).fetchall()

        return [
            SessionRecord(
                id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                destination=row[2],
                link_count=row[3]
            )
            for row in rows
        ]

    def save_trim_settings(
        self,
        session_id: int,
        source_folder: str,
        trim_start: int,
        trim_end: int
    ) -> None:
        """Save trim settings for a folder in a session.

        Args:
            session_id: The session ID.
            source_folder: Path to the source folder.
            trim_start: Number of images to trim from start.
            trim_end: Number of images to trim from end.

        Raises:
            DatabaseError: If saving fails.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO sequence_trim_settings
                       (session_id, source_folder, trim_start, trim_end)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(session_id, source_folder)
                       DO UPDATE SET trim_start=excluded.trim_start,
                                     trim_end=excluded.trim_end""",
                    (session_id, source_folder, trim_start, trim_end)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save trim settings: {e}") from e

    def get_trim_settings(
        self,
        session_id: int,
        source_folder: str
    ) -> tuple[int, int]:
        """Get trim settings for a folder in a session.

        Args:
            session_id: The session ID.
            source_folder: Path to the source folder.

        Returns:
            Tuple of (trim_start, trim_end). Returns (0, 0) if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT trim_start, trim_end FROM sequence_trim_settings
                   WHERE session_id = ? AND source_folder = ?""",
                (session_id, source_folder)
            ).fetchone()

        if row:
            return (row[0], row[1])
        return (0, 0)

    def get_all_trim_settings(self, session_id: int) -> dict[str, tuple[int, int]]:
        """Get all trim settings for a session.

        Args:
            session_id: The session ID.

        Returns:
            Dict mapping source folder paths to (trim_start, trim_end) tuples.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT source_folder, trim_start, trim_end
                   FROM sequence_trim_settings WHERE session_id = ?""",
                (session_id,)
            ).fetchall()

        return {row[0]: (row[1], row[2]) for row in rows}


# --- Business Logic ---
class SymlinkManager:
    """Manages symlink creation and cleanup operations."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        """Initialize the symlink manager.

        Args:
            db: Optional database manager for tracking operations.
        """
        self.db = db

    @staticmethod
    def get_supported_files(directories: list[Path]) -> list[tuple[Path, str]]:
        """Get all supported image files from multiple directories.

        Files are returned sorted by directory order (as provided), then
        alphabetically by filename within each directory.

        Args:
            directories: List of source directories to scan.

        Returns:
            List of (directory, filename) tuples.
        """
        files: list[tuple[Path, str]] = []

        for directory in directories:
            if not directory.is_dir():
                continue
            dir_files = []
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    dir_files.append((directory, item.name))
            # Sort files within this directory alphabetically
            dir_files.sort(key=lambda x: x[1].lower())
            files.extend(dir_files)

        return files

    @staticmethod
    def validate_paths(sources: list[Path], dest: Path) -> None:
        """Validate source and destination paths.

        Args:
            sources: List of source directories.
            dest: Destination directory.

        Raises:
            SourceNotFoundError: If any source directory doesn't exist.
            DestinationError: If destination cannot be created or accessed.
        """
        if not sources:
            raise SourceNotFoundError("No source directories specified")

        for source in sources:
            if not source.exists():
                raise SourceNotFoundError(f"Source directory not found: {source}")
            if not source.is_dir():
                raise SourceNotFoundError(f"Source is not a directory: {source}")

        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DestinationError(f"Cannot create destination directory: {e}") from e

        if not dest.is_dir():
            raise DestinationError(f"Destination is not a directory: {dest}")

    @staticmethod
    def cleanup_old_links(directory: Path) -> int:
        """Remove existing seq* symlinks from a directory.

        Handles both old format (seq_0000) and new format (seq01_0000).

        Args:
            directory: Directory to clean up.

        Returns:
            Number of files removed.

        Raises:
            CleanupError: If cleanup fails.
        """
        removed = 0
        try:
            for item in directory.iterdir():
                # Match both old (seq_NNNN) and new (seqNN_NNNN) formats
                if item.name.startswith("seq") and item.is_symlink():
                    item.unlink()
                    removed += 1
        except OSError as e:
            raise CleanupError(f"Failed to clean up old links: {e}") from e

        return removed

    def create_sequence_links(
        self,
        sources: list[Path],
        dest: Path,
        files: list[tuple],
        trim_settings: Optional[dict[Path, tuple[int, int]]] = None,
    ) -> tuple[list[LinkResult], Optional[int]]:
        """Create sequenced symlinks from source files to destination.

        Args:
            sources: List of source directories (for validation).
            dest: Destination directory.
            files: List of tuples. Can be:
                   - (source_dir, filename) for CLI mode (uses global sequence)
                   - (source_dir, filename, folder_idx, file_idx) for GUI mode
            trim_settings: Optional dict mapping folder paths to (trim_start, trim_end).

        Returns:
            Tuple of (list of LinkResult objects, session_id or None).
        """
        self.validate_paths(sources, dest)
        self.cleanup_old_links(dest)

        session_id = None
        if self.db:
            session_id = self.db.create_session(str(dest))

            # Save trim settings if provided
            if trim_settings and session_id:
                for folder, (trim_start, trim_end) in trim_settings.items():
                    if trim_start > 0 or trim_end > 0:
                        self.db.save_trim_settings(
                            session_id, str(folder), trim_start, trim_end
                        )

        results: list[LinkResult] = []

        # Check if we have folder indices (GUI mode) or not (CLI mode)
        use_folder_sequences = len(files) > 0 and len(files[0]) >= 4

        # For CLI mode without folder indices, calculate them
        if not use_folder_sequences:
            folder_to_index = {folder: i for i, folder in enumerate(sources)}
            folder_file_counts: dict[Path, int] = {}
            expanded_files = []
            for source_dir, filename in files:
                folder_idx = folder_to_index.get(source_dir, 0)
                file_idx = folder_file_counts.get(source_dir, 0)
                folder_file_counts[source_dir] = file_idx + 1
                expanded_files.append((source_dir, filename, folder_idx, file_idx))
            files = expanded_files

        for i, file_data in enumerate(files):
            source_dir, filename, folder_idx, file_idx = file_data
            source_path = source_dir / filename
            ext = source_path.suffix
            link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
            link_path = dest / link_name

            # Calculate relative path from destination to source
            rel_source = Path(os.path.relpath(source_path.resolve(), dest.resolve()))

            try:
                link_path.symlink_to(rel_source)

                if self.db and session_id:
                    self.db.record_symlink(
                        session_id=session_id,
                        source=str(source_path.resolve()),
                        link=str(link_path),
                        filename=filename,
                        seq=i
                    )

                results.append(LinkResult(
                    source_path=source_path,
                    link_path=link_path,
                    sequence_number=i,
                    success=True
                ))
            except OSError as e:
                results.append(LinkResult(
                    source_path=source_path,
                    link_path=link_path,
                    sequence_number=i,
                    success=False,
                    error=str(e)
                ))

        return results, session_id


# --- GUI ---
class SequenceLinkerUI(QWidget):
    """PyQt6 GUI for the Video Montage Linker."""

    def __init__(self) -> None:
        """Initialize the UI."""
        super().__init__()
        self.source_folders: list[Path] = []
        self.last_directory: Optional[str] = None
        self._last_resumed_dest: Optional[str] = None  # Track to avoid double resume
        self._folder_trim_settings: dict[Path, tuple[int, int]] = {}  # In-memory trim cache
        self._folder_file_counts: dict[Path, int] = {}  # Total files per folder (before trim)
        self._current_session_id: Optional[int] = None  # Track session for saving trim
        self.db = DatabaseManager()
        self.manager = SymlinkManager(self.db)
        self._setup_window()
        self._create_widgets()
        self._create_layout()
        self._connect_signals()
        self.setAcceptDrops(True)

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle('Video Montage Linker')
        self.setMinimumSize(1000, 700)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Source folders group
        self.source_group = QGroupBox("Source Folders (drag to reorder, drop folders here)")
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(100)
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.source_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.add_source_btn = QPushButton("Add Folder")
        self.remove_source_btn = QPushButton("Remove Folder")

        # Destination
        self.dst_label = QLabel("Destination Folder:")
        self.dst_path = QLineEdit(placeholderText="Select destination folder")
        self.dst_btn = QPushButton("Browse")

        # File list
        self.files_label = QLabel("Sequence Order (Drag to reorder within folder, Del to remove):")
        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["Sequence Name", "Original Filename", "Source Folder"])
        self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setRootIsDecorated(False)
        self.file_list.header().setStretchLastSection(True)
        self.file_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)

        # Action buttons
        self.remove_files_btn = QPushButton("Remove Files")
        self.refresh_btn = QPushButton("Refresh Files")
        self.run_btn = QPushButton("Generate Virtual Sequence")
        self.run_btn.setStyleSheet(
            "background-color: #3498db; color: white; "
            "height: 40px; font-weight: bold;"
        )

        # Preview tabs
        self.preview_tabs = QTabWidget()

        # Video preview tab
        self.video_tab = QWidget()
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 180)
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.video_combo = QComboBox()
        self.video_combo.setPlaceholderText("Select a video to preview")
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setRange(0, 0)
        self.video_time_label = QLabel("00:00 / 00:00")

        # Image sequence preview tab
        self.image_tab = QWidget()
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.viewport().installEventFilter(self)
        self.image_scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setScaledContents(False)
        self.image_scroll.setWidget(self.image_label)

        self.prev_image_btn = QPushButton("◀ Previous")
        self.next_image_btn = QPushButton("Next ▶")
        self.image_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_slider.setRange(0, 0)
        self.image_index_label = QLabel("0 / 0")
        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_reset_btn = QPushButton("Fit")
        self.zoom_reset_btn.setFixedWidth(40)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self._zoom_level = 1.0
        self._current_pixmap: Optional[QPixmap] = None
        self._pan_start = None
        self._pan_scrollbar_start = None

        # Trim slider for sequence trimming
        self.trim_slider = TrimSlider()
        self.trim_label = QLabel("Frames: All included")
        self.trim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _create_layout(self) -> None:
        """Arrange widgets in layouts."""
        main_layout = QVBoxLayout()

        # Source folders group layout
        source_group_layout = QVBoxLayout()
        source_btn_layout = QHBoxLayout()
        source_btn_layout.addWidget(self.add_source_btn)
        source_btn_layout.addWidget(self.remove_source_btn)
        source_btn_layout.addStretch()
        source_group_layout.addWidget(self.source_list)
        source_group_layout.addLayout(source_btn_layout)
        self.source_group.setLayout(source_group_layout)

        # Destination layout
        dst_layout = QHBoxLayout()
        dst_layout.addWidget(self.dst_path)
        dst_layout.addWidget(self.dst_btn)

        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.remove_files_btn)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()

        # Video preview tab layout
        video_tab_layout = QVBoxLayout(self.video_tab)
        video_tab_layout.addWidget(self.video_combo)
        video_tab_layout.addWidget(self.video_widget, 1)
        video_controls = QHBoxLayout()
        video_controls.addWidget(self.play_btn)
        video_controls.addWidget(self.stop_btn)
        video_controls.addWidget(self.video_slider, 1)
        video_controls.addWidget(self.video_time_label)
        video_tab_layout.addLayout(video_controls)

        # Image sequence preview tab layout
        image_tab_layout = QVBoxLayout(self.image_tab)
        # Top bar with name and zoom controls
        image_top_bar = QHBoxLayout()
        image_top_bar.addWidget(self.image_name_label, 1)
        image_top_bar.addWidget(self.zoom_out_btn)
        image_top_bar.addWidget(self.zoom_label)
        image_top_bar.addWidget(self.zoom_in_btn)
        image_top_bar.addWidget(self.zoom_reset_btn)
        image_tab_layout.addLayout(image_top_bar)
        image_tab_layout.addWidget(self.image_scroll, 1)
        image_controls = QHBoxLayout()
        image_controls.addWidget(self.prev_image_btn)
        image_controls.addWidget(self.image_slider, 1)
        image_controls.addWidget(self.next_image_btn)
        image_controls.addWidget(self.image_index_label)
        image_tab_layout.addLayout(image_controls)
        # Trim slider for selected folder
        image_tab_layout.addWidget(self.trim_label)
        image_tab_layout.addWidget(self.trim_slider)

        # Add tabs to tab widget
        self.preview_tabs.addTab(self.video_tab, "Video Preview")
        self.preview_tabs.addTab(self.image_tab, "Image Sequence")

        # Left panel (file list)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.files_label)
        left_layout.addWidget(self.file_list)
        left_layout.addLayout(btn_layout)

        # Splitter for file list and preview tabs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(self.preview_tabs)
        self.splitter.setSizes([400, 400])

        # Assemble main layout
        main_layout.addWidget(self.source_group)
        main_layout.addWidget(self.dst_label)
        main_layout.addLayout(dst_layout)
        main_layout.addWidget(self.splitter, 1)
        main_layout.addWidget(self.run_btn)

        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.add_source_btn.clicked.connect(self._add_source_folder)
        self.remove_source_btn.clicked.connect(self._remove_source_folder)
        self.dst_btn.clicked.connect(self._browse_destination)
        self.dst_path.editingFinished.connect(self._on_destination_changed)
        self.remove_files_btn.clicked.connect(self._remove_selected_files)
        self.refresh_btn.clicked.connect(self._refresh_files)
        self.run_btn.clicked.connect(self._process_links)
        # Connect reorder signals
        self.source_list.model().rowsMoved.connect(self._on_folders_reordered)
        self.file_list.model().rowsMoved.connect(self._recalculate_sequence_names)
        # Connect folder selection to update video list
        self.source_list.currentItemChanged.connect(self._on_folder_selected)
        # Video player signals
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
        self.play_btn.clicked.connect(self._toggle_play)
        self.stop_btn.clicked.connect(self._stop_video)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.video_slider.sliderMoved.connect(self._seek_video)
        # Image sequence signals
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        self.prev_image_btn.clicked.connect(self._prev_image)
        self.next_image_btn.clicked.connect(self._next_image)
        self.image_slider.valueChanged.connect(self._on_image_slider_changed)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.zoom_reset_btn.clicked.connect(self._zoom_reset)
        # Trim slider signals
        self.trim_slider.trimChanged.connect(self._on_trim_changed)

    def _add_source_folder(self, folder_path: Optional[str] = None) -> None:
        """Add a source folder via file dialog or direct path.

        Args:
            folder_path: Optional path to add directly (for drag-drop).
        """
        if folder_path:
            path = folder_path
        else:
            start_dir = self.last_directory or ""
            path = QFileDialog.getExistingDirectory(
                self, "Select Source Folder", start_dir
            )

        if path:
            folder = Path(path)
            if folder.is_dir() and folder not in self.source_folders:
                self.source_folders.append(folder)
                self.source_list.addItem(str(folder))
                self.last_directory = str(folder.parent)
                self._refresh_files()
                # Auto-select the newly added folder to show its videos
                self.source_list.setCurrentRow(self.source_list.count() - 1)

    def _remove_source_folder(self) -> None:
        """Remove selected source folder(s)."""
        selected = self.source_list.selectedItems()
        if not selected:
            return

        # Remove in reverse order to maintain correct indices
        rows = sorted([self.source_list.row(item) for item in selected], reverse=True)
        for row in rows:
            self.source_list.takeItem(row)
            del self.source_folders[row]
        self._refresh_files()

    def _remove_selected_files(self) -> None:
        """Remove selected files from the file list."""
        selected = self.file_list.selectedItems()
        if not selected:
            return

        # Remove in reverse order to maintain correct indices
        rows = sorted([self.file_list.indexOfTopLevelItem(item) for item in selected], reverse=True)
        for row in rows:
            self.file_list.takeTopLevelItem(row)

    def _browse_destination(self) -> None:
        """Select destination folder via file dialog."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", start_dir
        )
        if path:
            self.dst_path.setText(path)
            self.last_directory = str(Path(path).parent)
            self._try_resume_session(path)

    def _on_destination_changed(self) -> None:
        """Handle destination path text field changes."""
        path = self.dst_path.text().strip()
        if path and Path(path).is_dir():
            resolved = str(Path(path).resolve())
            # Only try resume if this is a new destination
            if resolved != self._last_resumed_dest:
                self._try_resume_session(path)

    def _try_resume_session(self, dest_path: str) -> bool:
        """Try to resume a previous session for the given destination.

        Checks if a session exists for this destination, extracts source folders
        from recorded symlinks, and populates the UI with files that still exist.
        Also restores trim settings.

        Args:
            dest_path: Path to the destination folder.

        Returns:
            True if a session was resumed, False otherwise.
        """
        dest = Path(dest_path).resolve()
        dest_str = str(dest)

        # Track that we've checked this destination
        self._last_resumed_dest = dest_str

        sessions = self.db.get_sessions_by_destination(dest_str)

        if not sessions:
            return False

        # Get the most recent session
        latest_session = sessions[0]
        symlinks = self.db.get_symlinks_by_session(latest_session.id)

        if not symlinks:
            return False

        # Load trim settings from database
        db_trim_settings = self.db.get_all_trim_settings(latest_session.id)

        # Parse folder and file indices from link names
        # New format: seqNN_NNNN.ext, Old format: seq_NNNN.ext
        new_pattern = re.compile(r'seq(\d+)_(\d+)')
        old_pattern = re.compile(r'seq_(\d+)')

        # Collect folder info: {folder_path: (folder_idx, [(file_idx, filename)])}
        folder_data: dict[str, tuple[int, list[tuple[int, str]]]] = {}
        missing_count = 0

        for link in symlinks:
            source_path = Path(link.source_path)
            if not source_path.exists():
                missing_count += 1
                continue

            folder = str(source_path.parent)
            link_name = Path(link.link_path).stem

            # Try new format first
            match = new_pattern.match(link_name)
            if match:
                folder_idx = int(match.group(1)) - 1  # Convert to 0-based
                file_idx = int(match.group(2))
            else:
                # Try old format (single sequence)
                match = old_pattern.match(link_name)
                if match:
                    folder_idx = 0
                    file_idx = int(match.group(1))
                else:
                    # Unknown format, use sequence_number from db
                    folder_idx = 0
                    file_idx = link.sequence_number

            if folder not in folder_data:
                folder_data[folder] = (folder_idx, [])
            folder_data[folder][1].append((file_idx, link.original_filename))

        if not folder_data:
            return False

        # Sort folders by their index, then sort files within each folder
        sorted_folders = sorted(folder_data.items(), key=lambda x: x[1][0])

        # Clear and populate source folders
        self.source_folders.clear()
        self.source_list.clear()
        self._folder_trim_settings.clear()

        for folder, (folder_idx, file_list) in sorted_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                self.source_folders.append(folder_path)
                self.source_list.addItem(folder)
                # Restore trim settings for this folder
                if folder in db_trim_settings:
                    self._folder_trim_settings[folder_path] = db_trim_settings[folder]

        # Store session ID
        self._current_session_id = latest_session.id

        # Call _refresh_files to properly populate file list with trim settings applied
        self._refresh_files()

        # Notify user
        total_files = self.file_list.topLevelItemCount()
        trim_count = sum(1 for ts in self._folder_trim_settings.values() if ts[0] > 0 or ts[1] > 0)
        msg = f"Resumed session from {latest_session.created_at.strftime('%Y-%m-%d %H:%M')}.\n"
        msg += f"Loaded {total_files} files from {len(self.source_folders)} folder(s)."
        if trim_count > 0:
            msg += f"\nRestored trim settings for {trim_count} folder(s)."
        if missing_count > 0:
            msg += f"\n{missing_count} file(s) no longer exist and were skipped."

        QMessageBox.information(self, "Session Resumed", msg)
        return True

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        in_image_tab = self.preview_tabs.currentWidget() == self.image_tab

        if event.key() == Qt.Key.Key_Delete:
            if self.file_list.hasFocus():
                self._remove_selected_files()
            elif self.source_list.hasFocus():
                self._remove_source_folder()
            elif in_image_tab:
                # Delete current image from sequence
                self._delete_current_image()
        elif event.key() == Qt.Key.Key_Left:
            if in_image_tab:
                self._prev_image()
        elif event.key() == Qt.Key.Key_Right:
            if in_image_tab:
                self._next_image()
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            if in_image_tab:
                self._zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            if in_image_tab:
                self._zoom_out()
        elif event.key() == Qt.Key.Key_0:
            if in_image_tab:
                self._zoom_reset()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Clean up media player when window closes."""
        self.media_player.stop()
        super().closeEvent(event)

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom in image tab."""
        if self.preview_tabs.currentWidget() == self.image_tab:
            # Check if mouse is over the image scroll area
            if self.image_scroll.underMouse():
                delta = event.angleDelta().y()
                if delta > 0:
                    self._zoom_in()
                elif delta < 0:
                    self._zoom_out()
                event.accept()
                return
        super().wheelEvent(event)

    def eventFilter(self, obj, event) -> bool:
        """Handle mouse events for panning the image."""
        if obj == self.image_scroll.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._pan_start = event.pos()
                    self._pan_scrollbar_start = QPoint(
                        self.image_scroll.horizontalScrollBar().value(),
                        self.image_scroll.verticalScrollBar().value()
                    )
                    self.image_scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif event.type() == QEvent.Type.MouseMove:
                if self._pan_start is not None:
                    delta = event.pos() - self._pan_start
                    self.image_scroll.horizontalScrollBar().setValue(
                        self._pan_scrollbar_start.x() - delta.x()
                    )
                    self.image_scroll.verticalScrollBar().setValue(
                        self._pan_scrollbar_start.y() - delta.y()
                    )
                    return True

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._pan_start is not None:
                    self._pan_start = None
                    self._pan_scrollbar_start = None
                    self.image_scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
                    return True

        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events with URLs (folders)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped folders."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).is_dir():
                self._add_source_folder(path)

    def _on_folders_reordered(self) -> None:
        """Handle folder list reordering."""
        # Rebuild source_folders from current list order
        self.source_folders.clear()
        for i in range(self.source_list.count()):
            item = self.source_list.item(i)
            self.source_folders.append(Path(item.text()))
        self._refresh_files()

    def _refresh_files(self, select_position: str = 'first') -> None:
        """Refresh the file list from all source folders, applying trim settings.

        Args:
            select_position: Which item to select after refresh.
                'first' - select first item (default)
                'last' - select last item
                'none' - don't change selection
        """
        self.file_list.clear()
        if not self.source_folders:
            self._folder_file_counts.clear()
            return

        # Build folder index map
        folder_to_index = {folder: i for i, folder in enumerate(self.source_folders)}

        # Get all files from all folders
        all_files = self.manager.get_supported_files(self.source_folders)

        # Group files by folder first to get total counts
        files_by_folder: dict[Path, list[str]] = {}
        for source_dir, filename in all_files:
            if source_dir not in files_by_folder:
                files_by_folder[source_dir] = []
            files_by_folder[source_dir].append(filename)

        # Store total file counts per folder (before trimming)
        self._folder_file_counts = {folder: len(files) for folder, files in files_by_folder.items()}

        # Apply trim settings and build file list
        folder_file_counts: dict[Path, int] = {}  # For sequence numbering after trim
        for folder in self.source_folders:
            if folder not in files_by_folder:
                continue

            folder_files = files_by_folder[folder]
            total_in_folder = len(folder_files)

            # Get trim settings for this folder
            trim_start, trim_end = self._folder_trim_settings.get(folder, (0, 0))

            # Clamp trim values to valid range
            trim_start = min(trim_start, max(0, total_in_folder - 1))
            trim_end = min(trim_end, max(0, total_in_folder - 1 - trim_start))

            # Apply trim - slice the file list
            end_idx = total_in_folder - trim_end
            trimmed_files = folder_files[trim_start:end_idx]

            folder_idx = folder_to_index.get(folder, 0)

            for filename in trimmed_files:
                file_idx = folder_file_counts.get(folder, 0)
                folder_file_counts[folder] = file_idx + 1

                # Generate sequence name preview
                ext = Path(filename).suffix
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"

                item = QTreeWidgetItem([seq_name, filename, str(folder)])
                # Store (source_dir, filename, folder_idx, file_idx) for symlink creation
                item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx))
                self.file_list.addTopLevelItem(item)

        # Update image slider and select appropriate item
        total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, total - 1))
        if total > 0 and select_position != 'none':
            if select_position == 'last':
                self.file_list.setCurrentItem(self.file_list.topLevelItem(total - 1))
            else:  # 'first' or default
                self.file_list.setCurrentItem(self.file_list.topLevelItem(0))

        # Update trim slider for currently selected folder
        self._update_trim_slider_for_selected_folder()

    def _get_files_in_order(self) -> list[tuple[Path, str, int, int]]:
        """Get files in the current list order with sequence info.

        Returns:
            List of (source_dir, filename, folder_idx, file_idx) tuples.
        """
        files = []
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                files.append(data)
        return files

    def _recalculate_sequence_names(self) -> None:
        """Recalculate sequence names after file reordering."""
        if not self.source_folders:
            return

        folder_to_index = {folder: i for i, folder in enumerate(self.source_folders)}
        folder_file_counts: dict[Path, int] = {}

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                source_dir = data[0]
                filename = data[1]
                folder_idx = folder_to_index.get(source_dir, 0)
                file_idx = folder_file_counts.get(source_dir, 0)
                folder_file_counts[source_dir] = file_idx + 1

                # Update sequence name
                ext = Path(filename).suffix
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                item.setText(0, seq_name)

                # Update stored data
                item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename, folder_idx, file_idx))

    # --- Video Preview Methods ---

    def _get_videos_in_folder(self, folder: Path) -> list[Path]:
        """Get all video files in the parent folder of the source.

        The video representing a sequence is typically one level above
        the folder containing the images.

        Args:
            folder: Source folder path (videos are in its parent).

        Returns:
            List of video file paths, sorted alphabetically.
        """
        videos = []
        parent = folder.parent
        if parent.is_dir():
            for item in parent.iterdir():
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(item)
        return sorted(videos, key=lambda p: p.name.lower())

    def _on_folder_selected(self, current, previous) -> None:
        """Handle folder selection change - update video list and trim slider."""
        self._stop_video()
        self.video_combo.clear()

        if current is None:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No folder selected")
            return

        folder = Path(current.text())

        # Update trim slider for selected folder
        self._update_trim_slider_for_selected_folder()

        # Update video list
        videos = self._get_videos_in_folder(folder)

        if not videos:
            self.video_combo.addItem("No videos found")
            self.video_combo.setEnabled(False)
            return

        self.video_combo.setEnabled(True)
        for video in videos:
            self.video_combo.addItem(video.name, video)

        # Auto-select first video
        self.video_combo.setCurrentIndex(0)

    def _update_trim_slider_for_selected_folder(self) -> None:
        """Update the trim slider to reflect the currently selected folder."""
        current_item = self.source_list.currentItem()
        if current_item is None:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No folder selected")
            return

        folder = Path(current_item.text())
        total = self._folder_file_counts.get(folder, 0)

        if total == 0:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No images in folder")
            return

        # Get current trim settings
        trim_start, trim_end = self._folder_trim_settings.get(folder, (0, 0))

        # Update trim slider
        self.trim_slider.setEnabled(True)
        self.trim_slider.setRange(total)
        self.trim_slider.setTrimStart(trim_start)
        self.trim_slider.setTrimEnd(trim_end)

        # Update label
        self._update_trim_label(folder, total, trim_start, trim_end)

    def _update_trim_label(self, folder: Path, total: int, trim_start: int, trim_end: int) -> None:
        """Update the trim label to show current trim range."""
        included_start = trim_start + 1  # 1-based for display
        included_end = total - trim_end
        included_count = included_end - trim_start

        if trim_start == 0 and trim_end == 0:
            self.trim_label.setText(f"Frames: All {total} included")
        elif included_count <= 0:
            self.trim_label.setText(f"Frames: None included (all {total} trimmed)")
        else:
            self.trim_label.setText(f"Frames {included_start}-{included_end} of {total} ({included_count} included)")

    def _on_trim_changed(self, trim_start: int, trim_end: int, handle: str) -> None:
        """Handle trim slider value changes.

        Args:
            trim_start: Number of frames trimmed from start.
            trim_end: Number of frames trimmed from end.
            handle: Which handle was dragged ('left' or 'right').
        """
        current_item = self.source_list.currentItem()
        if current_item is None:
            return

        folder = Path(current_item.text())
        total = self._folder_file_counts.get(folder, 0)

        # Store trim settings
        self._folder_trim_settings[folder] = (trim_start, trim_end)

        # Update label
        self._update_trim_label(folder, total, trim_start, trim_end)

        # Refresh file list to apply new trim settings (don't auto-select)
        self._refresh_files(select_position='none')

        # Select first or last image OF THE CURRENT FOLDER based on which handle was dragged
        # Left handle (trim start) -> show first visible frame of this folder
        # Right handle (trim end) -> show last visible frame of this folder
        self._select_folder_boundary(folder, 'first' if handle == 'left' else 'last')

    def _select_folder_boundary(self, folder: Path, position: str) -> None:
        """Select the first or last file of a specific folder in the file list.

        Args:
            folder: The folder whose files to search.
            position: 'first' or 'last'.
        """
        folder_str = str(folder)
        matching_indices = []

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and str(data[0]) == folder_str:
                matching_indices.append(i)

        if not matching_indices:
            return

        if position == 'last':
            select_idx = matching_indices[-1]
        else:
            select_idx = matching_indices[0]

        item = self.file_list.topLevelItem(select_idx)
        self.file_list.setCurrentItem(item)
        self.image_slider.setValue(select_idx)
        self._show_image_at_index(select_idx)

    def _on_video_selected(self, index: int) -> None:
        """Handle video selection from combo box."""
        self._stop_video()

        if index < 0:
            return

        video_path = self.video_combo.currentData()
        if video_path and isinstance(video_path, Path) and video_path.exists():
            self.media_player.setSource(QUrl.fromLocalFile(str(video_path)))

    def _toggle_play(self) -> None:
        """Toggle play/pause state."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("Play")
        else:
            self.media_player.play()
            self.play_btn.setText("Pause")

    def _stop_video(self) -> None:
        """Stop video playback."""
        self.media_player.stop()
        self.play_btn.setText("Play")
        self.video_slider.setValue(0)
        self.video_time_label.setText("00:00 / 00:00")

    def _on_position_changed(self, position: int) -> None:
        """Update slider and time label when playback position changes."""
        self.video_slider.setValue(position)
        self._update_time_label(position, self.media_player.duration())

    def _on_duration_changed(self, duration: int) -> None:
        """Update slider range when video duration is known."""
        self.video_slider.setRange(0, duration)
        self._update_time_label(self.media_player.position(), duration)

    def _seek_video(self, position: int) -> None:
        """Seek to a position in the video."""
        self.media_player.setPosition(position)

    def _update_time_label(self, position: int, duration: int) -> None:
        """Update the time label with current position and duration."""
        def format_time(ms: int) -> str:
            seconds = ms // 1000
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes:02d}:{seconds:02d}"

        self.video_time_label.setText(f"{format_time(position)} / {format_time(duration)}")

    # --- Image Sequence Preview Methods ---

    def _on_file_selected(self, current, previous) -> None:
        """Handle file selection in the list - update image preview."""
        if current is None:
            return

        # Update slider range based on total files
        total = self.file_list.topLevelItemCount()
        current_index = self.file_list.indexOfTopLevelItem(current)

        self.image_slider.setRange(0, max(0, total - 1))
        self.image_slider.setValue(current_index)

        self._show_image_at_index(current_index)

    def _show_image_at_index(self, index: int) -> None:
        """Display the image at the given index in the file list."""
        if index < 0 or index >= self.file_list.topLevelItemCount():
            self._current_pixmap = None
            return

        item = self.file_list.topLevelItem(index)
        if item is None:
            self._current_pixmap = None
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            self._current_pixmap = None
            return

        source_dir, filename = data[0], data[1]
        image_path = source_dir / filename

        if not image_path.exists():
            self.image_label.setText(f"Image not found:\n{image_path}")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Load and display image
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Cannot load image:\n{image_path}")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Store pixmap for zooming
        self._current_pixmap = pixmap
        self._apply_zoom()

        # Update labels
        total = self.file_list.topLevelItemCount()
        self.image_index_label.setText(f"{index + 1} / {total}")
        seq_name = item.text(0)
        self.image_name_label.setText(f"{seq_name} ({filename})")

        # Select the item in the file list
        self.file_list.setCurrentItem(item)

    def _apply_zoom(self) -> None:
        """Apply current zoom level to the image."""
        if self._current_pixmap is None:
            return

        if self._zoom_level == 1.0:
            # Fit to scroll area
            scaled = self._current_pixmap.scaled(
                self.image_scroll.size() * 0.95,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # Apply zoom level
            new_size = self._current_pixmap.size() * self._zoom_level
            scaled = self._current_pixmap.scaled(
                new_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        self.image_label.setPixmap(scaled)
        self.zoom_label.setText(f"{int(self._zoom_level * 100)}%")

    def _zoom_in(self) -> None:
        """Zoom in on the image."""
        if self._zoom_level < 5.0:
            self._zoom_level = min(5.0, self._zoom_level * 1.25)
            self._apply_zoom()

    def _zoom_out(self) -> None:
        """Zoom out on the image."""
        if self._zoom_level > 0.1:
            self._zoom_level = max(0.1, self._zoom_level / 1.25)
            self._apply_zoom()

    def _zoom_reset(self) -> None:
        """Reset zoom to fit the scroll area."""
        self._zoom_level = 1.0
        self._apply_zoom()

    def _delete_current_image(self) -> None:
        """Delete the currently displayed image from the sequence."""
        current_index = self.image_slider.value()
        total = self.file_list.topLevelItemCount()

        if total == 0 or current_index < 0 or current_index >= total:
            return

        # Remove from file list
        self.file_list.takeTopLevelItem(current_index)
        self._recalculate_sequence_names()

        # Update slider range
        new_total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, new_total - 1))

        if new_total == 0:
            self.image_label.clear()
            self.image_name_label.setText("")
            self.image_index_label.setText("0 / 0")
            self._current_pixmap = None
        else:
            # Show next image (or previous if we deleted the last one)
            new_index = min(current_index, new_total - 1)
            self.image_slider.setValue(new_index)
            self._show_image_at_index(new_index)

    def _prev_image(self) -> None:
        """Show the previous image in the sequence."""
        current = self.image_slider.value()
        if current > 0:
            self.image_slider.setValue(current - 1)

    def _next_image(self) -> None:
        """Show the next image in the sequence."""
        current = self.image_slider.value()
        if current < self.image_slider.maximum():
            self.image_slider.setValue(current + 1)

    def _on_image_slider_changed(self, value: int) -> None:
        """Handle image slider movement."""
        self._show_image_at_index(value)

    def _process_links(self) -> None:
        """Create symlinks based on current configuration."""
        dst = self.dst_path.text()

        if not self.source_folders:
            QMessageBox.warning(self, "Error", "Add at least one source folder!")
            return

        if not dst:
            QMessageBox.warning(self, "Error", "Select a destination folder!")
            return

        files = self._get_files_in_order()
        if not files:
            QMessageBox.warning(self, "Error", "No files to process!")
            return

        try:
            results, session_id = self.manager.create_sequence_links(
                sources=self.source_folders,
                dest=Path(dst),
                files=files,
                trim_settings=self._folder_trim_settings
            )

            # Store session ID for potential future use
            self._current_session_id = session_id

            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)

            if failed > 0:
                QMessageBox.warning(
                    self, "Partial Success",
                    f"Linked {successful} files, {failed} failed.\n"
                    f"Destination: {dst}"
                )
            else:
                QMessageBox.information(
                    self, "Success",
                    f"Linked {successful} files to {dst}"
                )

        except SymlinkError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))


# --- CLI ---
def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI mode.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog='symlink',
        description='Video Montage Linker - Create sequenced symlinks for image files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Launch GUI
  %(prog)s --gui                        Launch GUI
  %(prog)s --src /path/to/images --dst /path/to/dest
                                        Create symlinks from CLI
  %(prog)s --src /folder1 --src /folder2 --dst /path/to/dest
                                        Merge multiple source folders
  %(prog)s --list                       List tracked symlink sessions
  %(prog)s --clean /path/to/dest        Remove symlinks and session for destination
        """
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch the graphical interface'
    )

    parser.add_argument(
        '--src',
        action='append',
        metavar='PATH',
        help='Source folder(s) containing images (can be used multiple times)'
    )

    parser.add_argument(
        '--dst',
        metavar='PATH',
        help='Destination folder for symlinks'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all tracked symlink sessions'
    )

    parser.add_argument(
        '--clean',
        metavar='PATH',
        help='Clean up symlinks and remove session for the specified destination'
    )

    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Execute CLI commands.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    db = DatabaseManager()

    # List sessions
    if args.list:
        sessions = db.get_sessions()
        if not sessions:
            print("No symlink sessions found.")
            return 0

        print(f"{'ID':<6} {'Created':<20} {'Links':<8} Destination")
        print("-" * 80)
        for session in sessions:
            created = session.created_at.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{session.id:<6} {created:<20} {session.link_count:<8} {session.destination}")
        return 0

    # Clean up destination
    if args.clean:
        dest = Path(args.clean).resolve()

        # Remove symlinks from filesystem
        if dest.exists():
            try:
                removed = SymlinkManager.cleanup_old_links(dest)
                print(f"Removed {removed} symlinks from {dest}")
            except CleanupError as e:
                print(f"Error cleaning up files: {e}", file=sys.stderr)
                return 1

        # Remove from database
        sessions = db.get_sessions_by_destination(str(dest))
        for session in sessions:
            db.delete_session(session.id)
            print(f"Removed session {session.id} from database")

        if not sessions:
            print("No sessions found for this destination in database.")

        return 0

    # Create symlinks
    if args.src and args.dst:
        sources = [Path(s).resolve() for s in args.src]
        dest = Path(args.dst).resolve()

        manager = SymlinkManager(db)

        try:
            files = manager.get_supported_files(sources)
            if not files:
                print("No supported image files found in source folders.")
                return 1

            print(f"Found {len(files)} files in {len(sources)} source folder(s)")

            results, _ = manager.create_sequence_links(
                sources=sources,
                dest=dest,
                files=files
            )

            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)

            print(f"Created {successful} symlinks in {dest}")

            if failed > 0:
                print(f"Warning: {failed} operations failed", file=sys.stderr)
                for r in results:
                    if not r.success:
                        print(f"  - {r.source_path.name}: {r.error}", file=sys.stderr)
                return 1

            return 0

        except SymlinkError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # If src without dst or dst without src
    if args.src or args.dst:
        print("Error: Both --src and --dst are required for creating symlinks.",
              file=sys.stderr)
        return 1

    # No CLI args, show help
    create_parser().print_help()
    return 0


# --- Entry Point ---
def main() -> int:
    """Main entry point for the application.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Determine if we should launch GUI
    # GUI is launched if: --gui flag, OR no arguments at all
    launch_gui = args.gui or (
        not args.src and
        not args.dst and
        not args.list and
        not args.clean
    )

    if launch_gui:
        app = QApplication(sys.argv)
        window = SequenceLinkerUI()
        window.show()
        return app.exec()

    return run_cli(args)


if __name__ == '__main__':
    sys.exit(main())
