#!/usr/bin/env python3
"""Video Montage Linker - Create sequenced symlinks for image files.

Supports both GUI and CLI modes for creating numbered symlinks from one or more
source directories into a single destination directory.
"""

# --- Imports ---
import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
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
)

# --- Configuration ---
SUPPORTED_EXTENSIONS = ('.png', '.webp', '.jpg', '.jpeg')
DB_PATH = Path.home() / '.config' / 'video-montage-linker' / 'symlinks.db'


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
        """Remove existing seq_* symlinks from a directory.

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
                if item.name.startswith("seq_") and item.is_symlink():
                    item.unlink()
                    removed += 1
        except OSError as e:
            raise CleanupError(f"Failed to clean up old links: {e}") from e

        return removed

    def create_sequence_links(
        self,
        sources: list[Path],
        dest: Path,
        files: list[tuple[Path, str]],
    ) -> list[LinkResult]:
        """Create sequenced symlinks from source files to destination.

        Args:
            sources: List of source directories (for validation).
            dest: Destination directory.
            files: List of (source_dir, filename) tuples in desired order.

        Returns:
            List of LinkResult objects for each operation.
        """
        self.validate_paths(sources, dest)
        self.cleanup_old_links(dest)

        session_id = None
        if self.db:
            session_id = self.db.create_session(str(dest))

        results: list[LinkResult] = []

        for i, (source_dir, filename) in enumerate(files):
            source_path = source_dir / filename
            ext = source_path.suffix
            link_name = f"seq_{i:04d}{ext}"
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

        return results


# --- GUI ---
class SequenceLinkerUI(QWidget):
    """PyQt6 GUI for the Video Montage Linker."""

    def __init__(self) -> None:
        """Initialize the UI."""
        super().__init__()
        self.source_folders: list[Path] = []
        self.last_directory: Optional[str] = None
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
        self.setMinimumSize(700, 600)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Source folders group
        self.source_group = QGroupBox("Source Folders (drag & drop folders here)")
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(100)
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.add_source_btn = QPushButton("Add Folder")
        self.remove_source_btn = QPushButton("Remove Folder")

        # Destination
        self.dst_label = QLabel("Destination Folder:")
        self.dst_path = QLineEdit(placeholderText="Select destination folder")
        self.dst_btn = QPushButton("Browse")

        # File list
        self.files_label = QLabel("Sequence Order (Drag to reorder, Del to remove):")
        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["Filename", "Source Folder"])
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

        # Assemble main layout
        main_layout.addWidget(self.source_group)
        main_layout.addWidget(self.dst_label)
        main_layout.addLayout(dst_layout)
        main_layout.addWidget(self.files_label)
        main_layout.addWidget(self.file_list)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.run_btn)

        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.add_source_btn.clicked.connect(self._add_source_folder)
        self.remove_source_btn.clicked.connect(self._remove_source_folder)
        self.dst_btn.clicked.connect(self._browse_destination)
        self.remove_files_btn.clicked.connect(self._remove_selected_files)
        self.refresh_btn.clicked.connect(self._refresh_files)
        self.run_btn.clicked.connect(self._process_links)

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

    def keyPressEvent(self, event) -> None:
        """Handle key press events for deletion."""
        if event.key() == Qt.Key.Key_Delete:
            # Check which widget has focus
            if self.file_list.hasFocus():
                self._remove_selected_files()
            elif self.source_list.hasFocus():
                self._remove_source_folder()
        else:
            super().keyPressEvent(event)

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

    def _refresh_files(self) -> None:
        """Refresh the file list from all source folders."""
        self.file_list.clear()
        if not self.source_folders:
            return

        files = self.manager.get_supported_files(self.source_folders)
        for source_dir, filename in files:
            item = QTreeWidgetItem([filename, str(source_dir)])
            item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename))
            self.file_list.addTopLevelItem(item)

    def _get_files_in_order(self) -> list[tuple[Path, str]]:
        """Get files in the current list order.

        Returns:
            List of (source_dir, filename) tuples in display order.
        """
        files = []
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                files.append(data)
        return files

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
            results = self.manager.create_sequence_links(
                sources=self.source_folders,
                dest=Path(dst),
                files=files
            )

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

            results = manager.create_sequence_links(
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
