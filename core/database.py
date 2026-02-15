"""Database management for Video Montage Linker."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH
from .models import (
    BlendCurve,
    BlendMethod,
    FolderType,
    TransitionSettings,
    PerTransitionSettings,
    SymlinkRecord,
    SessionRecord,
    DatabaseError,
)


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
                    destination TEXT NOT NULL,
                    name TEXT DEFAULT NULL
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
                    folder_type TEXT DEFAULT 'auto',
                    UNIQUE(session_id, source_folder)
                );

                CREATE TABLE IF NOT EXISTS transition_settings (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    enabled INTEGER DEFAULT 0,
                    blend_curve TEXT DEFAULT 'linear',
                    output_format TEXT DEFAULT 'png',
                    webp_method INTEGER DEFAULT 4,
                    output_quality INTEGER DEFAULT 95,
                    trans_destination TEXT,
                    UNIQUE(session_id)
                );

                CREATE TABLE IF NOT EXISTS per_transition_settings (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    trans_folder TEXT NOT NULL,
                    left_overlap INTEGER DEFAULT 16,
                    right_overlap INTEGER DEFAULT 16,
                    UNIQUE(session_id, trans_folder)
                );

                CREATE TABLE IF NOT EXISTS removed_files (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    source_folder TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    UNIQUE(session_id, source_folder, filename)
                );

                CREATE TABLE IF NOT EXISTS direct_transition_settings (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    after_folder TEXT NOT NULL,
                    frame_count INTEGER DEFAULT 16,
                    method TEXT DEFAULT 'film',
                    enabled INTEGER DEFAULT 1,
                    UNIQUE(session_id, after_folder)
                );
            """)

            # Migration: add folder_type column if it doesn't exist
            try:
                conn.execute("SELECT folder_type FROM sequence_trim_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE sequence_trim_settings ADD COLUMN folder_type TEXT DEFAULT 'auto'")

            # Migration: add webp_method column if it doesn't exist
            try:
                conn.execute("SELECT webp_method FROM transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE transition_settings ADD COLUMN webp_method INTEGER DEFAULT 4")

            # Migration: add trans_destination column if it doesn't exist
            try:
                conn.execute("SELECT trans_destination FROM transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE transition_settings ADD COLUMN trans_destination TEXT")

            # Migration: add blend_method column if it doesn't exist
            try:
                conn.execute("SELECT blend_method FROM transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE transition_settings ADD COLUMN blend_method TEXT DEFAULT 'alpha'")

            # Migration: add rife_binary_path column if it doesn't exist
            try:
                conn.execute("SELECT rife_binary_path FROM transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE transition_settings ADD COLUMN rife_binary_path TEXT")

            # Migration: add folder_order column if it doesn't exist
            try:
                conn.execute("SELECT folder_order FROM sequence_trim_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE sequence_trim_settings ADD COLUMN folder_order INTEGER DEFAULT 0")

            # Migration: add name column to symlink_sessions if it doesn't exist
            try:
                conn.execute("SELECT name FROM symlink_sessions LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE symlink_sessions ADD COLUMN name TEXT DEFAULT NULL")

            # Migration: widen UNIQUE constraints to allow duplicate folder paths per session.
            # sequence_trim_settings: UNIQUE(session_id, source_folder) → UNIQUE(session_id, folder_order)
            self._migrate_unique_constraint(
                conn, 'sequence_trim_settings',
                """CREATE TABLE sequence_trim_settings_new (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    source_folder TEXT NOT NULL,
                    trim_start INTEGER DEFAULT 0,
                    trim_end INTEGER DEFAULT 0,
                    folder_type TEXT DEFAULT 'auto',
                    folder_order INTEGER DEFAULT 0,
                    UNIQUE(session_id, folder_order)
                )""",
                'session_id, source_folder, trim_start, trim_end, folder_type, folder_order',
            )

            # per_transition_settings: add folder_order, widen UNIQUE
            try:
                conn.execute("SELECT folder_order FROM per_transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE per_transition_settings ADD COLUMN folder_order INTEGER DEFAULT 0")
            self._migrate_unique_constraint(
                conn, 'per_transition_settings',
                """CREATE TABLE per_transition_settings_new (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    trans_folder TEXT NOT NULL,
                    left_overlap INTEGER DEFAULT 16,
                    right_overlap INTEGER DEFAULT 16,
                    folder_order INTEGER DEFAULT 0,
                    UNIQUE(session_id, trans_folder, folder_order)
                )""",
                'session_id, trans_folder, left_overlap, right_overlap, folder_order',
            )

            # removed_files: add folder_order, widen UNIQUE
            try:
                conn.execute("SELECT folder_order FROM removed_files LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE removed_files ADD COLUMN folder_order INTEGER DEFAULT 0")
            self._migrate_unique_constraint(
                conn, 'removed_files',
                """CREATE TABLE removed_files_new (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    source_folder TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    folder_order INTEGER DEFAULT 0,
                    UNIQUE(session_id, source_folder, filename, folder_order)
                )""",
                'session_id, source_folder, filename, folder_order',
            )

            # direct_transition_settings: add folder_order, widen UNIQUE
            try:
                conn.execute("SELECT folder_order FROM direct_transition_settings LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE direct_transition_settings ADD COLUMN folder_order INTEGER DEFAULT 0")
            self._migrate_unique_constraint(
                conn, 'direct_transition_settings',
                """CREATE TABLE direct_transition_settings_new (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES symlink_sessions(id) ON DELETE CASCADE,
                    after_folder TEXT NOT NULL,
                    frame_count INTEGER DEFAULT 16,
                    method TEXT DEFAULT 'film',
                    enabled INTEGER DEFAULT 1,
                    folder_order INTEGER DEFAULT 0,
                    UNIQUE(session_id, after_folder, folder_order)
                )""",
                'session_id, after_folder, frame_count, method, enabled, folder_order',
            )

            # Migration: remove overlap_frames from transition_settings (now per-transition)
            # We'll keep it for backward compatibility but won't use it

    @staticmethod
    def _migrate_unique_constraint(
        conn: sqlite3.Connection,
        table: str,
        create_new_sql: str,
        columns: str,
    ) -> None:
        """Recreate a table with a new UNIQUE constraint if needed.

        Tests whether duplicate folder_order=0 entries can be inserted.
        If an IntegrityError fires, the old constraint is too narrow and
        the table must be recreated.
        """
        new_table = f"{table}_new"
        try:
            # Test: can we insert two rows with same session+folder but different folder_order?
            # If the old UNIQUE is still (session_id, source_folder) this will fail.
            conn.execute(f"INSERT INTO {table} (session_id, {columns.split(',')[1].strip()}, folder_order) VALUES (-999, '__test__', 1)")
            conn.execute(f"INSERT INTO {table} (session_id, {columns.split(',')[1].strip()}, folder_order) VALUES (-999, '__test__', 2)")
            # Clean up test rows
            conn.execute(f"DELETE FROM {table} WHERE session_id = -999")
            # If we got here, the constraint already allows duplicates — no migration needed
            return
        except sqlite3.IntegrityError:
            # Old constraint is too narrow — need to recreate
            conn.execute(f"DELETE FROM {table} WHERE session_id = -999")
        except sqlite3.OperationalError:
            # Column might not exist yet or other issue — try migration anyway
            conn.execute(f"DELETE FROM {table} WHERE session_id = -999")

        try:
            conn.execute(f"DROP TABLE IF EXISTS {new_table}")
            conn.execute(create_new_sql)
            conn.execute(f"INSERT INTO {new_table} ({columns}) SELECT {columns} FROM {table}")
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {new_table} RENAME TO {table}")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            # Clean up failed migration attempt
            try:
                conn.execute(f"DROP TABLE IF EXISTS {new_table}")
            except sqlite3.OperationalError:
                pass

    def clear_session_data(self, session_id: int) -> None:
        """Delete all data for a session (symlinks, settings, etc.) but keep the session row."""
        try:
            with self._connect() as conn:
                for table in (
                    'symlinks', 'sequence_trim_settings', 'transition_settings',
                    'per_transition_settings', 'removed_files', 'direct_transition_settings',
                ):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to clear session data: {e}") from e

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_session(self, destination: str, name: Optional[str] = None) -> int:
        """Create a new linking session.

        Args:
            destination: The destination directory path.
            name: Optional display name (e.g. "autosave").

        Returns:
            The ID of the created session.

        Raises:
            DatabaseError: If session creation fails.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO symlink_sessions (destination, name) VALUES (?, ?)",
                    (destination, name)
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

    def record_symlinks_batch(
        self,
        session_id: int,
        records: list[tuple[str, str, str, int]],
    ) -> None:
        """Record multiple symlinks in a single transaction.

        Args:
            session_id: The session these symlinks belong to.
            records: List of (source, link, filename, seq) tuples.

        Raises:
            DatabaseError: If recording fails.
        """
        try:
            with self._connect() as conn:
                conn.executemany(
                    """INSERT INTO symlinks
                       (session_id, source_path, link_path, original_filename, sequence_number)
                       VALUES (?, ?, ?, ?, ?)""",
                    [(session_id, src, lnk, fname, seq) for src, lnk, fname, seq in records]
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to record symlinks: {e}") from e

    def get_sessions(self) -> list[SessionRecord]:
        """List all sessions with link counts.

        Returns:
            List of session records.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.id, s.created_at, s.destination, COUNT(l.id) as link_count, s.name
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
                link_count=row[3],
                name=row[4]
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
        """Delete a session and all its related data (CASCADE handles child tables).

        Args:
            session_id: The session ID to delete.

        Raises:
            DatabaseError: If deletion fails.
        """
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM symlink_sessions WHERE id = ?", (session_id,))
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete session: {e}") from e

    def delete_sessions(self, session_ids: list[int]) -> None:
        """Delete multiple sessions in a single transaction.

        Args:
            session_ids: List of session IDs to delete.

        Raises:
            DatabaseError: If deletion fails.
        """
        if not session_ids:
            return
        try:
            with self._connect() as conn:
                placeholders = ','.join('?' for _ in session_ids)
                conn.execute(
                    f"DELETE FROM symlink_sessions WHERE id IN ({placeholders})",
                    session_ids
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete sessions: {e}") from e

    def get_sessions_by_destination(self, dest: str) -> list[SessionRecord]:
        """Get all sessions for a destination directory.

        Args:
            dest: The destination directory path.

        Returns:
            List of session records.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.id, s.created_at, s.destination, COUNT(l.id) as link_count, s.name
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
                link_count=row[3],
                name=row[4]
            )
            for row in rows
        ]

    def save_trim_settings(
        self,
        session_id: int,
        source_folder: str,
        trim_start: int,
        trim_end: int,
        folder_type: FolderType = FolderType.AUTO,
        folder_order: int = 0,
    ) -> None:
        """Save trim settings for a folder in a session.

        Args:
            session_id: The session ID.
            source_folder: Path to the source folder.
            trim_start: Number of images to trim from start.
            trim_end: Number of images to trim from end.
            folder_type: The folder type (auto, main, or transition).
            folder_order: Position of this folder in source_folders list.

        Raises:
            DatabaseError: If saving fails.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO sequence_trim_settings
                       (session_id, source_folder, trim_start, trim_end, folder_type, folder_order)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, folder_order)
                       DO UPDATE SET source_folder=excluded.source_folder,
                                     trim_start=excluded.trim_start,
                                     trim_end=excluded.trim_end,
                                     folder_type=excluded.folder_type""",
                    (session_id, source_folder, trim_start, trim_end, folder_type.value, folder_order)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save trim settings: {e}") from e

    def get_trim_settings(
        self,
        session_id: int,
        source_folder: str
    ) -> tuple[int, int, FolderType]:
        """Get trim settings for a folder in a session.

        Args:
            session_id: The session ID.
            source_folder: Path to the source folder.

        Returns:
            Tuple of (trim_start, trim_end, folder_type). Returns (0, 0, AUTO) if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT trim_start, trim_end, folder_type FROM sequence_trim_settings
                   WHERE session_id = ? AND source_folder = ?""",
                (session_id, source_folder)
            ).fetchone()

        if row:
            try:
                folder_type = FolderType(row[2]) if row[2] else FolderType.AUTO
            except ValueError:
                folder_type = FolderType.AUTO
            return (row[0], row[1], folder_type)
        return (0, 0, FolderType.AUTO)

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

    def get_all_folder_settings(self, session_id: int) -> dict[str, tuple[int, int, FolderType]]:
        """Get all folder settings (trim + type) for a session, unordered.

        Returns:
            Dict mapping source_folder to (trim_start, trim_end, folder_type).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT source_folder, trim_start, trim_end, folder_type
                   FROM sequence_trim_settings WHERE session_id = ?""",
                (session_id,)
            ).fetchall()

        result = {}
        for row in rows:
            try:
                ft = FolderType(row[3]) if row[3] else FolderType.AUTO
            except ValueError:
                ft = FolderType.AUTO
            result[row[0]] = (row[1], row[2], ft)
        return result

    def get_ordered_folders(self, session_id: int) -> list[tuple[str, FolderType, int, int]]:
        """Get all folders for a session in saved order.

        Returns:
            List of (source_folder, folder_type, trim_start, trim_end) sorted by folder_order.
            Returns empty list if folder_order is not meaningful (all zeros from
            pre-migration sessions), so the caller falls back to symlink-derived order.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT source_folder, folder_type, trim_start, trim_end, folder_order
                   FROM sequence_trim_settings WHERE session_id = ?
                   ORDER BY folder_order""",
                (session_id,)
            ).fetchall()

        if not rows:
            return []

        # If all folder_order values are 0, this is a pre-migration session
        # where the ordering is not meaningful — return empty to trigger
        # the legacy symlink-derived ordering path.
        if len(rows) > 1 and all(row[4] == 0 for row in rows):
            return []

        result = []
        for row in rows:
            try:
                ft = FolderType(row[1]) if row[1] else FolderType.AUTO
            except ValueError:
                ft = FolderType.AUTO
            result.append((row[0], ft, row[2], row[3]))
        return result

    def save_transition_settings(
        self,
        session_id: int,
        settings: TransitionSettings
    ) -> None:
        """Save transition settings for a session.

        Args:
            session_id: The session ID.
            settings: TransitionSettings to save.

        Raises:
            DatabaseError: If saving fails.
        """
        try:
            trans_dest = str(settings.trans_destination) if settings.trans_destination else None
            rife_path = str(settings.rife_binary_path) if settings.rife_binary_path else None
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO transition_settings
                       (session_id, enabled, blend_curve, output_format, webp_method, output_quality, trans_destination, blend_method, rife_binary_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id)
                       DO UPDATE SET enabled=excluded.enabled,
                                     blend_curve=excluded.blend_curve,
                                     output_format=excluded.output_format,
                                     webp_method=excluded.webp_method,
                                     output_quality=excluded.output_quality,
                                     trans_destination=excluded.trans_destination,
                                     blend_method=excluded.blend_method,
                                     rife_binary_path=excluded.rife_binary_path""",
                    (session_id, 1 if settings.enabled else 0,
                     settings.blend_curve.value, settings.output_format,
                     settings.webp_method, settings.output_quality, trans_dest,
                     settings.blend_method.value, rife_path)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save transition settings: {e}") from e

    def get_transition_settings(self, session_id: int) -> Optional[TransitionSettings]:
        """Get transition settings for a session.

        Args:
            session_id: The session ID.

        Returns:
            TransitionSettings or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT enabled, blend_curve, output_format, webp_method, output_quality, trans_destination, blend_method, rife_binary_path
                   FROM transition_settings WHERE session_id = ?""",
                (session_id,)
            ).fetchone()

        if row:
            trans_dest = Path(row[5]) if row[5] else None
            try:
                blend_method = BlendMethod(row[6]) if row[6] else BlendMethod.ALPHA
            except ValueError:
                blend_method = BlendMethod.ALPHA
            rife_path = Path(row[7]) if row[7] else None
            return TransitionSettings(
                enabled=bool(row[0]),
                blend_curve=BlendCurve(row[1]),
                output_format=row[2],
                webp_method=row[3] if row[3] is not None else 4,
                output_quality=row[4],
                trans_destination=trans_dest,
                blend_method=blend_method,
                rife_binary_path=rife_path
            )
        return None

    def save_folder_type_override(
        self,
        session_id: int,
        folder: str,
        folder_type: FolderType,
        trim_start: int = 0,
        trim_end: int = 0
    ) -> None:
        """Save folder type override for a folder in a session.

        Args:
            session_id: The session ID.
            folder: Path to the source folder.
            folder_type: The folder type override.
            trim_start: Number of images to trim from start.
            trim_end: Number of images to trim from end.

        Raises:
            DatabaseError: If saving fails.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO sequence_trim_settings
                       (session_id, source_folder, trim_start, trim_end, folder_type)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, source_folder)
                       DO UPDATE SET trim_start=excluded.trim_start,
                                     trim_end=excluded.trim_end,
                                     folder_type=excluded.folder_type""",
                    (session_id, folder, trim_start, trim_end, folder_type.value)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save folder type override: {e}") from e

    def get_folder_type_overrides(self, session_id: int) -> dict[str, FolderType]:
        """Get all folder type overrides for a session.

        Args:
            session_id: The session ID.

        Returns:
            Dict mapping source folder paths to FolderType.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT source_folder, folder_type
                   FROM sequence_trim_settings WHERE session_id = ?""",
                (session_id,)
            ).fetchall()

        result = {}
        for row in rows:
            try:
                result[row[0]] = FolderType(row[1]) if row[1] else FolderType.AUTO
            except ValueError:
                result[row[0]] = FolderType.AUTO
        return result

    def save_per_transition_settings(
        self,
        session_id: int,
        settings: PerTransitionSettings,
        folder_order: int = 0,
    ) -> None:
        """Save per-transition overlap settings.

        Args:
            session_id: The session ID.
            settings: PerTransitionSettings to save.
            folder_order: Position of this folder in the source list.

        Raises:
            DatabaseError: If saving fails.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO per_transition_settings
                       (session_id, trans_folder, left_overlap, right_overlap, folder_order)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, folder_order)
                       DO UPDATE SET trans_folder=excluded.trans_folder,
                                     left_overlap=excluded.left_overlap,
                                     right_overlap=excluded.right_overlap""",
                    (session_id, str(settings.trans_folder),
                     settings.left_overlap, settings.right_overlap, folder_order)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save per-transition settings: {e}") from e

    def get_per_transition_settings(
        self,
        session_id: int,
        trans_folder: str
    ) -> Optional[PerTransitionSettings]:
        """Get per-transition settings for a specific transition folder.

        Args:
            session_id: The session ID.
            trans_folder: Path to the transition folder.

        Returns:
            PerTransitionSettings or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT left_overlap, right_overlap FROM per_transition_settings
                   WHERE session_id = ? AND trans_folder = ?""",
                (session_id, trans_folder)
            ).fetchone()

        if row:
            return PerTransitionSettings(
                trans_folder=Path(trans_folder),
                left_overlap=row[0],
                right_overlap=row[1]
            )
        return None

    def get_all_per_transition_settings(
        self,
        session_id: int
    ) -> list[tuple[str, int, int, int]]:
        """Get all per-transition settings for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of (trans_folder, left_overlap, right_overlap, folder_order) tuples.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT trans_folder, left_overlap, right_overlap, folder_order
                   FROM per_transition_settings WHERE session_id = ?
                   ORDER BY folder_order""",
                (session_id,)
            ).fetchall()

        return [(row[0], row[1], row[2], row[3]) for row in rows]

    def save_removed_files(
        self,
        session_id: int,
        source_folder: str,
        filenames: list[str],
        folder_order: int = 0,
    ) -> None:
        """Save removed files for a folder in a session.

        Args:
            session_id: The session ID.
            source_folder: Path to the source folder.
            filenames: List of removed filenames.
            folder_order: Position of this folder in the source list.
        """
        try:
            with self._connect() as conn:
                for filename in filenames:
                    conn.execute(
                        """INSERT OR IGNORE INTO removed_files
                           (session_id, source_folder, filename, folder_order)
                           VALUES (?, ?, ?, ?)""",
                        (session_id, source_folder, filename, folder_order)
                    )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save removed files: {e}") from e

    def get_removed_files(self, session_id: int) -> dict[int, set[str]]:
        """Get all removed files for a session, keyed by folder_order.

        Args:
            session_id: The session ID.

        Returns:
            Dict mapping folder_order to sets of removed filenames.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_folder, filename, folder_order FROM removed_files WHERE session_id = ?",
                (session_id,)
            ).fetchall()

        result: dict[int, set[str]] = {}
        for folder, filename, folder_order in rows:
            if folder_order not in result:
                result[folder_order] = set()
            result[folder_order].add(filename)
        return result

    def save_direct_transition(
        self,
        session_id: int,
        after_folder: str,
        frame_count: int,
        method: str,
        enabled: bool,
        folder_order: int = 0,
    ) -> None:
        """Save direct interpolation settings for a folder transition."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO direct_transition_settings
                       (session_id, after_folder, frame_count, method, enabled, folder_order)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, folder_order)
                       DO UPDATE SET after_folder=excluded.after_folder,
                                     frame_count=excluded.frame_count,
                                     method=excluded.method,
                                     enabled=excluded.enabled""",
                    (session_id, after_folder, frame_count, method, 1 if enabled else 0, folder_order)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save direct transition: {e}") from e

    def get_direct_transitions(self, session_id: int) -> list[tuple[str, int, str, bool, int]]:
        """Get direct interpolation settings for a session.

        Returns:
            List of (after_folder, frame_count, method, enabled, folder_order) tuples.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT after_folder, frame_count, method, enabled, folder_order "
                "FROM direct_transition_settings WHERE session_id = ?",
                (session_id,)
            ).fetchall()
        return [(r[0], r[1], r[2], bool(r[3]), r[4]) for r in rows]
