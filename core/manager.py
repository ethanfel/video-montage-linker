"""Symlink management for Video Montage Linker."""

import os
import re
from pathlib import Path
from typing import Optional

from config import SUPPORTED_EXTENSIONS
from .models import LinkResult, CleanupError, SourceNotFoundError, DestinationError
from .database import DatabaseManager


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
        """Remove existing seq* symlinks and temporary files from a directory.

        Handles all naming formats:
        - Old folder-indexed: seq01_0000.png
        - Continuous: seq_00000.png
        Also removes blended image files and film_temp_*.png temporaries.

        Args:
            directory: Directory to clean up.

        Returns:
            Number of files removed.

        Raises:
            CleanupError: If cleanup fails.
        """
        removed = 0
        seq_pattern = re.compile(
            r'^seq\d*_\d+\.(png|jpg|jpeg|webp)$', re.IGNORECASE
        )
        temp_pattern = re.compile(
            r'^film_temp_\d+\.png$', re.IGNORECASE
        )
        try:
            for item in directory.iterdir():
                should_remove = False
                if item.name.startswith("seq"):
                    if item.is_symlink():
                        should_remove = True
                    elif item.is_file() and seq_pattern.match(item.name):
                        should_remove = True
                elif item.is_file() and temp_pattern.match(item.name):
                    should_remove = True

                if should_remove:
                    item.unlink()
                    removed += 1
        except OSError as e:
            raise CleanupError(f"Failed to clean up old links: {e}") from e

        return removed

    @staticmethod
    def remove_orphan_files(directory: Path, keep_names: set[str]) -> int:
        """Remove seq* files and film_temp_* not in the keep set.

        Same pattern matching as cleanup_old_links but skips filenames
        present in keep_names.

        Args:
            directory: Directory to clean orphans from.
            keep_names: Set of filenames to keep.

        Returns:
            Number of files removed.

        Raises:
            CleanupError: If removal fails.
        """
        removed = 0
        seq_pattern = re.compile(
            r'^seq\d*_\d+\.(png|jpg|jpeg|webp)$', re.IGNORECASE
        )
        temp_pattern = re.compile(
            r'^film_temp_\d+\.png$', re.IGNORECASE
        )
        try:
            for item in directory.iterdir():
                if item.name in keep_names:
                    continue
                should_remove = False
                if item.name.startswith("seq"):
                    if item.is_symlink():
                        should_remove = True
                    elif item.is_file() and seq_pattern.match(item.name):
                        should_remove = True
                elif item.is_file() and temp_pattern.match(item.name):
                    should_remove = True

                if should_remove:
                    item.unlink()
                    removed += 1
        except OSError as e:
            raise CleanupError(f"Failed to remove orphan files: {e}") from e

        return removed

    @staticmethod
    def symlink_matches(link_path: Path, expected_source: Path) -> bool:
        """Check if existing symlink resolves to expected source."""
        if not link_path.is_symlink():
            return False
        try:
            return link_path.resolve() == expected_source.resolve()
        except OSError:
            return False

    @staticmethod
    def copy_matches(dest_path: Path, source_path: Path) -> bool:
        """Check if existing copy matches source.

        Fast path: size + mtime comparison.  If sizes match but mtimes
        differ, falls back to comparing file contents so that a
        re-export after touching (but not changing) the source is still
        skipped, while a genuine content change is caught.
        """
        if not dest_path.is_file() or dest_path.is_symlink():
            return False
        try:
            src_stat = source_path.stat()
            dst_stat = dest_path.stat()
            if src_stat.st_size != dst_stat.st_size:
                return False
            # Fast path: identical mtime means the copy2 wrote this file
            if abs(src_stat.st_mtime - dst_stat.st_mtime) < 2.0:
                return True
            # Size matches but mtime differs — compare contents
            return SymlinkManager._files_equal(source_path, dest_path)
        except OSError:
            return False

    @staticmethod
    def _files_equal(a: Path, b: Path, chunk_size: int = 65536) -> bool:
        """Compare two files by reading in chunks."""
        try:
            with open(a, 'rb') as fa, open(b, 'rb') as fb:
                while True:
                    ca = fa.read(chunk_size)
                    cb = fb.read(chunk_size)
                    if ca != cb:
                        return False
                    if not ca:
                        return True
        except OSError:
            return False

    def create_sequence_links(
        self,
        sources: list[Path],
        dest: Path,
        files: list[tuple],
        trim_settings: Optional[dict[Path, tuple[int, int]]] = None,
        copy_files: bool = False,
    ) -> tuple[list[LinkResult], Optional[int]]:
        """Create sequenced symlinks or copies from source files to destination.

        Args:
            sources: List of source directories (for validation).
            dest: Destination directory.
            files: List of tuples. Can be:
                   - (source_dir, filename) for CLI mode (uses global sequence)
                   - (source_dir, filename, folder_idx, file_idx) for GUI mode
            trim_settings: Optional dict mapping folder paths to (trim_start, trim_end).
            copy_files: If True, copy files instead of creating symlinks.

        Returns:
            Tuple of (list of LinkResult objects, session_id or None).
        """
        self.validate_paths(sources, dest)

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

        # Build planned names for orphan removal
        planned_names: set[str] = set()
        for file_data in files:
            _, fn, fi, fli = file_data
            ext = Path(fn).suffix
            planned_names.add(f"seq{fi + 1:02d}_{fli:04d}{ext}")

        for i, file_data in enumerate(files):
            source_dir, filename, folder_idx, file_idx = file_data
            source_path = source_dir / filename
            ext = source_path.suffix
            link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
            link_path = dest / link_name

            try:
                # Check if existing file already matches
                already_correct = False
                if link_path.exists() or link_path.is_symlink():
                    if copy_files:
                        already_correct = self.copy_matches(link_path, source_path)
                    else:
                        already_correct = self.symlink_matches(link_path, source_path)

                if not already_correct:
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()

                    if copy_files:
                        import shutil
                        shutil.copy2(source_path, link_path)
                    else:
                        rel_source = Path(os.path.relpath(source_path.resolve(), dest.resolve()))
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

        # Remove orphan seq*/film_temp_* files not in the planned set
        try:
            self.remove_orphan_files(dest, planned_names)
        except CleanupError:
            pass

        return results, session_id
