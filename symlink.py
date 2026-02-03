#!/usr/bin/env python3
"""Video Montage Linker - Create sequenced symlinks for image files.

Supports both GUI and CLI modes for creating numbered symlinks from one or more
source directories into a single destination directory.
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from core import (
    DatabaseManager,
    SymlinkManager,
    CleanupError,
)
from ui import SequenceLinkerUI


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
        from core import SymlinkError

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


def main() -> int:
    """Main entry point for the application.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Determine if we should launch GUI
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
