"""Configuration constants for Video Montage Linker."""

from pathlib import Path

# Supported file extensions
SUPPORTED_EXTENSIONS = ('.png', '.webp', '.jpg', '.jpeg')
VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v')

# Database path
DB_PATH = Path.home() / '.config' / 'video-montage-linker' / 'symlinks.db'
