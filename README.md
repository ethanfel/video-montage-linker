# Video Montage Linker

A PyQt6 application to create sequenced symlinks for image files. Useful for preparing image sequences for video editing or montage creation.

## Features

### Multiple Source Folders
- Add multiple source folders to merge images from different locations
- Files are ordered by folder (first added = first in sequence), then alphabetically within each folder
- Drag & drop folders directly onto the window to add them
- Multi-select support for removing folders

### File Management
- Two-column view showing filename and source path
- Drag & drop to reorder files in the sequence
- Multi-select files (Ctrl+click, Shift+click)
- Remove files with Delete key or "Remove Files" button
- Refresh to rescan source folders

### Symlink Creation
- Creates numbered symlinks (`seq_0000.png`, `seq_0001.png`, etc.)
- Uses relative paths for portability
- Automatically cleans up old `seq_*` links before creating new ones

### Session Tracking
- SQLite database tracks all symlink sessions
- Located at `~/.config/video-montage-linker/symlinks.db`
- List past sessions and clean up by destination

### Supported Formats
- PNG, WEBP, JPG, JPEG

## Installation

Requires Python 3 and PyQt6:

```bash
pip install PyQt6
```

## Usage

### GUI Mode

```bash
# Launch the graphical interface
python symlink.py
python symlink.py --gui
```

1. Click "Add Folder" or drag & drop folders to add source directories
2. Reorder files by dragging them in the list
3. Remove unwanted files (select + Delete key)
4. Select destination folder
5. Click "Generate Virtual Sequence"

### CLI Mode

```bash
# Create symlinks from a single source
python symlink.py --src /path/to/images --dst /path/to/dest

# Merge multiple source folders
python symlink.py --src /folder1 --src /folder2 --dst /path/to/dest

# List all tracked sessions
python symlink.py --list

# Clean up symlinks and remove session record
python symlink.py --clean /path/to/dest
```

## System Installation (Linux)

To add as a system application:

```bash
# Make executable and add to PATH
chmod +x symlink.py
ln -s /path/to/symlink.py ~/.local/bin/video-montage-linker

# Create desktop entry
cat > ~/.local/share/applications/video-montage-linker.desktop << 'EOF'
[Desktop Entry]
Name=Video Montage Linker
Comment=Create sequenced symlinks for image files
Exec=/path/to/symlink.py
Icon=emblem-symbolic-link
Terminal=false
Type=Application
Categories=Utility;Graphics;
EOF

# Update desktop database
update-desktop-database ~/.local/share/applications/
```

## License

MIT
