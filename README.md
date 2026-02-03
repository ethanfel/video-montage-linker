# Video Montage Linker

A PyQt6 application for creating sequenced symlinks from image folders with advanced cross-dissolve transitions. Perfect for preparing image sequences for video editing, time-lapse assembly, or montage creation.

## Features

### Source Folder Management
- Add multiple source folders to merge images from different locations
- Drag & drop folders directly onto the window
- Alternating folder types: odd positions = **Main**, even positions = **Transition**
- Override folder types via right-click context menu
- Reorder folders with up/down buttons
- Per-folder trim settings (exclude frames from start/end)

### Cross-Dissolve Transitions
Smooth blending between folder boundaries with four blend methods:

| Method | Description | Quality | Speed |
|--------|-------------|---------|-------|
| **Cross-Dissolve** | Simple alpha blend | Good | Fastest |
| **Optical Flow** | Motion-compensated blend using OpenCV Farneback | Better | Medium |
| **RIFE (ncnn)** | Neural network interpolation via rife-ncnn-vulkan | Best | Fast (GPU) |
| **RIFE (Practical)** | PyTorch-based Practical-RIFE (v4.25/v4.26) | Best | Medium (GPU) |

- **Asymmetric overlap**: Set different frame counts for each side of a transition
- **Blend curves**: Linear, Ease In, Ease Out, Ease In/Out
- **Output formats**: PNG, JPEG (with quality), WebP (lossless with method setting)
- **RIFE auto-download**: Automatically downloads rife-ncnn-vulkan binary
- **Practical-RIFE models**: Auto-downloads from Google Drive on first use

### Preview
- **Video Preview**: Play video files from source folders
- **Image Sequence Preview**: Browse frames with zoom (scroll wheel) and pan (drag)
- **Sequence Table**: 2-column view showing Main/Transition frame pairing
- **Trim Slider**: Visual frame range selection per folder

### Dual Export Destinations
- **Sequence destination**: Regular symlinks only
- **Transition destination**: Symlinks + blended transition frames

### Session Persistence
- SQLite database tracks all sessions and settings
- Resume previous session by selecting the same destination folder
- Restores: source folders, trim settings, folder types, transition settings, per-transition overlaps

## Installation

### Requirements
- Python 3.10+
- PyQt6
- Pillow
- NumPy
- OpenCV (optional, for Optical Flow)

```bash
pip install PyQt6 Pillow numpy opencv-python
```

**Note:** Practical-RIFE creates its own isolated venv with PyTorch. The `gdown` package is installed automatically for downloading models from Google Drive.

### RIFE ncnn (Optional)
For AI-powered frame interpolation using Vulkan GPU acceleration:
- Select **RIFE (ncnn)** as the blend method
- Click **Download** to auto-fetch [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
- Or specify a custom binary path
- Models: rife-v4.6, rife-v4.15-lite, etc.

### Practical-RIFE (Optional)
For PyTorch-based frame interpolation with latest models:
- Select **RIFE (Practical)** as the blend method
- Click **Setup PyTorch** to create an isolated venv with PyTorch (~2GB)
- Models auto-download from Google Drive on first use
- Available models: v4.26, v4.25, v4.22, v4.20, v4.18, v4.15
- Optional ensemble mode for higher quality (slower)

The venv is stored at `~/.cache/video-montage-linker/venv-rife/`

## Usage

### GUI Mode

```bash
python symlink.py        # Launch GUI (default)
python symlink.py --gui  # Explicit GUI launch
```

**Workflow:**
1. Add source folders (drag & drop or click "Add Folder")
2. Adjust trim settings per folder if needed (right-click or use trim slider)
3. Set destination folder(s)
4. Enable transitions and configure blend method/settings
5. Click **Export Sequence** or **Export with Transitions**

### CLI Mode

```bash
# Create symlinks from source folders
python symlink.py --src /path/to/folder1 --src /path/to/folder2 --dst /path/to/dest

# List all tracked sessions
python symlink.py --list

# Clean up symlinks and remove session
python symlink.py --clean /path/to/dest
```

## File Structure

```
video-montage-linker/
├── symlink.py          # Entry point, CLI
├── config.py           # Constants, paths
├── core/
│   ├── models.py       # Enums, dataclasses
│   ├── database.py     # SQLite session management
│   ├── blender.py      # Image blending, RIFE downloader, Practical-RIFE env
│   ├── rife_worker.py  # Practical-RIFE inference (runs in isolated venv)
│   └── manager.py      # Symlink operations
└── ui/
    ├── widgets.py      # TrimSlider, custom widgets
    └── main_window.py  # Main application window
```

## Supported Formats

**Images:** PNG, WEBP, JPG, JPEG, TIFF, BMP, EXR

**Videos (preview only):** MP4, MOV, AVI, MKV, WEBM

## Database

Session data stored at: `~/.config/video-montage-linker/symlinks.db`

## System Installation (Linux)

```bash
# Make executable
chmod +x symlink.py

# Add to PATH
ln -s /full/path/to/symlink.py ~/.local/bin/video-montage-linker

# Create desktop entry
cat > ~/.local/share/applications/video-montage-linker.desktop << 'EOF'
[Desktop Entry]
Name=Video Montage Linker
Comment=Create sequenced symlinks with cross-dissolve transitions
Exec=/full/path/to/symlink.py
Icon=emblem-symbolic-link
Terminal=false
Type=Application
Categories=Utility;Graphics;AudioVideo;
EOF

update-desktop-database ~/.local/share/applications/
```

## License

MIT
