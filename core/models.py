"""Data models, enums, and exceptions for Video Montage Linker."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# --- Enums ---

class BlendCurve(Enum):
    """Blend curve types for cross-dissolve transitions."""
    LINEAR = 'linear'
    EASE_IN = 'ease_in'
    EASE_OUT = 'ease_out'
    EASE_IN_OUT = 'ease_in_out'


class BlendMethod(Enum):
    """Blend method types for transitions."""
    ALPHA = 'alpha'           # Simple cross-dissolve (PIL.Image.blend)
    OPTICAL_FLOW = 'optical'  # OpenCV Farneback optical flow
    RIFE = 'rife'             # AI frame interpolation (NCNN binary)
    RIFE_PRACTICAL = 'rife_practical'  # Practical-RIFE Python/PyTorch implementation


class FolderType(Enum):
    """Folder type for transition detection."""
    AUTO = 'auto'
    MAIN = 'main'
    TRANSITION = 'transition'


class DirectInterpolationMethod(Enum):
    """Method for direct frame interpolation between sequences."""
    RIFE = 'rife'
    FILM = 'film'


# --- Data Classes ---

@dataclass
class TransitionSettings:
    """Settings for cross-dissolve transitions."""
    enabled: bool = False
    blend_curve: BlendCurve = BlendCurve.LINEAR
    output_format: str = 'png'
    webp_method: int = 4  # 0-6, used when format is webp (compression effort)
    output_quality: int = 95  # used for jpeg only
    trans_destination: Optional[Path] = None  # separate destination for transition output
    blend_method: BlendMethod = BlendMethod.ALPHA  # blending method
    rife_binary_path: Optional[Path] = None  # path to rife-ncnn-vulkan binary
    rife_model: str = 'rife-v4.6'  # RIFE model to use
    rife_uhd: bool = False  # Enable UHD mode for high resolution
    rife_tta: bool = False  # Enable TTA mode for better quality
    # Practical-RIFE settings
    practical_rife_model: str = 'v4.25'  # v4.25, v4.26, v4.22, etc.
    practical_rife_ensemble: bool = False  # Ensemble mode for better quality (slower)
    # Optical flow settings
    of_preset: str = 'balanced'  # fast, balanced, quality, max
    of_levels: int = 3           # pyramid levels (1-7)
    of_winsize: int = 15         # window size (5-51, odd)
    of_iterations: int = 3       # iterations (1-10)
    of_poly_n: int = 5           # polynomial neighborhood (5 or 7)
    of_poly_sigma: float = 1.2   # gaussian sigma (0.5-2.0)


@dataclass
class PerTransitionSettings:
    """Per-transition overlap settings for cross-dissolves."""
    trans_folder: Path
    left_overlap: int = 16   # overlap count at left boundary (MAIN→TRANS)
    right_overlap: int = 16  # overlap count at right boundary (TRANS→MAIN)


@dataclass
class DirectTransitionSettings:
    """Settings for direct AI interpolation between sequences (no transition folder)."""
    after_folder: Path  # The folder after which this transition occurs
    frame_count: int = 16  # Number of interpolated frames to generate
    method: DirectInterpolationMethod = DirectInterpolationMethod.FILM
    enabled: bool = True


@dataclass
class VideoPreset:
    """Preset for video encoding via ffmpeg."""
    label: str           # Display name
    container: str       # 'mp4' or 'webm'
    codec: str           # ffmpeg codec: libx264, libx265, libvpx-vp9, libaom-av1
    crf: int
    pixel_format: str = 'yuv420p'
    preset: str = 'medium'           # x264/x265 speed preset
    max_height: Optional[int] = None  # Downscale filter
    extra_args: list[str] = field(default_factory=list)

VIDEO_PRESETS: dict[str, VideoPreset] = {
    'web_streaming':  VideoPreset('Web Streaming',   'mp4',  'libx264',    23, preset='medium'),
    'high_quality':   VideoPreset('High Quality',    'mp4',  'libx264',    18, preset='slow'),
    'archive':        VideoPreset('Archive (H.265)', 'mp4',  'libx265',    18, preset='slow', extra_args=['-tag:v', 'hvc1']),
    'social_media':   VideoPreset('Social Media',    'mp4',  'libx264',    23, preset='fast', max_height=1080),
    'fast_preview':   VideoPreset('Fast Preview',    'mp4',  'libx264',    28, preset='ultrafast'),
    'webm_vp9':       VideoPreset('WebM VP9',        'webm', 'libvpx-vp9', 30, extra_args=['-b:v', '0']),
    'webm_av1':       VideoPreset('WebM AV1',        'webm', 'libaom-av1', 30, extra_args=['-b:v', '0', '-strict', 'experimental']),
    'godot_theora':   VideoPreset('Godot (Theora)',   'ogv',  'libtheora',   8, extra_args=['-g', '512']),
}


@dataclass
class BlendResult:
    """Result of an image blend operation."""
    output_path: Path
    source_a: Path
    source_b: Path
    blend_factor: float
    success: bool
    error: Optional[str] = None


@dataclass
class TransitionSpec:
    """Specification for a transition boundary between two folders."""
    main_folder: Path
    trans_folder: Path
    main_files: list[str]
    trans_files: list[str]
    left_overlap: int   # asymmetric: frames from main folder end
    right_overlap: int  # asymmetric: frames from trans folder start
    # Indices into the overall file list
    main_start_idx: int
    trans_start_idx: int
    # Position indices in the folders list (for duplicate folder support)
    main_folder_idx: int = 0
    trans_folder_idx: int = 0


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
    name: Optional[str] = None


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
