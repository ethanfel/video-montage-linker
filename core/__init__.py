"""Core modules for Video Montage Linker."""

from .models import (
    BlendCurve,
    BlendMethod,
    FolderType,
    TransitionSettings,
    PerTransitionSettings,
    BlendResult,
    TransitionSpec,
    LinkResult,
    SymlinkRecord,
    SessionRecord,
    SymlinkError,
    PathValidationError,
    SourceNotFoundError,
    DestinationError,
    CleanupError,
    DatabaseError,
)
from .database import DatabaseManager
from .blender import ImageBlender, TransitionGenerator, RifeDownloader, PracticalRifeEnv, OPTICAL_FLOW_PRESETS
from .manager import SymlinkManager

__all__ = [
    'BlendCurve',
    'BlendMethod',
    'FolderType',
    'TransitionSettings',
    'PerTransitionSettings',
    'BlendResult',
    'TransitionSpec',
    'LinkResult',
    'SymlinkRecord',
    'SessionRecord',
    'SymlinkError',
    'PathValidationError',
    'SourceNotFoundError',
    'DestinationError',
    'CleanupError',
    'DatabaseError',
    'DatabaseManager',
    'ImageBlender',
    'TransitionGenerator',
    'RifeDownloader',
    'PracticalRifeEnv',
    'SymlinkManager',
    'OPTICAL_FLOW_PRESETS',
]
