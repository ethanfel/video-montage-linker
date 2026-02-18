"""Main window UI for Video Montage Linker."""

import json
import os
import re
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, QEvent, QPoint, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QPainter, QFont, QFontMetrics
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
    QListWidgetItem,
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
    QSpinBox,
    QMenu,
    QProgressDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QCheckBox,
    QDoubleSpinBox,
)
from PyQt6.QtGui import QPixmap

from config import VIDEO_EXTENSIONS
from core import (
    BlendCurve,
    BlendMethod,
    FolderType,
    DirectInterpolationMethod,
    TransitionSettings,
    PerTransitionSettings,
    DirectTransitionSettings,
    VideoPreset,
    VIDEO_PRESETS,
    TransitionSpec,
    SymlinkError,
    CleanupError,
    DatabaseManager,
    TransitionGenerator,
    RifeDownloader,
    encode_image_sequence,
    encode_from_file_list,
    find_ffmpeg,
    PracticalRifeEnv,
    FilmEnv,
    SymlinkManager,
    OPTICAL_FLOW_PRESETS,
)
from .widgets import TrimSlider


class TimelineTreeWidget(QTreeWidget):
    """QTreeWidget with timeline markers drawn in the background."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.fps = 16
        self._text_color = QColor(100, 100, 100)

    def set_fps(self, fps: int) -> None:
        """Update FPS for timeline display."""
        self.fps = max(1, fps)
        self.viewport().update()

    def paintEvent(self, event) -> None:
        """Draw timeline markers in background, then call parent paint."""
        # Draw the timeline background on the viewport
        painter = QPainter(self.viewport())

        frame_count = self.topLevelItemCount()
        if frame_count > 0 and self.fps > 0:
            # Get row height from first visible item
            first_item = self.topLevelItem(0)
            if first_item:
                # Get column positions
                col0_width = self.columnWidth(0)
                col1_right = col0_width + self.columnWidth(1)
                viewport_width = self.viewport().width()

                # Font for time labels
                font = QFont("Monospace", 9)
                painter.setFont(font)
                metrics = QFontMetrics(font)

                # Draw for each row
                for i in range(frame_count):
                    item = self.topLevelItem(i)
                    if not item:
                        continue

                    item_rect = self.visualItemRect(item)
                    if item_rect.isNull() or item_rect.bottom() < 0 or item_rect.top() > self.viewport().height():
                        continue  # Not visible

                    y_center = item_rect.center().y()

                    # Calculate time for this frame
                    time_seconds = i / self.fps
                    is_major = (i % self.fps == 0)  # Every second

                    if is_major:
                        # Format time
                        minutes = int(time_seconds // 60)
                        seconds = int(time_seconds % 60)
                        if minutes > 0:
                            time_str = f"{minutes}:{seconds:02d}"
                        else:
                            time_str = f"{seconds}s"

                        text_width = metrics.horizontalAdvance(time_str)
                        painter.setPen(self._text_color)

                        # Draw time label on right of column 0
                        painter.drawText(col0_width - text_width - 6, y_center + metrics.ascent() // 2, time_str)

                        # Draw time label on right of column 1 (before the # column)
                        painter.drawText(col1_right - text_width - 6, y_center + metrics.ascent() // 2, time_str)

        painter.end()

        # Call parent to draw the actual tree content
        super().paintEvent(event)


class OverlapDialog(QDialog):
    """Dialog for setting per-transition overlap frames."""

    def __init__(
        self,
        parent: Optional[QWidget],
        folder_name: str,
        left_overlap: int = 16,
        right_overlap: int = 16
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Overlap Frames")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(f"Transition folder: {folder_name}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Form for overlap settings
        form_layout = QFormLayout()

        self.left_spin = QSpinBox()
        self.left_spin.setRange(1, 120)
        self.left_spin.setValue(left_overlap)
        self.left_spin.setToolTip("Overlap frames at the Main → Transition boundary")
        form_layout.addRow("Left boundary overlap:", self.left_spin)

        self.right_spin = QSpinBox()
        self.right_spin.setRange(1, 120)
        self.right_spin.setValue(right_overlap)
        self.right_spin.setToolTip("Overlap frames at the Transition → Main boundary")
        form_layout.addRow("Right boundary overlap:", self.right_spin)

        layout.addLayout(form_layout)

        # Explanation
        explain = QLabel(
            "Left: overlap frames at the Main → Trans boundary.\n"
            "Right: overlap frames at the Trans → Main boundary.\n"
            "Each side blends that many frames from both folders."
        )
        explain.setStyleSheet("color: gray; font-size: 10px;")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> tuple[int, int]:
        """Get the overlap values."""
        return self.left_spin.value(), self.right_spin.value()


class DirectTransitionDialog(QDialog):
    """Dialog for configuring direct frame interpolation between MAIN sequences."""

    def __init__(
        self,
        parent: Optional[QWidget],
        folder_name: str,
        frame_count: int = 16,
        method: DirectInterpolationMethod = DirectInterpolationMethod.FILM,
        enabled: bool = True
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Direct Interpolation Settings")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(f"Interpolate after: {folder_name}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Form for settings
        form_layout = QFormLayout()

        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItem("RIFE (Fast, small motion)", DirectInterpolationMethod.RIFE)
        self.method_combo.addItem("FILM (Slow, large motion)", DirectInterpolationMethod.FILM)
        if method == DirectInterpolationMethod.FILM:
            self.method_combo.setCurrentIndex(1)
        form_layout.addRow("Method:", self.method_combo)

        # Frame count
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 60)
        self.frame_spin.setValue(frame_count)
        self.frame_spin.setToolTip("Number of interpolated frames to generate")
        form_layout.addRow("Frames:", self.frame_spin)

        # Enable checkbox
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(enabled)
        form_layout.addRow("", self.enabled_check)

        layout.addLayout(form_layout)

        # Status label for setup state
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.status_label)

        # Setup button (for installing RIFE/FILM)
        self.setup_btn = QPushButton("Setup PyTorch Environment")
        self.setup_btn.setToolTip("Install PyTorch and required packages")
        self.setup_btn.clicked.connect(self._on_setup)
        layout.addWidget(self.setup_btn)

        # Explanation
        explain = QLabel(
            "RIFE: Fast AI interpolation, best for small motion and color shifts.\n"
            "FILM: Google Research model, better for large motion and scene gaps.\n\n"
            "Generated frames bridge the gap between the last frame of this\n"
            "sequence and the first frame of the next MAIN sequence."
        )
        explain.setStyleSheet("color: gray; font-size: 10px;")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        # Buttons
        button_layout = QHBoxLayout()

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Remove this direct transition")
        button_layout.addWidget(self.remove_btn)

        button_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)

        layout.addLayout(button_layout)

        self._removed = False
        self.remove_btn.clicked.connect(self._on_remove)
        self.method_combo.currentIndexChanged.connect(self._update_status)
        self._update_status()

    def _update_status(self) -> None:
        """Update the status label and setup button based on current method."""
        method = self.method_combo.currentData()

        rife_ready = PracticalRifeEnv.is_setup()
        film_ready = FilmEnv.is_setup() if rife_ready else False

        if method == DirectInterpolationMethod.RIFE:
            if rife_ready:
                self.status_label.setText("RIFE: Ready")
                self.status_label.setStyleSheet("color: green; font-size: 10px;")
                self.setup_btn.setVisible(False)
            else:
                self.status_label.setText("RIFE: Not installed (PyTorch required)")
                self.status_label.setStyleSheet("color: orange; font-size: 10px;")
                self.setup_btn.setVisible(True)
                self.setup_btn.setText("Setup PyTorch Environment")
        else:  # FILM
            if film_ready:
                self.status_label.setText("FILM: Ready")
                self.status_label.setStyleSheet("color: green; font-size: 10px;")
                self.setup_btn.setVisible(False)
            elif rife_ready:
                self.status_label.setText("FILM: Package not installed")
                self.status_label.setStyleSheet("color: orange; font-size: 10px;")
                self.setup_btn.setVisible(True)
                self.setup_btn.setText("Install FILM Package")
            else:
                self.status_label.setText("FILM: Not installed (PyTorch required first)")
                self.status_label.setStyleSheet("color: orange; font-size: 10px;")
                self.setup_btn.setVisible(True)
                self.setup_btn.setText("Setup PyTorch Environment")

    def _on_setup(self) -> None:
        """Handle setup button click."""
        method = self.method_combo.currentData()
        rife_ready = PracticalRifeEnv.is_setup()

        if not rife_ready:
            # Need to set up PyTorch venv first
            progress = QProgressDialog(
                "Setting up PyTorch environment...", "Cancel", 0, 100, self
            )
            progress.setWindowTitle("Setup")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            cancelled = [False]

            def progress_cb(msg, pct):
                progress.setLabelText(msg)
                progress.setValue(pct)

            def cancelled_check():
                QApplication.processEvents()
                return progress.wasCanceled()

            success = PracticalRifeEnv.setup_venv(progress_cb, cancelled_check)
            progress.close()

            if not success:
                if not cancelled_check():
                    QMessageBox.warning(
                        self, "Setup Failed",
                        "Failed to set up PyTorch environment."
                    )
                return

        # If FILM selected and we need to install FILM package
        if method == DirectInterpolationMethod.FILM and not FilmEnv.is_setup():
            progress = QProgressDialog(
                "Installing FILM package...", "Cancel", 0, 100, self
            )
            progress.setWindowTitle("Setup")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            def progress_cb(msg, pct):
                progress.setLabelText(msg)
                progress.setValue(pct)

            def cancelled_check():
                QApplication.processEvents()
                return progress.wasCanceled()

            success = FilmEnv.setup_film(progress_cb, cancelled_check)
            progress.close()

            if not success:
                if not cancelled_check():
                    QMessageBox.warning(
                        self, "Setup Failed",
                        "Failed to install FILM package."
                    )
                return

        self._update_status()

    def _on_remove(self) -> None:
        """Handle remove button click."""
        self._removed = True
        self.reject()

    def was_removed(self) -> bool:
        """Check if the user clicked Remove."""
        return self._removed

    def get_values(self) -> tuple[DirectInterpolationMethod, int, bool]:
        """Get the dialog values."""
        return (
            self.method_combo.currentData(),
            self.frame_spin.value(),
            self.enabled_check.isChecked()
        )


class SequenceLinkerUI(QWidget):
    """PyQt6 GUI for the Video Montage Linker."""

    def __init__(self) -> None:
        """Initialize the UI."""
        super().__init__()
        self.source_folders: list[Path] = []
        self._folder_ids: list[int] = []     # parallel to source_folders, same length
        self._next_folder_id: int = 1        # monotonic counter
        self.last_directory: Optional[str] = None
        self._last_resumed_dest: Optional[str] = None
        self._folder_trim_settings: dict[int, tuple[int, int]] = {}   # fid -> (start, end)
        self._folder_file_counts: dict[int, int] = {}                 # fid -> count
        self._folder_type_overrides: dict[int, FolderType] = {}       # fid -> type
        self._transition_settings = TransitionSettings()
        self._per_transition_settings: dict[int, PerTransitionSettings] = {}   # fid -> PTS
        self._direct_transitions: dict[int, DirectTransitionSettings] = {}     # fid -> DTS
        self._removed_files: dict[int, set[str]] = {}                         # fid -> set
        self._sequence_frame_count: int = 0  # Full output count including transition frames
        self._current_session_id: Optional[int] = None
        self.db = DatabaseManager()
        self.manager = SymlinkManager(self.db)
        self._setup_window()
        self._create_widgets()
        self._create_layout()
        self._connect_signals()
        self.setAcceptDrops(True)
        # Initialize sequence table FPS
        self.sequence_table.set_fps(self.fps_spin.value())

    def _allocate_folder_id(self) -> int:
        """Allocate a new unique folder entry ID."""
        fid = self._next_folder_id
        self._next_folder_id += 1
        return fid

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle('Video Montage Linker')
        self.setMinimumSize(1000, 700)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Source folders panel - side panel with single unified list
        self.source_panel = QWidget()
        self.source_panel.setMinimumWidth(150)

        # Single unified source list (odd=Main, even=Transition)
        self.source_list = QListWidget()
        self.source_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_list.setAlternatingRowColors(True)

        # Hidden lists for compatibility with old code
        self.main_list = QListWidget()
        self.main_list.setVisible(False)
        self.trans_list = QListWidget()
        self.trans_list.setVisible(False)

        # Folder buttons
        self.add_folder_btn = QPushButton("+ Add Folder")
        self.remove_source_btn = QPushButton("Remove")
        self.move_up_btn = QPushButton("▲")
        self.move_up_btn.setFixedWidth(40)
        self.move_down_btn = QPushButton("▼")
        self.move_down_btn.setFixedWidth(40)

        # Destination - now with two paths (editable combo boxes with history)
        self.dst_label = QLabel("Destination Folder:")
        self.dst_path = QComboBox()
        self.dst_path.setEditable(True)
        self.dst_path.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.dst_path.lineEdit().setPlaceholderText("Select destination folder for symlinks")
        self.dst_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dst_btn = QPushButton("Browse")

        self.trans_dst_label = QLabel("Transition Destination:")
        self.trans_dst_path = QComboBox()
        self.trans_dst_path.setEditable(True)
        self.trans_dst_path.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.trans_dst_path.lineEdit().setPlaceholderText("Select destination for transition output (optional)")
        self.trans_dst_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.trans_dst_btn = QPushButton("Browse")

        # Load path history
        self._load_path_history()

        # File list (Sequence Order tab)
        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["Sequence Name", "Original Filename", "Source Folder", "Frame"])
        self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setRootIsDecorated(False)
        self.file_list.header().setStretchLastSection(False)
        self.file_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.file_list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_list.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.file_list.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.file_list.header().resizeSection(0, 120)
        self.file_list.header().resizeSection(2, 150)
        self.file_list.header().resizeSection(3, 50)
        self.file_list.header().setMinimumSectionSize(40)
        self.file_list.setToolTip("Drag to reorder within folder, Del to remove")

        # Action buttons
        self.remove_files_btn = QPushButton("Remove Files")
        self.refresh_btn = QPushButton("Refresh Files")

        # Split export buttons
        self.export_btn = QPushButton("Export Sequence")
        self.export_btn.setStyleSheet(
            "background-color: #27ae60; color: white; "
            "height: 40px; font-weight: bold;"
        )
        self.export_trans_btn = QPushButton("Export with Transitions")
        self.export_trans_btn.setStyleSheet(
            "background-color: #3498db; color: white; "
            "height: 40px; font-weight: bold;"
        )

        self.encode_video_btn = QPushButton("Encode Video Only")
        self.encode_video_btn.setToolTip(
            "Encode an existing seq_* image sequence in the destination folder to video.\n"
            "No export is performed — frames must already exist."
        )

        self.save_session_btn = QPushButton("Save Session")
        self.save_session_btn.setToolTip(
            "Save the current session state (folders, files, trim, transitions, etc.)\n"
            "so you can resume exactly where you left off."
        )

        self.restore_session_btn = QPushButton("Restore Session")
        self.restore_session_btn.setToolTip(
            "Pick a previously saved session to restore."
        )

        self.copy_files_check = QCheckBox("Copy files (instead of symlinks)")
        self.copy_files_check.setToolTip(
            "Copy actual files instead of creating symlinks.\n"
            "Use this when the destination is accessed from Docker or a remote system."
        )

        # Export options group (collapsible via checkable)
        self.export_options_group = QGroupBox("Export Options")
        self.export_options_group.setCheckable(True)
        self.export_options_group.setChecked(False)

        # Range selection
        self.range_start_spin = QSpinBox()
        self.range_start_spin.setMinimum(0)
        self.range_start_spin.setMaximum(0)
        self.range_start_spin.setToolTip("First frame index to export")

        self.range_end_spin = QSpinBox()
        self.range_end_spin.setMinimum(0)
        self.range_end_spin.setMaximum(0)
        self.range_end_spin.setToolTip("Last frame index to export")

        self.range_reset_btn = QPushButton("Reset Range")
        self.range_reset_btn.setToolTip("Reset range to full sequence")

        # Video encoding
        self.video_export_check = QCheckBox("Encode video")
        self.video_export_check.setToolTip("Encode output frames to video after export")

        self.video_preset_combo = QComboBox()
        for key, vp in VIDEO_PRESETS.items():
            self.video_preset_combo.addItem(vp.label, key)
        self.video_preset_combo.setToolTip("Video encoding preset")

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
        self._blend_preview_cache: dict[str, QPixmap] = {}  # Cache for generated blend frames

        # Trim slider
        self.trim_slider = TrimSlider()
        self.trim_label = QLabel("Frames: All included")
        self.trim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Sequence table (2-column: Main Frame | Transition Frame) with timeline background
        self.sequence_table = TimelineTreeWidget()
        self.sequence_table.setHeaderLabels(["Main Frame", "Transition Frame", "#"])
        self.sequence_table.setColumnCount(3)
        self.sequence_table.setRootIsDecorated(False)
        self.sequence_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sequence_table.setAlternatingRowColors(True)
        self.sequence_table.header().setStretchLastSection(False)
        self.sequence_table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.sequence_table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.sequence_table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.sequence_table.header().resizeSection(0, 200)
        self.sequence_table.header().resizeSection(2, 50)
        self.sequence_table.header().setMinimumSectionSize(40)

        # Cross-dissolve transition settings group - horizontal layout
        self.transition_group = QGroupBox("Cross-Dissolve Transitions")
        self.transition_group.setCheckable(True)
        self.transition_group.setChecked(False)

        self.curve_combo = QComboBox()
        self.curve_combo.addItem("Linear", BlendCurve.LINEAR)
        self.curve_combo.addItem("Ease In", BlendCurve.EASE_IN)
        self.curve_combo.addItem("Ease Out", BlendCurve.EASE_OUT)
        self.curve_combo.addItem("Ease In/Out", BlendCurve.EASE_IN_OUT)
        self.curve_combo.setToolTip("Blend curve type for transitions")

        self.blend_format_combo = QComboBox()
        self.blend_format_combo.addItem("PNG", "png")
        self.blend_format_combo.addItem("JPEG", "jpeg")
        self.blend_format_combo.addItem("WebP", "webp")
        self.blend_format_combo.setToolTip("Output format for blended frames")

        # WebP method (replaces quality for WebP)
        self.webp_method_label = QLabel("Method:")
        self.webp_method_spin = QSpinBox()
        self.webp_method_spin.setRange(0, 6)
        self.webp_method_spin.setValue(4)
        self.webp_method_spin.setToolTip("WebP compression method (0=fast/larger, 6=slow/smaller)")
        self.webp_method_label.setVisible(False)
        self.webp_method_spin.setVisible(False)

        # JPEG quality
        self.quality_label = QLabel("Quality:")
        self.blend_quality_spin = QSpinBox()
        self.blend_quality_spin.setRange(1, 100)
        self.blend_quality_spin.setValue(95)
        self.blend_quality_spin.setToolTip("Quality for JPEG output (1-100)")
        self.quality_label.setVisible(False)
        self.blend_quality_spin.setVisible(False)

        # Blend method combo
        self.blend_method_combo = QComboBox()
        self.blend_method_combo.addItem("Cross-Dissolve", BlendMethod.ALPHA)
        self.blend_method_combo.addItem("Optical Flow", BlendMethod.OPTICAL_FLOW)
        self.blend_method_combo.addItem("RIFE (ncnn)", BlendMethod.RIFE)
        self.blend_method_combo.addItem("RIFE (Practical)", BlendMethod.RIFE_PRACTICAL)
        self.blend_method_combo.setToolTip(
            "Blending method:\n"
            "- Cross-Dissolve: Simple alpha blend (fast, may ghost)\n"
            "- Optical Flow: Motion-compensated blend (slower, less ghosting)\n"
            "- RIFE (ncnn): AI frame interpolation (fast, Vulkan GPU, models up to v4.6)\n"
            "- RIFE (Practical): AI frame interpolation (PyTorch, latest models v4.25/v4.26)"
        )

        # RIFE binary path
        self.rife_path_label = QLabel("RIFE:")
        self.rife_path_input = QLineEdit(placeholderText="Path to rife-ncnn-vulkan (optional, auto-downloads)")
        self.rife_path_input.setToolTip("Path to rife-ncnn-vulkan binary. Leave empty to auto-download.")
        self.rife_path_btn = QPushButton("...")
        self.rife_path_btn.setFixedWidth(30)
        self.rife_download_btn = QPushButton("Download")
        self.rife_download_btn.setToolTip("Download latest rife-ncnn-vulkan from GitHub")
        self.rife_path_label.setVisible(False)
        self.rife_path_input.setVisible(False)
        self.rife_path_btn.setVisible(False)
        self.rife_download_btn.setVisible(False)

        # RIFE model selection
        self.rife_model_label = QLabel("Model:")
        self.rife_model_combo = QComboBox()
        self.rife_model_combo.addItem("v4.6 (Best)", "rife-v4.6")
        self.rife_model_combo.addItem("v4", "rife-v4")
        self.rife_model_combo.addItem("v3.1", "rife-v3.1")
        self.rife_model_combo.addItem("v2.4", "rife-v2.4")
        self.rife_model_combo.addItem("Anime", "rife-anime")
        self.rife_model_combo.addItem("UHD", "rife-UHD")
        self.rife_model_combo.addItem("HD", "rife-HD")
        self.rife_model_combo.setToolTip("RIFE model version:\n- v4.6: Latest, best quality\n- Anime: Optimized for animation\n- UHD/HD: For high resolution content")
        self.rife_model_label.setVisible(False)
        self.rife_model_combo.setVisible(False)

        # RIFE UHD mode
        self.rife_uhd_check = QCheckBox("UHD")
        self.rife_uhd_check.setToolTip("Enable UHD mode for high resolution images (4K+)")
        self.rife_uhd_check.setVisible(False)

        # RIFE TTA mode
        self.rife_tta_check = QCheckBox("TTA")
        self.rife_tta_check.setToolTip("Enable TTA (Test-Time Augmentation) for better quality (slower)")
        self.rife_tta_check.setVisible(False)

        # Practical-RIFE settings
        self.practical_model_label = QLabel("Model:")
        self.practical_model_combo = QComboBox()
        self.practical_model_combo.addItem("v4.26 (Latest)", "v4.26")
        self.practical_model_combo.addItem("v4.25 (Recommended)", "v4.25")
        self.practical_model_combo.addItem("v4.22", "v4.22")
        self.practical_model_combo.addItem("v4.20", "v4.20")
        self.practical_model_combo.addItem("v4.18", "v4.18")
        self.practical_model_combo.addItem("v4.15", "v4.15")
        self.practical_model_combo.setCurrentIndex(1)  # Default to v4.25
        self.practical_model_combo.setToolTip(
            "Practical-RIFE model version:\n"
            "- v4.26: Latest version\n"
            "- v4.25: Recommended, good balance of quality and speed"
        )
        self.practical_model_label.setVisible(False)
        self.practical_model_combo.setVisible(False)

        self.practical_ensemble_check = QCheckBox("Ensemble")
        self.practical_ensemble_check.setToolTip("Enable ensemble mode for better quality (slower)")
        self.practical_ensemble_check.setVisible(False)

        self.practical_setup_btn = QPushButton("Setup PyTorch")
        self.practical_setup_btn.setToolTip("Create local venv and install PyTorch (~2GB download)")
        self.practical_setup_btn.setVisible(False)

        self.practical_status_label = QLabel("")
        self.practical_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.practical_status_label.setVisible(False)

        # Optical flow settings
        self.of_preset_label = QLabel("OF Preset:")
        self.of_preset_combo = QComboBox()
        self.of_preset_combo.addItem("Fast", "fast")
        self.of_preset_combo.addItem("Balanced", "balanced")
        self.of_preset_combo.addItem("Quality", "quality")
        self.of_preset_combo.addItem("Max", "max")
        self.of_preset_combo.addItem("Custom", "custom")
        self.of_preset_combo.setCurrentIndex(3)  # Default to Max
        self.of_preset_combo.setToolTip(
            "Optical flow quality preset:\n"
            "- Fast: Quick processing, lower quality\n"
            "- Balanced: Good balance of speed and quality\n"
            "- Quality: Higher quality, slower\n"
            "- Max: Best quality, slowest"
        )
        self.of_preset_label.setVisible(False)
        self.of_preset_combo.setVisible(False)

        self.of_levels_label = QLabel("Levels:")
        self.of_levels_spin = QSpinBox()
        self.of_levels_spin.setRange(1, 7)
        self.of_levels_spin.setValue(3)
        self.of_levels_spin.setToolTip("Pyramid levels (1-7): Higher = handles larger motion")
        self.of_levels_label.setVisible(False)
        self.of_levels_spin.setVisible(False)

        self.of_winsize_label = QLabel("WinSize:")
        self.of_winsize_spin = QSpinBox()
        self.of_winsize_spin.setRange(5, 51)
        self.of_winsize_spin.setSingleStep(2)
        self.of_winsize_spin.setValue(15)
        self.of_winsize_spin.setToolTip("Window size (5-51, odd): Larger = smoother but slower")
        self.of_winsize_label.setVisible(False)
        self.of_winsize_spin.setVisible(False)

        self.of_iterations_label = QLabel("Iters:")
        self.of_iterations_spin = QSpinBox()
        self.of_iterations_spin.setRange(1, 10)
        self.of_iterations_spin.setValue(3)
        self.of_iterations_spin.setToolTip("Iterations (1-10): More = better convergence")
        self.of_iterations_label.setVisible(False)
        self.of_iterations_spin.setVisible(False)

        self.of_poly_n_label = QLabel("PolyN:")
        self.of_poly_n_combo = QComboBox()
        self.of_poly_n_combo.addItem("5", 5)
        self.of_poly_n_combo.addItem("7", 7)
        self.of_poly_n_combo.setToolTip("Polynomial neighborhood (5 or 7): 7 = more robust")
        self.of_poly_n_label.setVisible(False)
        self.of_poly_n_combo.setVisible(False)

        self.of_poly_sigma_label = QLabel("Sigma:")
        self.of_poly_sigma_spin = QDoubleSpinBox()
        self.of_poly_sigma_spin.setRange(0.5, 2.0)
        self.of_poly_sigma_spin.setSingleStep(0.1)
        self.of_poly_sigma_spin.setValue(1.2)
        self.of_poly_sigma_spin.setDecimals(1)
        self.of_poly_sigma_spin.setToolTip("Poly sigma (0.5-2.0): Gaussian smoothing")
        self.of_poly_sigma_label.setVisible(False)
        self.of_poly_sigma_spin.setVisible(False)

        # FPS setting for sequence playback and timeline
        self.fps_label = QLabel("FPS:")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(16)
        self.fps_spin.setToolTip("Frames per second for sequence preview and timeline")

        # Timeline duration label
        self.timeline_label = QLabel("Duration: 00:00.000 (0 frames)")
        self.timeline_label.setStyleSheet("font-family: monospace;")

        # Sequence playback button and timer
        self.seq_play_btn = QPushButton("▶ Play")
        self.seq_play_btn.setToolTip("Play image sequence at configured FPS")
        self.sequence_timer = QTimer(self)
        self.sequence_playing = False

    def _create_layout(self) -> None:
        """Arrange widgets in layouts."""
        # === LEFT SIDE PANEL: Source Folders ===
        source_panel_layout = QVBoxLayout(self.source_panel)
        source_panel_layout.setContentsMargins(0, 0, 0, 0)

        # Title with legend
        source_title = QLabel("Source Folders")
        source_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        source_panel_layout.addWidget(source_title)

        legend = QLabel("Odd = Main [M], Even = Transition [T]")
        legend.setStyleSheet("color: gray; font-size: 10px;")
        source_panel_layout.addWidget(legend)

        # Single unified list
        source_panel_layout.addWidget(self.source_list, 1)
        source_panel_layout.addWidget(self.add_folder_btn)

        # Folder control buttons
        folder_btn_layout = QHBoxLayout()
        folder_btn_layout.addWidget(self.move_up_btn)
        folder_btn_layout.addWidget(self.move_down_btn)
        folder_btn_layout.addStretch()
        folder_btn_layout.addWidget(self.remove_source_btn)
        source_panel_layout.addLayout(folder_btn_layout)

        # === RIGHT SIDE: Main Content ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Destination layout
        dst_layout = QHBoxLayout()
        dst_layout.addWidget(self.dst_label)
        dst_layout.addWidget(self.dst_path, 1)
        dst_layout.addWidget(self.dst_btn)

        trans_dst_layout = QHBoxLayout()
        trans_dst_layout.addWidget(self.trans_dst_label)
        trans_dst_layout.addWidget(self.trans_dst_path, 1)
        trans_dst_layout.addWidget(self.trans_dst_btn)

        # Transition settings group layout - horizontal
        transition_layout = QHBoxLayout()
        transition_layout.addWidget(QLabel("Method:"))
        transition_layout.addWidget(self.blend_method_combo)
        transition_layout.addWidget(QLabel("Curve:"))
        transition_layout.addWidget(self.curve_combo)
        transition_layout.addWidget(QLabel("Format:"))
        transition_layout.addWidget(self.blend_format_combo)
        transition_layout.addWidget(self.webp_method_label)
        transition_layout.addWidget(self.webp_method_spin)
        transition_layout.addWidget(self.quality_label)
        transition_layout.addWidget(self.blend_quality_spin)
        transition_layout.addWidget(self.rife_path_label)
        transition_layout.addWidget(self.rife_path_input)
        transition_layout.addWidget(self.rife_path_btn)
        transition_layout.addWidget(self.rife_download_btn)
        transition_layout.addWidget(self.rife_model_label)
        transition_layout.addWidget(self.rife_model_combo)
        transition_layout.addWidget(self.rife_uhd_check)
        transition_layout.addWidget(self.rife_tta_check)
        transition_layout.addWidget(self.practical_model_label)
        transition_layout.addWidget(self.practical_model_combo)
        transition_layout.addWidget(self.practical_ensemble_check)
        transition_layout.addWidget(self.practical_setup_btn)
        transition_layout.addWidget(self.practical_status_label)
        transition_layout.addWidget(self.of_preset_label)
        transition_layout.addWidget(self.of_preset_combo)
        transition_layout.addWidget(self.of_levels_label)
        transition_layout.addWidget(self.of_levels_spin)
        transition_layout.addWidget(self.of_winsize_label)
        transition_layout.addWidget(self.of_winsize_spin)
        transition_layout.addWidget(self.of_iterations_label)
        transition_layout.addWidget(self.of_iterations_spin)
        transition_layout.addWidget(self.of_poly_n_label)
        transition_layout.addWidget(self.of_poly_n_combo)
        transition_layout.addWidget(self.of_poly_sigma_label)
        transition_layout.addWidget(self.of_poly_sigma_spin)
        transition_layout.addWidget(self.fps_label)
        transition_layout.addWidget(self.fps_spin)
        transition_layout.addWidget(self.timeline_label)
        transition_layout.addWidget(self.seq_play_btn)
        transition_layout.addStretch()
        self.transition_group.setLayout(transition_layout)

        # File list action buttons (refresh far from remove to avoid misclicks)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.remove_files_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.refresh_btn)

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

        # Image preview area
        image_top_bar = QHBoxLayout()
        image_top_bar.addWidget(self.image_name_label, 1)
        image_top_bar.addWidget(self.zoom_out_btn)
        image_top_bar.addWidget(self.zoom_label)
        image_top_bar.addWidget(self.zoom_in_btn)
        image_top_bar.addWidget(self.zoom_reset_btn)
        image_tab_layout.addLayout(image_top_bar)
        image_tab_layout.addWidget(self.image_scroll, 1)

        # Image navigation controls
        image_controls = QHBoxLayout()
        image_controls.addWidget(self.prev_image_btn)
        image_controls.addWidget(self.image_slider, 1)
        image_controls.addWidget(self.next_image_btn)
        image_controls.addWidget(self.image_index_label)
        image_tab_layout.addLayout(image_controls)
        image_tab_layout.addWidget(self.trim_label)
        image_tab_layout.addWidget(self.trim_slider)

        # Add tabs to preview widget
        self.preview_tabs.addTab(self.video_tab, "Video Preview")
        self.preview_tabs.addTab(self.image_tab, "Image Sequence")

        # File list panel with tabs for Sequence Order and With Transitions
        file_list_panel = QWidget()
        file_list_layout = QVBoxLayout(file_list_panel)
        file_list_layout.setContentsMargins(0, 0, 0, 0)

        # Tabs for sequence views
        self.sequence_tabs = QTabWidget()

        # Tab 1: Sequence Order (editable file list)
        sequence_order_tab = QWidget()
        sequence_order_layout = QVBoxLayout(sequence_order_tab)
        sequence_order_layout.setContentsMargins(0, 0, 0, 0)
        sequence_order_layout.addWidget(self.file_list)
        self.sequence_tabs.addTab(sequence_order_tab, "Sequence Order")

        # Tab 2: With Transitions (2-column view with timeline rulers)
        trans_sequence_tab = QWidget()
        trans_sequence_layout = QVBoxLayout(trans_sequence_tab)
        trans_sequence_layout.setContentsMargins(0, 0, 0, 0)

        trans_sequence_layout.addWidget(self.sequence_table)

        self.sequence_tabs.addTab(trans_sequence_tab, "With Transitions")

        file_list_layout.addWidget(self.sequence_tabs)
        file_list_layout.addLayout(btn_layout)

        # Splitter for file list and preview tabs
        file_list_panel.setMinimumWidth(200)
        self.preview_tabs.setMinimumWidth(200)
        file_list_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.preview_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.addWidget(file_list_panel)
        self.content_splitter.addWidget(self.preview_tabs)
        self.content_splitter.setHandleWidth(5)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setSizes([350, 450])

        # Export buttons layout
        export_layout = QHBoxLayout()
        export_layout.addWidget(self.copy_files_check)
        export_layout.addWidget(self.save_session_btn)
        export_layout.addWidget(self.restore_session_btn)
        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.export_trans_btn)
        export_layout.addWidget(self.encode_video_btn)

        # Export options group layout
        export_opts_layout = QVBoxLayout()
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range:"))
        range_layout.addWidget(self.range_start_spin)
        range_layout.addWidget(QLabel("—"))
        range_layout.addWidget(self.range_end_spin)
        range_layout.addWidget(self.range_reset_btn)
        range_layout.addStretch()
        export_opts_layout.addLayout(range_layout)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_export_check)
        video_layout.addWidget(self.video_preset_combo, 1)
        export_opts_layout.addLayout(video_layout)
        self.export_options_group.setLayout(export_opts_layout)

        # Assemble right panel
        right_layout.addLayout(dst_layout)
        right_layout.addLayout(trans_dst_layout)
        right_layout.addWidget(self.transition_group)
        right_layout.addWidget(self.content_splitter, 1)
        right_layout.addLayout(export_layout)
        right_layout.addWidget(self.export_options_group)

        # === MAIN SPLITTER: Source Panel | Main Content ===
        right_panel.setMinimumWidth(400)
        self.source_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.source_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([250, 750])

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.main_splitter, 1)
        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Folder buttons
        self.add_folder_btn.clicked.connect(self._add_source_folder)
        self.remove_source_btn.clicked.connect(self._remove_source_folder)
        self.move_up_btn.clicked.connect(self._move_folder_up)
        self.move_down_btn.clicked.connect(self._move_folder_down)
        self.dst_btn.clicked.connect(self._browse_destination)
        self.dst_path.lineEdit().editingFinished.connect(self._on_destination_changed)
        self.dst_path.currentIndexChanged.connect(self._on_destination_changed)
        self.trans_dst_btn.clicked.connect(self._browse_trans_destination)
        self.remove_files_btn.clicked.connect(self._remove_selected_files)
        self.refresh_btn.clicked.connect(self._refresh_files)

        # Export buttons
        self.save_session_btn.clicked.connect(self._save_session)
        self.restore_session_btn.clicked.connect(self._pick_and_restore_session)
        self.export_btn.clicked.connect(self._export_sequence)
        self.export_trans_btn.clicked.connect(self._export_with_transitions)
        self.encode_video_btn.clicked.connect(self._encode_video_only)

        # Export options signals
        self.video_export_check.toggled.connect(self.video_preset_combo.setEnabled)
        self.range_reset_btn.clicked.connect(self._reset_export_range)
        self.range_start_spin.valueChanged.connect(
            lambda v: self.range_end_spin.setMinimum(v)
        )
        self.range_end_spin.valueChanged.connect(
            lambda v: self.range_start_spin.setMaximum(v)
        )

        # Connect reorder signals
        self.file_list.model().rowsMoved.connect(self._recalculate_sequence_names)

        # Context menu for folder type and overlap override
        self.source_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.source_list.customContextMenuRequested.connect(self._show_folder_context_menu)

        # Context menu for file list (split sequence, remove frame)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)

        # Update folder indicators when transition setting changes
        self.transition_group.toggled.connect(self._update_folder_type_indicators)

        # Connect folder selection to update video list
        self.source_list.currentItemChanged.connect(self._on_folder_selected)
        self.source_list.itemClicked.connect(self._on_source_item_clicked)

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
        self.trim_slider.trimDragFinished.connect(self._on_trim_drag_finished)

        # Format combo change - show/hide quality/method widgets
        self.blend_format_combo.currentIndexChanged.connect(self._on_format_changed)

        # Blend method combo change - show/hide RIFE path
        self.blend_method_combo.currentIndexChanged.connect(self._on_blend_method_changed)
        self.curve_combo.currentIndexChanged.connect(self._clear_blend_cache)
        self.rife_model_combo.currentIndexChanged.connect(self._clear_blend_cache)
        self.rife_uhd_check.stateChanged.connect(self._clear_blend_cache)
        self.rife_tta_check.stateChanged.connect(self._clear_blend_cache)
        self.rife_path_btn.clicked.connect(self._browse_rife_binary)
        self.rife_download_btn.clicked.connect(self._download_rife_binary)

        # Practical-RIFE signals
        self.practical_model_combo.currentIndexChanged.connect(self._clear_blend_cache)
        self.practical_ensemble_check.stateChanged.connect(self._clear_blend_cache)
        self.practical_setup_btn.clicked.connect(self._setup_practical_rife)

        # Optical flow signals
        self.of_preset_combo.currentIndexChanged.connect(self._on_of_preset_changed)
        self.of_levels_spin.valueChanged.connect(self._on_of_param_changed)
        self.of_winsize_spin.valueChanged.connect(self._on_of_param_changed)
        self.of_iterations_spin.valueChanged.connect(self._on_of_param_changed)
        self.of_poly_n_combo.currentIndexChanged.connect(self._on_of_param_changed)
        self.of_poly_sigma_spin.valueChanged.connect(self._on_of_param_changed)

        # Sequence table selection - show image
        self.sequence_table.currentItemChanged.connect(self._on_sequence_table_selected)
        # Also handle clicks on non-selectable items (direct interpolation rows)
        self.sequence_table.itemClicked.connect(self._on_sequence_table_clicked)

        # Update sequence table when transitions setting changes
        self.transition_group.toggled.connect(self._update_sequence_table)

        # Update sequence table when switching to "With Transitions" tab
        self.sequence_tabs.currentChanged.connect(self._on_sequence_tab_changed)

        # FPS and sequence playback signals
        self.fps_spin.valueChanged.connect(self._update_timeline_display)
        self.seq_play_btn.clicked.connect(self._toggle_sequence_play)
        self.sequence_timer.timeout.connect(self._advance_sequence_frame)

        # Update sequence table FPS when spinner changes
        self.fps_spin.valueChanged.connect(self.sequence_table.set_fps)

    def _on_format_changed(self, index: int) -> None:
        """Handle format combo change to show/hide quality/method widgets."""
        fmt = self.blend_format_combo.currentData()
        if fmt == 'webp':
            self.webp_method_label.setVisible(True)
            self.webp_method_spin.setVisible(True)
            self.quality_label.setVisible(False)
            self.blend_quality_spin.setVisible(False)
        elif fmt == 'jpeg':
            self.webp_method_label.setVisible(False)
            self.webp_method_spin.setVisible(False)
            self.quality_label.setVisible(True)
            self.blend_quality_spin.setVisible(True)
        else:  # png
            self.webp_method_label.setVisible(False)
            self.webp_method_spin.setVisible(False)
            self.quality_label.setVisible(False)
            self.blend_quality_spin.setVisible(False)

    def _on_blend_method_changed(self, index: int) -> None:
        """Handle blend method combo change to show/hide RIFE path widgets."""
        method = self.blend_method_combo.currentData()
        is_rife_ncnn = (method == BlendMethod.RIFE)
        is_rife_practical = (method == BlendMethod.RIFE_PRACTICAL)
        is_optical_flow = (method == BlendMethod.OPTICAL_FLOW)

        # RIFE ncnn settings
        self.rife_path_label.setVisible(is_rife_ncnn)
        self.rife_path_input.setVisible(is_rife_ncnn)
        self.rife_path_btn.setVisible(is_rife_ncnn)
        self.rife_download_btn.setVisible(is_rife_ncnn)
        self.rife_model_label.setVisible(is_rife_ncnn)
        self.rife_model_combo.setVisible(is_rife_ncnn)
        self.rife_uhd_check.setVisible(is_rife_ncnn)
        self.rife_tta_check.setVisible(is_rife_ncnn)

        # Practical-RIFE settings
        self.practical_model_label.setVisible(is_rife_practical)
        self.practical_model_combo.setVisible(is_rife_practical)
        self.practical_ensemble_check.setVisible(is_rife_practical)
        self.practical_setup_btn.setVisible(is_rife_practical)
        self.practical_status_label.setVisible(is_rife_practical)

        # Optical flow settings
        self.of_preset_label.setVisible(is_optical_flow)
        self.of_preset_combo.setVisible(is_optical_flow)
        self.of_levels_label.setVisible(is_optical_flow)
        self.of_levels_spin.setVisible(is_optical_flow)
        self.of_winsize_label.setVisible(is_optical_flow)
        self.of_winsize_spin.setVisible(is_optical_flow)
        self.of_iterations_label.setVisible(is_optical_flow)
        self.of_iterations_spin.setVisible(is_optical_flow)
        self.of_poly_n_label.setVisible(is_optical_flow)
        self.of_poly_n_combo.setVisible(is_optical_flow)
        self.of_poly_sigma_label.setVisible(is_optical_flow)
        self.of_poly_sigma_spin.setVisible(is_optical_flow)

        if is_rife_ncnn:
            self._update_rife_download_button()

        if is_rife_practical:
            self._update_practical_rife_status()

        # Clear blend preview cache when method changes
        self._blend_preview_cache.clear()

    def _clear_blend_cache(self) -> None:
        """Clear the blend preview cache."""
        self._blend_preview_cache.clear()

    def _on_of_preset_changed(self, index: int) -> None:
        """Handle optical flow preset change."""
        preset = self.of_preset_combo.currentData()
        if preset == 'custom':
            # User selected custom - don't change sliders
            self._clear_blend_cache()
            return

        # Apply preset values to sliders
        if preset in OPTICAL_FLOW_PRESETS:
            values = OPTICAL_FLOW_PRESETS[preset]
            # Block signals while updating to avoid triggering _on_of_param_changed
            self.of_levels_spin.blockSignals(True)
            self.of_winsize_spin.blockSignals(True)
            self.of_iterations_spin.blockSignals(True)
            self.of_poly_n_combo.blockSignals(True)
            self.of_poly_sigma_spin.blockSignals(True)

            self.of_levels_spin.setValue(values['levels'])
            self.of_winsize_spin.setValue(values['winsize'])
            self.of_iterations_spin.setValue(values['iterations'])
            # Set poly_n combo
            poly_n_idx = 0 if values['poly_n'] == 5 else 1
            self.of_poly_n_combo.setCurrentIndex(poly_n_idx)
            self.of_poly_sigma_spin.setValue(values['poly_sigma'])

            self.of_levels_spin.blockSignals(False)
            self.of_winsize_spin.blockSignals(False)
            self.of_iterations_spin.blockSignals(False)
            self.of_poly_n_combo.blockSignals(False)
            self.of_poly_sigma_spin.blockSignals(False)

        self._clear_blend_cache()

    def _on_of_param_changed(self) -> None:
        """Handle optical flow parameter change - set preset to Custom."""
        # Check if current values match any preset
        current_values = {
            'levels': self.of_levels_spin.value(),
            'winsize': self.of_winsize_spin.value(),
            'iterations': self.of_iterations_spin.value(),
            'poly_n': self.of_poly_n_combo.currentData(),
            'poly_sigma': self.of_poly_sigma_spin.value(),
        }

        # Find matching preset
        matching_preset = None
        for preset_name, preset_values in OPTICAL_FLOW_PRESETS.items():
            if (preset_values['levels'] == current_values['levels'] and
                preset_values['winsize'] == current_values['winsize'] and
                preset_values['iterations'] == current_values['iterations'] and
                preset_values['poly_n'] == current_values['poly_n'] and
                abs(preset_values['poly_sigma'] - current_values['poly_sigma']) < 0.05):
                matching_preset = preset_name
                break

        # Update preset combo without triggering _on_of_preset_changed
        self.of_preset_combo.blockSignals(True)
        if matching_preset:
            # Find index of matching preset
            for i in range(self.of_preset_combo.count()):
                if self.of_preset_combo.itemData(i) == matching_preset:
                    self.of_preset_combo.setCurrentIndex(i)
                    break
        else:
            # Set to Custom
            for i in range(self.of_preset_combo.count()):
                if self.of_preset_combo.itemData(i) == 'custom':
                    self.of_preset_combo.setCurrentIndex(i)
                    break
        self.of_preset_combo.blockSignals(False)

        self._clear_blend_cache()

    def _browse_rife_binary(self) -> None:
        """Browse for RIFE binary."""
        start_dir = self.last_directory or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select rife-ncnn-vulkan Binary", start_dir,
            "Executable Files (*)"
        )
        if path:
            self.rife_path_input.setText(path)
            self.last_directory = str(Path(path).parent)
            self._update_rife_download_button()

    def _update_rife_download_button(self) -> None:
        """Update the RIFE download button text based on availability."""
        import shutil

        # Check if user specified a path
        user_path = self.rife_path_input.text().strip()
        if user_path and Path(user_path).exists():
            self.rife_download_btn.setText("Ready")
            self.rife_download_btn.setEnabled(False)
            self.rife_download_btn.setToolTip("RIFE binary found at specified path")
            return

        # Check system PATH
        if shutil.which('rife-ncnn-vulkan'):
            self.rife_download_btn.setText("Ready")
            self.rife_download_btn.setEnabled(False)
            self.rife_download_btn.setToolTip("RIFE binary found in system PATH")
            return

        # Check cached
        cached = RifeDownloader.get_cached_binary()
        if cached:
            self.rife_download_btn.setText("Ready")
            self.rife_download_btn.setEnabled(False)
            self.rife_download_btn.setToolTip(f"RIFE binary cached at: {cached}")
            return

        # Not available - show download button
        self.rife_download_btn.setText("Download")
        self.rife_download_btn.setEnabled(True)
        self.rife_download_btn.setToolTip("Download latest rife-ncnn-vulkan from GitHub")

    def _download_rife_binary(self) -> None:
        """Download the RIFE binary with progress dialog."""
        # Check platform support
        platform_id = RifeDownloader.get_platform_identifier()
        if not platform_id:
            QMessageBox.warning(
                self, "Unsupported Platform",
                "RIFE auto-download is not supported on this platform.\n"
                "Please download manually from:\n"
                "https://github.com/nihui/rife-ncnn-vulkan/releases"
            )
            return

        # Get release info
        self.rife_download_btn.setText("Checking...")
        self.rife_download_btn.setEnabled(False)
        QApplication.processEvents()

        release_info = RifeDownloader.get_latest_release_info()
        if not release_info:
            QMessageBox.warning(
                self, "Download Failed",
                "Failed to fetch release info from GitHub.\n"
                "Check your internet connection or download manually."
            )
            self._update_rife_download_button()
            return

        asset_url = RifeDownloader.find_asset_url(release_info, platform_id)
        if not asset_url:
            QMessageBox.warning(
                self, "Download Failed",
                f"No binary found for platform: {platform_id}\n"
                "Please download manually from:\n"
                "https://github.com/nihui/rife-ncnn-vulkan/releases"
            )
            self._update_rife_download_button()
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Downloading rife-ncnn-vulkan...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Downloading RIFE")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # Download with progress callback
        def progress_callback(downloaded, total):
            if progress.wasCanceled():
                return
            if total > 0:
                percent = int(downloaded * 100 / total)
                progress.setValue(percent)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                progress.setLabelText(
                    f"Downloading rife-ncnn-vulkan...\n"
                    f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB"
                )
            QApplication.processEvents()

        def cancelled_check():
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            binary_path = RifeDownloader.download_and_extract(
                asset_url, progress_callback, cancelled_check
            )
            progress.close()

            if progress.wasCanceled():
                self._update_rife_download_button()
                return

            if binary_path:
                QMessageBox.information(
                    self, "Download Complete",
                    f"RIFE downloaded successfully!\n\n"
                    f"Location: {binary_path}"
                )
                self._update_rife_download_button()
            else:
                QMessageBox.warning(
                    self, "Download Failed",
                    "Failed to download or extract RIFE binary.\n"
                    "Please download manually."
                )
                self._update_rife_download_button()

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, "Download Error",
                f"Error downloading RIFE: {e}"
            )
            self._update_rife_download_button()

    def _update_practical_rife_status(self) -> None:
        """Update the Practical-RIFE status label and setup button."""
        if PracticalRifeEnv.is_setup():
            torch_version = PracticalRifeEnv.get_torch_version()
            if torch_version:
                self.practical_status_label.setText(f"Ready (PyTorch {torch_version})")
                self.practical_status_label.setStyleSheet("color: green; font-size: 10px;")
            else:
                self.practical_status_label.setText("Ready")
                self.practical_status_label.setStyleSheet("color: green; font-size: 10px;")
            self.practical_setup_btn.setText("Reinstall")
            self.practical_setup_btn.setToolTip("Reinstall PyTorch environment")
            self.practical_model_combo.setEnabled(True)
            self.practical_ensemble_check.setEnabled(True)
        else:
            self.practical_status_label.setText("Not configured")
            self.practical_status_label.setStyleSheet("color: orange; font-size: 10px;")
            self.practical_setup_btn.setText("Setup PyTorch")
            self.practical_setup_btn.setToolTip("Create local venv and install PyTorch (~2GB download)")
            self.practical_model_combo.setEnabled(False)
            self.practical_ensemble_check.setEnabled(False)

    def _setup_practical_rife(self) -> None:
        """Setup Practical-RIFE environment with progress dialog."""
        # Confirm if already setup
        if PracticalRifeEnv.is_setup():
            reply = QMessageBox.question(
                self, "Reinstall PyTorch?",
                "PyTorch environment is already set up.\n"
                "Do you want to reinstall it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Create progress dialog
        progress = QProgressDialog(
            "Setting up PyTorch environment...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Setup Practical-RIFE")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # Progress callback
        def progress_callback(message, percent):
            if not progress.wasCanceled():
                progress.setLabelText(message)
                progress.setValue(percent)
                QApplication.processEvents()

        def cancelled_check():
            QApplication.processEvents()
            return progress.wasCanceled()

        try:
            success = PracticalRifeEnv.setup_venv(progress_callback, cancelled_check)
            progress.close()

            if progress.wasCanceled():
                self._update_practical_rife_status()
                return

            if success:
                QMessageBox.information(
                    self, "Setup Complete",
                    "PyTorch environment set up successfully!\n\n"
                    f"Location: {PracticalRifeEnv.VENV_DIR}\n\n"
                    "You can now use RIFE (Practical) for frame interpolation."
                )
            else:
                QMessageBox.warning(
                    self, "Setup Failed",
                    "Failed to set up PyTorch environment.\n"
                    "Check your internet connection and try again."
                )

            self._update_practical_rife_status()

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self, "Setup Error",
                f"Error setting up PyTorch: {e}"
            )
            self._update_practical_rife_status()

    def _on_sequence_tab_changed(self, index: int) -> None:
        """Handle sequence tab change to update the With Transitions view."""
        if index == 1:  # "With Transitions" tab
            self._update_sequence_table()

    def _update_sequence_table(self, _=None) -> None:
        """Update the 2-column sequence table showing Main/Transition frame pairing."""
        self.sequence_table.setUpdatesEnabled(False)
        self.sequence_table.clear()

        if not self.source_folders:
            self._sequence_frame_count = 0
            self.sequence_table.setUpdatesEnabled(True)
            self._update_timeline_display()
            self._update_export_range_max()
            return

        files = self._get_files_in_order()
        if not files:
            self._sequence_frame_count = 0
            self.sequence_table.setUpdatesEnabled(True)
            self._update_timeline_display()
            self._update_export_range_max()
            return

        # Build files_by_idx: position index → file list (supports duplicate folders)
        fid_to_pos = {fid: i for i, fid in enumerate(self._folder_ids)}
        files_by_idx: dict[int, list[str]] = {}
        for source_dir, filename, folder_idx, file_idx, fid in files:
            pi = fid_to_pos.get(fid, 0)
            if pi not in files_by_idx:
                files_by_idx[pi] = []
            files_by_idx[pi].append(filename)

        # Also include TRANSITION folder files (not in file_list but needed for blending)
        from config import SUPPORTED_EXTENSIONS
        for idx, folder in enumerate(self.source_folders):
            if idx not in files_by_idx:
                ft = self._get_effective_folder_type(idx, folder)
                if ft == FolderType.TRANSITION and folder.is_dir():
                    trans_files = sorted(
                        [item.name for item in folder.iterdir()
                         if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS],
                        key=str.lower
                    )
                    # Apply trim settings to transition folders
                    fid = self._folder_ids[idx]
                    ts, te = self._folder_trim_settings.get(fid, (0, 0))
                    if ts > 0 or te > 0:
                        total_t = len(trans_files)
                        ts = min(ts, max(0, total_t - 1))
                        te = min(te, max(0, total_t - 1 - ts))
                        trans_files = trans_files[ts:total_t - te]
                    if trans_files:
                        files_by_idx[idx] = trans_files

        # Check if transitions are enabled
        if not self.transition_group.isChecked():
            # Just show symlinks in Main column only
            for frame_num, (source_dir, filename, folder_idx, file_idx, _fid) in enumerate(files):
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}"
                item = QTreeWidgetItem([f"{seq_name} ({filename})", "", str(frame_num)])
                item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename, folder_idx, file_idx, 'symlink'))
                item.setData(0, Qt.ItemDataRole.UserRole + 2, _fid)
                self.sequence_table.addTopLevelItem(item)
            self._sequence_frame_count = len(files)
            self.sequence_table.setUpdatesEnabled(True)
            self._update_timeline_display()
            self._update_export_range_max()
            return

        # Build index-keyed overrides and per-transition dicts for blender
        overrides_by_idx: dict[int, FolderType] = {}
        per_trans_by_idx: dict[int, PerTransitionSettings] = {}
        for i in range(len(self.source_folders)):
            fid = self._folder_ids[i]
            if fid in self._folder_type_overrides:
                overrides_by_idx[i] = self._folder_type_overrides[fid]
            if fid in self._per_transition_settings:
                per_trans_by_idx[i] = self._per_transition_settings[fid]

        # Get transition specs
        settings = self._get_transition_settings()
        generator = TransitionGenerator(settings)
        transitions = generator.identify_transition_boundaries(
            self.source_folders,
            files_by_idx,
            overrides_by_idx,
            per_trans_by_idx
        )

        # Build lookup for transitions by position index
        trans_at_main_end: dict[int, TransitionSpec] = {}
        trans_at_trans_start: dict[int, TransitionSpec] = {}
        for trans in transitions:
            trans_at_main_end[trans.main_folder_idx] = trans
            trans_at_trans_start[trans.trans_folder_idx] = trans

        # Find consecutive MAIN folders (for direct interpolation)
        consecutive_main_pairs: list[tuple[int, int]] = []
        for i in range(len(self.source_folders) - 1):
            folder_a = self.source_folders[i]
            folder_b = self.source_folders[i + 1]
            type_a = self._get_effective_folder_type(i, folder_a)
            type_b = self._get_effective_folder_type(i + 1, folder_b)
            # Two consecutive MAIN folders with no transition between them
            if type_a == FolderType.MAIN and type_b == FolderType.MAIN:
                if i not in trans_at_main_end:
                    consecutive_main_pairs.append((i, i + 1))

        # Process each folder
        output_seq = 0  # Track continuous sequence number for preview
        for folder_idx, folder in enumerate(self.source_folders):
            folder_files = files_by_idx.get(folder_idx, [])
            if not folder_files:
                continue

            num_files = len(folder_files)
            trans_at_end = trans_at_main_end.get(folder_idx)
            trans_at_start = trans_at_trans_start.get(folder_idx)
            folder_type = self._get_effective_folder_type(folder_idx, folder)

            for file_idx, filename in enumerate(folder_files):
                should_blend = False
                blend_trans = None
                blend_idx_in_overlap = 0

                # Check if in blend zone at end of folder
                if trans_at_end:
                    left_overlap = trans_at_end.left_overlap
                    main_overlap_start = num_files - left_overlap
                    if file_idx >= main_overlap_start:
                        should_blend = True
                        blend_trans = trans_at_end
                        blend_idx_in_overlap = file_idx - main_overlap_start

                # Check if in consumed zone at start of folder (skip these)
                # But don't skip if the frame is also in the blend zone at the end
                if trans_at_start and not should_blend:
                    right_overlap = trans_at_start.right_overlap
                    if file_idx < right_overlap:
                        # These frames are consumed by the blend - skip them
                        continue

                seq_name = f"seq_{output_seq:05d}"

                if should_blend and blend_trans:
                    # Calculate which trans frame this blends with
                    output_count = max(blend_trans.left_overlap, blend_trans.right_overlap)
                    t = blend_idx_in_overlap / (output_count - 1) if output_count > 1 else 0
                    trans_pos = t * (blend_trans.right_overlap - 1) if blend_trans.right_overlap > 1 else 0
                    trans_idx = min(round(trans_pos), blend_trans.right_overlap - 1)
                    trans_file = blend_trans.trans_files[trans_idx]

                    # Outgoing frame with [B] marker, incoming frame with arrow
                    main_text = f"[B] {seq_name} ({filename})"
                    trans_text = f"→ {trans_file}"

                    item = QTreeWidgetItem([main_text, trans_text, str(output_seq)])
                    item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'blend', output_seq))
                    item.setData(1, Qt.ItemDataRole.UserRole, (blend_trans.trans_folder, trans_file))
                    # Blue color for blend frames
                    item.setForeground(0, QColor(100, 150, 255))
                    item.setForeground(1, QColor(100, 150, 255))
                    output_seq += 1
                elif folder_type == FolderType.TRANSITION:
                    # Transition folder middle frames — output as symlinks just like MAIN
                    item = QTreeWidgetItem([f"[T] {seq_name} ({filename})", "", str(output_seq)])
                    item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'symlink', output_seq))
                    item.setForeground(0, QColor(180, 140, 255))  # Purple tint for transition frames
                    output_seq += 1
                else:
                    # Main folder files go in Main column only
                    item = QTreeWidgetItem([f"{seq_name} ({filename})", "", str(output_seq)])
                    item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'symlink', output_seq))
                    output_seq += 1

                item.setData(0, Qt.ItemDataRole.UserRole + 2, self._folder_ids[folder_idx])
                self.sequence_table.addTopLevelItem(item)

            # Check if this folder starts a direct interpolation gap
            # (current MAIN followed by another MAIN with no transition)
            for pair_idx_a, pair_idx_b in consecutive_main_pairs:
                if folder_idx == pair_idx_a:
                    fid = self._folder_ids[folder_idx]
                    # Add direct interpolation row after this folder's files
                    self._add_direct_interpolation_row(fid, folder, pair_idx_b)

        self._sequence_frame_count = output_seq
        self.sequence_table.setUpdatesEnabled(True)
        # Update timeline display after rebuilding sequence table
        self._update_timeline_display()
        self._update_export_range_max()

    def _add_direct_interpolation_row(self, fid: int, after_folder: Path, next_folder_idx: int) -> None:
        """Add a clickable direct interpolation row between MAIN sequences.

        Args:
            fid: Folder entry ID for the folder after which interpolation occurs.
            after_folder: The folder Path.
            next_folder_idx: Index of the next MAIN folder.
        """
        direct_settings = self._direct_transitions.get(fid)

        if direct_settings and direct_settings.enabled:
            # Configured: show green row with settings + placeholder frames
            method_name = direct_settings.method.value.upper()
            frame_count = direct_settings.frame_count

            # Header row (clickable to edit)
            header_text = f"  [{method_name}: {frame_count} frames] (click to edit)"
            header_item = QTreeWidgetItem([header_text, ""])
            header_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_header', fid))
            header_item.setForeground(0, QColor(50, 180, 100))  # Green
            header_item.setFlags(header_item.flags() & ~Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.sequence_table.addTopLevelItem(header_item)

            # Add placeholder rows for each interpolated frame
            for i in range(frame_count):
                placeholder_text = f"    [{method_name} {i + 1}/{frame_count}]"
                placeholder_item = QTreeWidgetItem([placeholder_text, ""])
                placeholder_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_placeholder', fid, i))
                placeholder_item.setForeground(0, QColor(100, 180, 220))  # Light blue
                # Make placeholders non-selectable
                placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.sequence_table.addTopLevelItem(placeholder_item)
        else:
            # Unconfigured: show grey "+" row
            add_text = "  [+ Add RIFE/FILM transition] (click to configure)"
            add_item = QTreeWidgetItem([add_text, ""])
            add_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_add', fid))
            add_item.setForeground(0, QColor(150, 150, 150))  # Grey
            add_item.setFlags(add_item.flags() & ~Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.sequence_table.addTopLevelItem(add_item)

    def _on_sequence_table_selected(self, current, previous) -> None:
        """Handle sequence table row selection - show image in preview."""
        if current is None:
            return

        # Check column 0 first (Main frame), then column 1 (Transition frame)
        data0 = current.data(0, Qt.ItemDataRole.UserRole)
        data1 = current.data(1, Qt.ItemDataRole.UserRole)

        data = data0 if data0 else data1
        if not data:
            return

        # Handle direct interpolation rows
        if isinstance(data, tuple) and len(data) >= 2:
            if data[0] == 'direct_add':
                # "+" row - only open dialog if not playing (skip during playback)
                if not self.sequence_playing:
                    self._show_direct_transition_dialog(data[1])
                return
            elif data[0] == 'direct_header':
                # Header row - only open dialog if not playing (skip during playback)
                if not self.sequence_playing:
                    self._show_direct_transition_dialog(data[1])
                return
            elif data[0] == 'direct_placeholder':
                # Show preview of interpolated frame
                after_fid = data[1]
                frame_index = data[2]
                self._show_direct_interpolation_preview(after_fid, frame_index)
                return

        # Sync source list selection so the trim slider shows this frame's folder
        fid = current.data(0, Qt.ItemDataRole.UserRole + 2)
        if fid is not None:
            self._select_folder_in_lists(fid)

        frame_type = data[4] if len(data) > 4 else 'symlink'

        # For blend frames, generate cross-dissolve preview
        if frame_type == 'blend' and data0 and data1:
            self._show_blend_preview(current, data0, data1)
        else:
            # Regular frame - just show the image
            source_dir, filename = data[0], data[1]
            image_path = source_dir / filename

            if not image_path.exists():
                self.image_label.setText(f"Image not found:\n{image_path}")
                self.image_name_label.setText("")
                self._current_pixmap = None
                return

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                self.image_label.setText(f"Cannot load image:\n{image_path}")
                self.image_name_label.setText("")
                self._current_pixmap = None
                return

            self._current_pixmap = pixmap
            self._apply_zoom()

            # Update labels
            row_idx = self.sequence_table.indexOfTopLevelItem(current)
            total = self.sequence_table.topLevelItemCount()
            self.image_index_label.setText(f"{row_idx + 1} / {total}")

            # Use continuous format for transitions, folder-based for regular export
            if self.transition_group.isChecked() and len(data) > 5 and data[5] >= 0:
                seq_name = f"seq_{data[5]:05d}"
            else:
                seq_name = f"seq{data[2] + 1:02d}_{data[3]:04d}"
            self.image_name_label.setText(f"{seq_name} ({filename})")

    def _on_sequence_table_clicked(self, item, column: int) -> None:
        """Handle clicks on sequence table items, including non-selectable ones."""
        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        # Handle direct interpolation rows
        if isinstance(data, tuple) and len(data) >= 2:
            if data[0] == 'direct_add':
                # Clicked on "+" row to add direct transition
                self._show_direct_transition_dialog(data[1])
            elif data[0] == 'direct_header':
                # Clicked on configured direct transition header
                self._show_direct_transition_dialog(data[1])
            elif data[0] == 'direct_placeholder':
                # Clicked on placeholder row - show preview of interpolated frame
                after_fid = data[1]
                frame_index = data[2]
                self._show_direct_interpolation_preview(after_fid, frame_index)

    def _show_blend_preview(self, item, data0, data1) -> None:
        """Show a cross-dissolve preview for a blend frame."""
        from PIL import Image
        from PIL.ImageQt import ImageQt

        # Get source paths
        main_dir, main_file = data0[0], data0[1]
        trans_dir, trans_file = data1[0], data1[1]
        main_path = main_dir / main_file
        trans_path = trans_dir / trans_file

        if not main_path.exists() or not trans_path.exists():
            self.image_label.setText(f"Image not found")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        try:
            # Calculate blend factor based on position in sequence table
            # Find this frame's position in the blend sequence
            row_idx = self.sequence_table.indexOfTopLevelItem(item)
            total = self.sequence_table.topLevelItemCount()

            # Count blend frames to determine factor
            blend_start = -1
            blend_count = 0
            blend_position = 0
            for i in range(total):
                check_item = self.sequence_table.topLevelItem(i)
                check_data = check_item.data(0, Qt.ItemDataRole.UserRole)
                if check_data and len(check_data) > 4 and check_data[4] == 'blend':
                    # Check if same transition (same trans folder)
                    check_trans = check_item.data(1, Qt.ItemDataRole.UserRole)
                    if check_trans and check_trans[0] == trans_dir:
                        if blend_start < 0:
                            blend_start = i
                        blend_count += 1
                        if i == row_idx:
                            blend_position = blend_count - 1

            # Calculate factor
            if blend_count > 1:
                factor = blend_position / (blend_count - 1)
            else:
                factor = 0.5

            # Apply curve from settings
            settings = self._get_transition_settings()
            from core import ImageBlender
            factor = ImageBlender.calculate_blend_factor(
                blend_position, blend_count, settings.blend_curve
            )

            # Create cache key (include method-specific settings)
            cache_key = f"{main_path}|{trans_path}|{factor:.6f}|{settings.blend_method.value}|{settings.blend_curve.value}"
            if settings.blend_method == BlendMethod.RIFE:
                cache_key += f"|{settings.rife_model}|{settings.rife_uhd}|{settings.rife_tta}"
            elif settings.blend_method == BlendMethod.OPTICAL_FLOW:
                cache_key += f"|{settings.of_levels}|{settings.of_winsize}|{settings.of_iterations}|{settings.of_poly_n}|{settings.of_poly_sigma}"

            # Check cache first
            if cache_key in self._blend_preview_cache:
                pixmap = self._blend_preview_cache[cache_key]
            else:
                # Load images
                img_a = Image.open(main_path)
                img_b = Image.open(trans_path)

                # Resize B to match A if needed
                if img_a.size != img_b.size:
                    img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

                # Convert to RGBA
                if img_a.mode != 'RGBA':
                    img_a = img_a.convert('RGBA')
                if img_b.mode != 'RGBA':
                    img_b = img_b.convert('RGBA')

                # Blend images using selected method
                if settings.blend_method == BlendMethod.OPTICAL_FLOW:
                    blended = ImageBlender.optical_flow_blend(
                        img_a, img_b, factor,
                        levels=settings.of_levels,
                        winsize=settings.of_winsize,
                        iterations=settings.of_iterations,
                        poly_n=settings.of_poly_n,
                        poly_sigma=settings.of_poly_sigma
                    )
                elif settings.blend_method == BlendMethod.RIFE:
                    blended = ImageBlender.rife_blend(
                        img_a, img_b, factor, settings.rife_binary_path,
                        model=settings.rife_model,
                        uhd=settings.rife_uhd,
                        tta=settings.rife_tta
                    )
                elif settings.blend_method == BlendMethod.RIFE_PRACTICAL:
                    blended = ImageBlender.practical_rife_blend(
                        img_a, img_b, factor,
                        settings.practical_rife_model,
                        settings.practical_rife_ensemble
                    )
                else:
                    blended = Image.blend(img_a, img_b, factor)

                # Convert to QPixmap
                qim = ImageQt(blended.convert('RGBA'))
                pixmap = QPixmap.fromImage(qim)

                # Store in cache
                self._blend_preview_cache[cache_key] = pixmap

                img_a.close()
                img_b.close()

            self._current_pixmap = pixmap
            self._apply_zoom()

            # Update labels
            self.image_index_label.setText(f"{row_idx + 1} / {total}")
            # Use continuous format for blend frames (always with transitions)
            output_seq_num = data0[5] if len(data0) > 5 else row_idx
            seq_name = f"seq_{output_seq_num:05d}"
            self.image_name_label.setText(f"[B] {seq_name} ({main_file} + {trans_file}) @ {factor:.0%}")

        except Exception as e:
            self.image_label.setText(f"Error generating blend preview:\n{e}")
            self.image_name_label.setText("")
            self._current_pixmap = None

    def _show_direct_interpolation_preview(self, after_fid: int, frame_index: int) -> None:
        """Generate and show a preview for a direct interpolation placeholder frame.

        For RIFE: Generates one frame at a time (RIFE handles arbitrary timesteps well).
        For FILM: Generates ALL frames at once on first click (FILM works best this way),
                  then caches all frames for instant subsequent access.

        Args:
            after_fid: The folder entry ID after which the interpolation occurs.
            frame_index: The index of the interpolated frame (0-based).
        """
        from PIL import Image
        from PIL.ImageQt import ImageQt
        from core import ImageBlender

        # Get direct transition settings
        direct_settings = self._direct_transitions.get(after_fid)
        if not direct_settings or not direct_settings.enabled:
            self.image_label.setText("Direct interpolation not configured")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Find the folder index and next folder
        if after_fid not in self._folder_ids:
            self.image_label.setText("Folder not found in sequence")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return
        folder_idx = self._folder_ids.index(after_fid)
        after_folder = self.source_folders[folder_idx]

        if folder_idx >= len(self.source_folders) - 1:
            self.image_label.setText("No next folder for interpolation")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        next_folder = self.source_folders[folder_idx + 1]
        next_fid = self._folder_ids[folder_idx + 1]

        # Get files for both folders keyed by fid
        files = self._get_files_in_order()
        files_by_fid: dict[int, list[str]] = {}
        for source_dir, filename, f_idx, file_idx, fid in files:
            if fid not in files_by_fid:
                files_by_fid[fid] = []
            files_by_fid[fid].append(filename)

        after_files = files_by_fid.get(after_fid, [])
        next_files = files_by_fid.get(next_fid, [])

        if not after_files or not next_files:
            self.image_label.setText("Missing frames for interpolation")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Get last frame of after_folder and first frame of next_folder
        last_frame_path = after_folder / after_files[-1]
        first_frame_path = next_folder / next_files[0]

        if not last_frame_path.exists() or not first_frame_path.exists():
            self.image_label.setText(f"Frame files not found")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Calculate timestep
        frame_count = direct_settings.frame_count
        t = (frame_index + 1) / (frame_count + 1)  # Evenly spaced between 0 and 1

        # Create cache key - include frame_count so changing count invalidates cache
        cache_key = f"direct|{after_fid}|{frame_index}|{direct_settings.method.value}|{frame_count}"

        try:
            # Check cache first
            if cache_key in self._blend_preview_cache:
                pixmap = self._blend_preview_cache[cache_key]
            elif direct_settings.method == DirectInterpolationMethod.FILM and FilmEnv.is_setup():
                # FILM: Generate ALL frames at once for better quality
                # Check if we need to generate (first frame not cached means none are)
                first_cache_key = f"direct|{after_fid}|0|{direct_settings.method.value}|{frame_count}"
                if first_cache_key not in self._blend_preview_cache:
                    # Generate all frames at once
                    error_msg = self._generate_all_film_preview_frames(
                        after_fid, last_frame_path, first_frame_path, frame_count
                    )
                    if error_msg:
                        # Error already displayed in image_label by the method
                        self._current_pixmap = None
                        return

                # Now retrieve the specific frame from cache
                if cache_key in self._blend_preview_cache:
                    pixmap = self._blend_preview_cache[cache_key]
                else:
                    # Fallback if batch generation failed
                    self.image_label.setText("FILM batch generation failed - check console for details")
                    self.image_name_label.setText("")
                    self._current_pixmap = None
                    return
            else:
                # RIFE (or FILM not set up): Generate one frame at a time
                # Load images
                img_a = Image.open(last_frame_path)
                img_b = Image.open(first_frame_path)

                # Resize B to match A if needed
                if img_a.size != img_b.size:
                    img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

                # Convert to RGBA
                if img_a.mode != 'RGBA':
                    img_a = img_a.convert('RGBA')
                if img_b.mode != 'RGBA':
                    img_b = img_b.convert('RGBA')

                # Generate interpolated frame
                if direct_settings.method == DirectInterpolationMethod.FILM:
                    # FILM not set up, use fallback
                    blended = ImageBlender.film_blend(img_a, img_b, t)
                else:  # RIFE
                    settings = self._get_transition_settings()
                    blended = ImageBlender.practical_rife_blend(
                        img_a, img_b, t,
                        settings.practical_rife_model,
                        settings.practical_rife_ensemble
                    )

                # Convert to QPixmap
                qim = ImageQt(blended.convert('RGBA'))
                pixmap = QPixmap.fromImage(qim)

                # Store in cache
                self._blend_preview_cache[cache_key] = pixmap

                img_a.close()
                img_b.close()

            self._current_pixmap = pixmap
            self._apply_zoom()

            # Update labels
            method_name = direct_settings.method.value.upper()
            self.image_name_label.setText(
                f"[{method_name} {frame_index + 1}/{frame_count}] @ t={t:.2f}"
            )

            # Find the item index in the table for image_index_label
            for i in range(self.sequence_table.topLevelItemCount()):
                item = self.sequence_table.topLevelItem(i)
                item_data = item.data(0, Qt.ItemDataRole.UserRole)
                if (isinstance(item_data, tuple) and len(item_data) >= 3 and
                    item_data[0] == 'direct_placeholder' and
                    item_data[1] == after_fid and
                    item_data[2] == frame_index):
                    total = self.sequence_table.topLevelItemCount()
                    self.image_index_label.setText(f"{i + 1} / {total}")
                    break

        except Exception as e:
            self.image_label.setText(f"Error generating interpolation preview:\n{e}")
            self.image_name_label.setText("")
            self._current_pixmap = None

    def _generate_all_film_preview_frames(
        self,
        after_fid: int,
        last_frame_path: Path,
        first_frame_path: Path,
        frame_count: int
    ) -> Optional[str]:
        """Generate all FILM preview frames at once and cache them.

        FILM works best when generating all frames at once using its
        recursive approach. This method generates all frames and stores
        them in the preview cache.

        Args:
            after_fid: The folder entry ID after which the interpolation occurs.
            last_frame_path: Path to the last frame of the current sequence.
            first_frame_path: Path to the first frame of the next sequence.
            frame_count: Number of frames to generate.

        Returns:
            None on success, error message string on failure.
        """
        from PIL import Image
        from PIL.ImageQt import ImageQt
        import tempfile

        # Show progress dialog
        progress = QProgressDialog(
            f"Generating {frame_count} FILM frames...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("FILM Interpolation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(10)
        QApplication.processEvents()

        try:
            # Use a temp directory for FILM batch output
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                progress.setLabelText("Running FILM batch interpolation...")
                progress.setValue(20)
                QApplication.processEvents()

                # Run batch interpolation
                success, error, output_paths = FilmEnv.run_batch_interpolation(
                    last_frame_path,
                    first_frame_path,
                    tmp_path,
                    frame_count,
                    'frame_{:04d}.png'
                )

                if not success:
                    progress.close()
                    error_msg = f"FILM error: {error}"
                    self.image_label.setText(error_msg)
                    self.image_name_label.setText("")
                    return error_msg

                progress.setLabelText("Loading generated frames...")
                progress.setValue(70)
                QApplication.processEvents()

                # Load all frames and cache them
                for i, output_path in enumerate(output_paths):
                    if progress.wasCanceled():
                        break

                    if output_path.exists():
                        frame = Image.open(output_path)
                        qim = ImageQt(frame.convert('RGBA'))
                        pixmap = QPixmap.fromImage(qim)

                        # Cache with the standard key format (include frame_count)
                        cache_key = f"direct|{after_fid}|{i}|film|{frame_count}"
                        self._blend_preview_cache[cache_key] = pixmap

                        frame.close()

                    # Update progress
                    pct = 70 + int(30 * (i + 1) / frame_count)
                    progress.setValue(pct)
                    QApplication.processEvents()

            progress.close()
            return None  # Success

        except Exception as e:
            progress.close()
            error_msg = f"FILM batch error: {e}"
            self.image_label.setText(error_msg)
            self.image_name_label.setText("")
            return error_msg

    def _update_timeline_display(self) -> None:
        """Update the timeline duration display based on frame count and FPS."""
        frame_count = self.sequence_table.topLevelItemCount()
        fps = self.fps_spin.value()

        if fps > 0 and frame_count > 0:
            total_seconds = frame_count / fps
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.timeline_label.setText(
                f"Duration: {minutes:02d}:{seconds:06.3f} ({frame_count} frames @ {fps}fps)"
            )
        else:
            self.timeline_label.setText("Duration: 00:00.000 (0 frames)")

        # Refresh the sequence table to update timeline background
        self.sequence_table.viewport().update()

    def _toggle_sequence_play(self) -> None:
        """Toggle sequence playback."""
        if self.sequence_playing:
            self._stop_sequence_play()
        else:
            self._start_sequence_play()

    def _start_sequence_play(self) -> None:
        """Start playing the image sequence."""
        if self.sequence_table.topLevelItemCount() == 0:
            return

        fps = self.fps_spin.value()
        interval = int(1000 / fps)  # milliseconds per frame
        self.sequence_timer.setInterval(interval)
        self.sequence_timer.start()
        self.sequence_playing = True
        self.seq_play_btn.setText("⏸ Pause")

        # If no item selected, start from first
        if self.sequence_table.currentItem() is None:
            first_item = self.sequence_table.topLevelItem(0)
            if first_item:
                self.sequence_table.setCurrentItem(first_item)

    def _stop_sequence_play(self) -> None:
        """Stop sequence playback."""
        self.sequence_timer.stop()
        self.sequence_playing = False
        self.seq_play_btn.setText("▶ Play")

    def _advance_sequence_frame(self) -> None:
        """Advance to next frame in sequence."""
        current_item = self.sequence_table.currentItem()
        if current_item is None:
            self._stop_sequence_play()
            return

        current_idx = self.sequence_table.indexOfTopLevelItem(current_item)
        total = self.sequence_table.topLevelItemCount()

        # Find next valid frame (skip direct_add and direct_header rows)
        next_idx = current_idx + 1
        while next_idx < total:
            next_item = self.sequence_table.topLevelItem(next_idx)
            data = next_item.data(0, Qt.ItemDataRole.UserRole)
            # Skip non-frame rows (direct_add, direct_header)
            if isinstance(data, tuple) and len(data) >= 1 and data[0] in ('direct_add', 'direct_header'):
                next_idx += 1
                continue
            # Found a valid frame
            self.sequence_table.setCurrentItem(next_item)
            return

        # Reached end - stop playback
        self._stop_sequence_play()

    def _browse_trans_destination(self) -> None:
        """Select transition destination folder via file dialog."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Transition Destination Folder", start_dir
        )
        if path:
            self._add_to_path_history(self.trans_dst_path, path)
            self.last_directory = str(Path(path).parent)

    def _add_source_folder(
        self,
        folder_path: Optional[str] = None,
        folder_type: Optional[FolderType] = None,
        insert_index: Optional[int] = None,
    ) -> None:
        """Add a source folder via file dialog or direct path.

        Args:
            folder_path: Path string, or None to open a file dialog.
            folder_type: Explicit type override (MAIN/TRANSITION). None = position-based.
            insert_index: Position in source_folders to insert at. None = append.
        """
        if folder_path and not isinstance(folder_path, str):
            folder_path = None

        if folder_path:
            path = folder_path
        else:
            start_dir = self.last_directory or ""
            path = QFileDialog.getExistingDirectory(
                self, "Select Source Folder", start_dir
            )

        if path:
            folder = Path(path).resolve()
            if folder.is_dir():
                fid = self._allocate_folder_id()
                if insert_index is not None and 0 <= insert_index <= len(self.source_folders):
                    # Pin effective types for all folders at/after the insert point
                    # so the index shift doesn't flip their position-based types.
                    for j in range(insert_index, len(self.source_folders)):
                        jfid = self._folder_ids[j]
                        if jfid not in self._folder_type_overrides:
                            self._folder_type_overrides[jfid] = self._get_effective_folder_type(j, self.source_folders[j])
                    self.source_folders.insert(insert_index, folder)
                    self._folder_ids.insert(insert_index, fid)
                else:
                    self.source_folders.append(folder)
                    self._folder_ids.append(fid)
                if folder_type is not None and folder_type != FolderType.AUTO:
                    self._folder_type_overrides[fid] = folder_type
                self.last_directory = str(folder.parent)
                self._sync_dual_lists()
                self._refresh_files()
                self._update_flow_arrows()

    def _sync_dual_lists(self) -> None:
        """Synchronize the source list with source_folders.

        Single unified list where:
        - Odd positions (1, 3, 5...) = Main [M] folders
        - Even positions (2, 4, 6...) = Transition [T] folders
        """
        self.source_list.clear()

        # Calculate common prefix to compress paths
        common_prefix = ""
        if len(self.source_folders) > 1:
            paths = [str(f) for f in self.source_folders]
            # Find common prefix
            common_prefix = os.path.commonpath(paths) if paths else ""
            # Only use prefix if it's a meaningful directory (not just "/")
            if len(common_prefix) <= 1:
                common_prefix = ""

        main_folder_indices = self._get_main_folder_indices()
        sub_indices = self._get_main_folder_sub_indices()

        num_folders = len(self.source_folders)
        for i, folder in enumerate(self.source_folders):
            fid = self._folder_ids[i]
            folder_type = self._get_effective_folder_type(i, folder)

            # Get per-transition settings if available
            pts = self._per_transition_settings.get(fid)
            overlap_text = ""
            if pts and folder_type == FolderType.TRANSITION:
                overlap_text = f" [L:{pts.left_overlap} R:{pts.right_overlap}]"

            # Type indicator with sub-sequence label for split folders
            sub_idx = sub_indices.get(fid, 0)
            if sub_idx > 0 and folder_type == FolderType.MAIN:
                base_idx = main_folder_indices.get(fid, 0)
                type_tag = f"[M{base_idx + 1}-{sub_idx}]"
            else:
                type_tag = "[M]" if folder_type == FolderType.MAIN else "[T]"

            # Compress path if common prefix exists
            if common_prefix:
                display_path = "[...]" + str(folder)[len(common_prefix):]
            else:
                display_path = str(folder)

            # Show path with index and type
            display_name = f"{i+1}. {type_tag} {display_path}{overlap_text}"
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, (folder, fid))
            item.setToolTip(str(folder))  # Full path on hover

            # Color and alignment: Main = left/default, Transition = right/purple
            if folder_type == FolderType.TRANSITION:
                item.setForeground(QColor(155, 89, 182))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self.source_list.addItem(item)

            # After a MAIN folder, insert a placeholder if the next folder is not TRANSITION
            if folder_type == FolderType.MAIN:
                next_is_transition = False
                if i + 1 < num_folders:
                    next_type = self._get_effective_folder_type(
                        i + 1, self.source_folders[i + 1]
                    )
                    next_is_transition = (next_type == FolderType.TRANSITION)
                if not next_is_transition:
                    ph = QListWidgetItem("   [T] (click or drop to add transition)")
                    ph.setData(Qt.ItemDataRole.UserRole, None)
                    ph.setData(Qt.ItemDataRole.UserRole + 1, i + 1)  # insert index
                    ph.setForeground(QColor(130, 130, 130))
                    font = ph.font()
                    font.setItalic(True)
                    ph.setFont(font)
                    ph.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                    ph.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    self.source_list.addItem(ph)

    def _get_effective_folder_type(self, index: int, folder: Path) -> FolderType:
        """Get the effective folder type considering overrides."""
        if index < len(self._folder_ids):
            fid = self._folder_ids[index]
            if fid in self._folder_type_overrides:
                override = self._folder_type_overrides[fid]
                if override != FolderType.AUTO:
                    return override
        return FolderType.MAIN if index % 2 == 0 else FolderType.TRANSITION

    def _is_placeholder_item(self, item: QListWidgetItem) -> bool:
        """Return True if the item is a transition placeholder slot."""
        if item is None:
            return False
        return (
            item.data(Qt.ItemDataRole.UserRole) is None
            and item.data(Qt.ItemDataRole.UserRole + 1) is not None
        )

    def _get_placeholder_insert_index(self, item: QListWidgetItem) -> Optional[int]:
        """Return the source_folders insert index stored on a placeholder item, or None."""
        if item is None:
            return None
        idx = item.data(Qt.ItemDataRole.UserRole + 1)
        return int(idx) if idx is not None else None

    def _get_main_folder_indices(self) -> dict[int, int]:
        """Return a map from each MAIN folder's fid to its MAIN-only sequence index.

        Transition folders are excluded so that inserting or removing a
        transition never changes the seq numbers of existing MAIN folders.

        Consecutive MAIN folders with the same resolved path (split sequences)
        share the same base index so that splitting doesn't shift subsequent
        sequence numbers.
        """
        indices: dict[int, int] = {}
        main_count = 0
        last_main_resolved: Optional[str] = None
        for i, folder in enumerate(self.source_folders):
            if self._get_effective_folder_type(i, folder) == FolderType.MAIN:
                fid = self._folder_ids[i]
                resolved = str(folder.resolve())
                if last_main_resolved is not None and resolved != last_main_resolved:
                    main_count += 1
                indices[fid] = main_count
                last_main_resolved = resolved
        return indices

    def _get_main_folder_sub_indices(self) -> dict[int, int]:
        """Return sub-index for each MAIN folder's fid.

        Returns 0 for non-split folders.  For consecutive MAIN folders
        sharing the same resolved path (split), returns 1, 2, ... so
        they can be labelled seq01-1, seq01-2, etc.
        """
        sub_indices: dict[int, int] = {}
        last_main_resolved: Optional[str] = None
        group_fids: list[int] = []

        def _finalize_group():
            if len(group_fids) > 1:
                for j, gfid in enumerate(group_fids):
                    sub_indices[gfid] = j + 1
            elif group_fids:
                sub_indices[group_fids[0]] = 0

        for i, folder in enumerate(self.source_folders):
            if self._get_effective_folder_type(i, folder) == FolderType.MAIN:
                fid = self._folder_ids[i]
                resolved = str(folder.resolve())
                if resolved == last_main_resolved:
                    group_fids.append(fid)
                else:
                    _finalize_group()
                    group_fids = [fid]
                last_main_resolved = resolved

        _finalize_group()
        return sub_indices

    def _update_flow_arrows(self) -> None:
        """Update visual indicators."""
        pass

    def _get_selected_folder(self) -> Optional[tuple[Path, int]]:
        """Get the currently selected folder and its index in source_folders."""
        selected = self.source_list.selectedItems()
        if selected:
            data = selected[0].data(Qt.ItemDataRole.UserRole)
            if data is not None and isinstance(data, tuple) and len(data) == 2:
                folder, fid = data
                if fid in self._folder_ids:
                    return folder, self._folder_ids.index(fid)
        return None

    def _move_folder_up(self) -> None:
        """Move the selected folder up in the sequence."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result
        if idx > 0:
            fid_a = self._folder_ids[idx]
            fid_b = self._folder_ids[idx - 1]
            type_a = self._get_effective_folder_type(idx, folder)
            type_b = self._get_effective_folder_type(idx - 1, self.source_folders[idx - 1])
            self.source_folders[idx], self.source_folders[idx - 1] = self.source_folders[idx - 1], folder
            self._folder_ids[idx], self._folder_ids[idx - 1] = fid_b, fid_a
            self._folder_type_overrides[fid_a] = type_a
            self._folder_type_overrides[fid_b] = type_b
            self._sync_dual_lists()
            self._refresh_files()
            self._update_flow_arrows()
            self._select_folder_in_lists(fid_a)

    def _move_folder_down(self) -> None:
        """Move the selected folder down in the sequence."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result
        if idx < len(self.source_folders) - 1:
            fid_a = self._folder_ids[idx]
            fid_b = self._folder_ids[idx + 1]
            type_a = self._get_effective_folder_type(idx, folder)
            type_b = self._get_effective_folder_type(idx + 1, self.source_folders[idx + 1])
            self.source_folders[idx], self.source_folders[idx + 1] = self.source_folders[idx + 1], folder
            self._folder_ids[idx], self._folder_ids[idx + 1] = fid_b, fid_a
            self._folder_type_overrides[fid_a] = type_a
            self._folder_type_overrides[fid_b] = type_b
            self._sync_dual_lists()
            self._refresh_files()
            self._update_flow_arrows()
            self._select_folder_in_lists(fid_a)

    def _select_folder_in_lists(self, fid: int) -> None:
        """Select a folder in the source list widget by its entry ID."""
        for i in range(self.source_list.count()):
            item = self.source_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and len(data) == 2 and data[1] == fid:
                self.source_list.setCurrentItem(item)
                return

    def _remove_source_folder(self) -> None:
        """Remove selected source folder(s), preserving sequence order of remaining files."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result
        fid = self._folder_ids[idx]

        if fid in self._folder_type_overrides:
            del self._folder_type_overrides[fid]
        if fid in self._per_transition_settings:
            del self._per_transition_settings[fid]
        if fid in self._folder_trim_settings:
            del self._folder_trim_settings[fid]
        if fid in self._folder_file_counts:
            del self._folder_file_counts[fid]
        if fid in self._removed_files:
            del self._removed_files[fid]
        if fid in self._direct_transitions:
            del self._direct_transitions[fid]

        del self.source_folders[idx]
        del self._folder_ids[idx]

        self._sync_dual_lists()

        # Remove only files from the deleted folder, preserving order of others
        self._remove_files_from_folder(folder)

        # Renumber sequence names to reflect new folder indices
        self._recalculate_sequence_names()

        # Update the sequence table (With Transitions tab)
        self._update_sequence_table()
        self._update_export_range_max()

        self._update_flow_arrows()

    def _replace_source_folder(self, old_folder: Path, idx: int) -> None:
        """Replace a source folder with a new one, preserving edits.

        Keeps: folder type override, trim settings, per-transition settings,
        removed files set, direct transition settings, and position in list.
        The fid stays the same — all settings keyed by fid are preserved automatically.
        """
        start_dir = str(old_folder.parent) if old_folder.exists() else (self.last_directory or "")
        path = QFileDialog.getExistingDirectory(
            self, "Select Replacement Folder", start_dir
        )
        if not path:
            return

        new_folder = Path(path).resolve()
        if new_folder == old_folder:
            return

        # Keep same fid — settings are keyed by fid, not path
        fid = self._folder_ids[idx]
        self.source_folders[idx] = new_folder

        # Update per-transition settings folder reference
        if fid in self._per_transition_settings:
            pts = self._per_transition_settings[fid]
            self._per_transition_settings[fid] = PerTransitionSettings(
                trans_folder=new_folder,
                left_overlap=pts.left_overlap,
                right_overlap=pts.right_overlap,
            )

        # Clear file counts since the folder content changed
        if fid in self._folder_file_counts:
            del self._folder_file_counts[fid]

        self.last_directory = str(new_folder.parent)
        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

    def _remove_files_from_folder(self, folder: Path) -> None:
        """Remove all files from a specific folder without affecting order of other files."""
        folder_str = str(folder)
        rows_to_remove = []

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            if item and item.text(2) == folder_str:
                rows_to_remove.append(i)

        # Remove in reverse order to preserve indices
        for row in reversed(rows_to_remove):
            self.file_list.takeTopLevelItem(row)

        # Clean up separators (remove consecutive or leading/trailing separators)
        self._cleanup_separators()

        # Update slider range
        total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, total - 1))

    def _cleanup_separators(self) -> None:
        """Remove unnecessary separators (consecutive, leading, or trailing)."""
        rows_to_remove = []
        prev_was_separator = True  # Treat start as "separator" to remove leading ones

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            is_separator = self._is_separator_item(item)

            if is_separator and prev_was_separator:
                rows_to_remove.append(i)
            prev_was_separator = is_separator

        # Check if last item is a separator
        if self.file_list.topLevelItemCount() > 0:
            last_item = self.file_list.topLevelItem(self.file_list.topLevelItemCount() - 1)
            if self._is_separator_item(last_item):
                last_idx = self.file_list.topLevelItemCount() - 1
                if last_idx not in rows_to_remove:
                    rows_to_remove.append(last_idx)

        # Remove in reverse order
        for row in sorted(rows_to_remove, reverse=True):
            self.file_list.takeTopLevelItem(row)

    def _remove_selected_files(self) -> None:
        """Remove selected files from the file list."""
        selected = self.file_list.selectedItems()
        if not selected:
            return

        # Track removed files for persistence
        for item in selected:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                filename = data[1]
                fid = item.data(0, Qt.ItemDataRole.UserRole + 2) or 0
                if fid not in self._removed_files:
                    self._removed_files[fid] = set()
                self._removed_files[fid].add(filename)

        rows = sorted([self.file_list.indexOfTopLevelItem(item) for item in selected], reverse=True)
        for row in rows:
            self.file_list.takeTopLevelItem(row)

        # Update the With Transitions tab to reflect the removal
        self._update_sequence_table()

    def _get_path_history_file(self) -> Path:
        """Get the path to the history JSON file."""
        cache_dir = Path.home() / '.cache' / 'video-montage-linker'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / 'path_history.json'

    def _load_path_history(self) -> None:
        """Load path history from disk and populate combo boxes."""
        history_file = self._get_path_history_file()
        if not history_file.exists():
            return

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            # Populate destination combo
            dst_history = history.get('destination', [])
            for path in dst_history:
                if Path(path).exists():
                    self.dst_path.addItem(path)

            # Populate transition destination combo
            trans_history = history.get('transition', [])
            for path in trans_history:
                if Path(path).exists():
                    self.trans_dst_path.addItem(path)

        except (json.JSONDecodeError, IOError):
            pass

    def _save_path_history(self) -> None:
        """Save path history to disk."""
        history_file = self._get_path_history_file()

        # Collect paths from combo boxes
        dst_paths = [self.dst_path.itemText(i) for i in range(self.dst_path.count())]
        trans_paths = [self.trans_dst_path.itemText(i) for i in range(self.trans_dst_path.count())]

        history = {
            'destination': dst_paths,
            'transition': trans_paths
        }

        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except IOError:
            pass

    def _add_to_path_history(self, combo: QComboBox, path: str, max_items: int = 10) -> None:
        """Add a path to the combo box history if not already present."""
        if not path:
            return

        # Normalize path
        path = str(Path(path).resolve())

        # Check if already in list
        for i in range(combo.count()):
            if combo.itemText(i) == path:
                # Move to top if not already there
                if i > 0:
                    combo.removeItem(i)
                    combo.insertItem(0, path)
                    combo.setCurrentIndex(0)
                return

        # Add to top of list
        combo.insertItem(0, path)
        combo.setCurrentIndex(0)

        # Trim to max items
        while combo.count() > max_items:
            combo.removeItem(combo.count() - 1)

        # Save history
        self._save_path_history()

    def _browse_destination(self) -> None:
        """Select destination folder via file dialog."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", start_dir
        )
        if path:
            self._add_to_path_history(self.dst_path, path)
            self.last_directory = str(Path(path).parent)
            self._try_resume_session(path)

    def _on_destination_changed(self) -> None:
        """Handle destination path text field changes."""
        path = self.dst_path.currentText().strip()
        if path and Path(path).is_dir():
            resolved = str(Path(path).resolve())
            # Add to history if it's a valid directory
            self._add_to_path_history(self.dst_path, path)
            if resolved != self._last_resumed_dest:
                self._try_resume_session(path)

    def _try_resume_session(self, dest_path: str) -> bool:
        """Try to resume the latest session for the given destination."""
        dest = Path(dest_path).resolve()
        dest_str = str(dest)

        self._last_resumed_dest = dest_str

        sessions = self.db.get_sessions_by_destination(dest_str)
        if not sessions:
            return False

        return self._restore_session_by_id(sessions[0])

    def _restore_session_by_id(self, session: 'SessionRecord', silent: bool = False) -> bool:
        """Restore a specific session by its record.

        Args:
            session: The SessionRecord to restore.
            silent: If True, don't show the summary dialog.

        Returns:
            True if session was restored successfully.
        """
        symlinks = self.db.get_symlinks_by_session(session.id)
        ordered_folders = self.db.get_ordered_folders(session.id)
        db_folder_settings = self.db.get_all_folder_settings(session.id)
        db_transition_settings = self.db.get_transition_settings(session.id)
        db_per_trans_settings = self.db.get_all_per_transition_settings(session.id)
        db_removed_files = self.db.get_removed_files(session.id)
        db_direct_transitions = self.db.get_direct_transitions(session.id)

        if not symlinks and not ordered_folders and not db_folder_settings:
            if not silent:
                QMessageBox.warning(self, "Empty Session", "This session has no data.")
            return False

        # Build file data from symlink records (only MAIN folders have these)
        # Pattern for new continuous format: seq_00000
        continuous_pattern = re.compile(r'seq_(\d+)')
        # Pattern for old folder-based format: seq01_0000
        folder_pattern = re.compile(r'seq(\d+)_(\d+)')

        folder_data: dict[str, tuple[int, list[tuple[int, str]]]] = {}
        folder_first_seq: dict[str, int] = {}
        missing_count = 0
        _folder_exists_cache: dict[str, bool] = {}

        for link in symlinks:
            source_path = Path(link.source_path)
            folder = str(source_path.parent)

            if folder not in _folder_exists_cache:
                _folder_exists_cache[folder] = Path(folder).is_dir()
            if not _folder_exists_cache[folder]:
                missing_count += 1
                continue
            link_name = Path(link.link_path).stem

            match = continuous_pattern.match(link_name)
            if match:
                seq_num = int(match.group(1))
                if folder not in folder_first_seq:
                    folder_first_seq[folder] = seq_num
                file_idx = seq_num
            else:
                match = folder_pattern.match(link_name)
                if match:
                    folder_idx_from_name = int(match.group(1)) - 1
                    file_idx = int(match.group(2))
                    if folder not in folder_first_seq:
                        folder_first_seq[folder] = folder_idx_from_name * 10000 + file_idx
                else:
                    file_idx = link.sequence_number
                    if folder not in folder_first_seq:
                        folder_first_seq[folder] = file_idx

            if folder not in folder_data:
                folder_data[folder] = (0, [])
            folder_data[folder][1].append((file_idx, link.original_filename))

        # Renumber files within each folder
        for folder in folder_data:
            sort_key = folder_first_seq.get(folder, 0)
            file_list = folder_data[folder][1]
            file_list.sort(key=lambda x: x[0])
            renumbered = [(i, fname) for i, (_, fname) in enumerate(file_list)]
            folder_data[folder] = (sort_key, renumbered)

        # Use ordered_folders as the authoritative folder list (includes TRANSITION).
        # Fall back to folder_data ordering for old sessions without folder_order.
        self.source_folders.clear()
        self._folder_ids.clear()
        self._next_folder_id = 1
        self.source_list.clear()
        self._folder_trim_settings.clear()
        self._folder_type_overrides.clear()
        self._per_transition_settings.clear()
        self._removed_files.clear()
        self._direct_transitions.clear()

        # Build resolved-path lookup for folder_data, db_per_trans_settings,
        # and db_removed_files.  Symlink source paths in the DB are resolved,
        # but ordered_folders / per-trans / removed paths may be unresolved
        # (old sessions) or resolved (new sessions).  We try both forms.
        def _resolve_lookup(key: str, mapping: dict) -> str | None:
            """Find key in mapping, trying both raw and resolved forms."""
            if key in mapping:
                return key
            resolved = str(Path(key).resolve())
            if resolved in mapping:
                return resolved
            return None

        # db_per_trans_settings is now list of (trans_folder, left, right, folder_order)
        # Build a lookup by folder_order for new sessions
        db_per_trans_by_order: dict[int, tuple[str, int, int]] = {}
        for trans_folder, left, right, fo in db_per_trans_settings:
            db_per_trans_by_order[fo] = (trans_folder, left, right)

        # db_removed_files is now dict[int, set[str]] keyed by folder_order
        # db_direct_transitions is now list of (after_folder, frame_count, method, enabled, folder_order)
        db_direct_by_order: dict[int, tuple[str, int, str, bool]] = {}
        for after_folder_str, frame_count, method_str, enabled, fo in db_direct_transitions:
            db_direct_by_order[fo] = (after_folder_str, frame_count, method_str, enabled)

        if ordered_folders:
            # New path: ordered_folders has every folder in saved order
            main_idx = 0
            seen_resolved: set[str] = set()
            for position_idx, (folder_str, folder_type, trim_start, trim_end) in enumerate(ordered_folders):
                folder_path = Path(folder_str)
                # Resolve symlinks for consistent path matching
                if not folder_path.exists():
                    continue
                folder_path = folder_path.resolve()
                resolved_str = str(folder_path)
                fid = self._allocate_folder_id()
                self.source_folders.append(folder_path)
                self._folder_ids.append(fid)
                self.source_list.addItem(resolved_str)
                seen_resolved.add(resolved_str)
                if trim_start > 0 or trim_end > 0:
                    self._folder_trim_settings[fid] = (trim_start, trim_end)
                # Always set explicit type override so position-based index%2
                # fallback never silently flips a folder's type after reordering
                # or transition insertion.  AUTO defaults to MAIN (safe for
                # legacy sessions that pre-date the type system).
                if folder_type != FolderType.AUTO:
                    self._folder_type_overrides[fid] = folder_type
                else:
                    self._folder_type_overrides[fid] = FolderType.MAIN
                # Per-transition settings by folder_order
                if position_idx in db_per_trans_by_order:
                    tf, left, right = db_per_trans_by_order[position_idx]
                    self._per_transition_settings[fid] = PerTransitionSettings(
                        trans_folder=folder_path, left_overlap=left, right_overlap=right
                    )
                # Removed files by folder_order
                if position_idx in db_removed_files:
                    self._removed_files[fid] = db_removed_files[position_idx]
                # Direct transitions by folder_order
                if position_idx in db_direct_by_order:
                    af_str, fc, ms, en = db_direct_by_order[position_idx]
                    try:
                        method = DirectInterpolationMethod(ms)
                    except ValueError:
                        method = DirectInterpolationMethod.FILM
                    self._direct_transitions[fid] = DirectTransitionSettings(
                        after_folder=folder_path, frame_count=fc, method=method, enabled=en,
                    )
                # Assign folder_data index for MAIN folders used by _restore_files_from_session
                effective_type = folder_type if folder_type != FolderType.AUTO else FolderType.MAIN
                fd_key = _resolve_lookup(folder_str, folder_data)
                if fd_key is None:
                    fd_key = _resolve_lookup(resolved_str, folder_data)
                if fd_key is not None:
                    if effective_type == FolderType.TRANSITION:
                        effective_type = FolderType.MAIN
                        self._folder_type_overrides[fid] = FolderType.MAIN
                    folder_data[fd_key] = (main_idx, folder_data[fd_key][1])
                    main_idx += 1

            # For old sessions, ordered_folders may be incomplete (only folders
            # with trim/type overrides were saved). Append any symlink-derived
            # folders that weren't already included, in their original order.
            sorted_remaining = sorted(
                [(f, d) for f, d in folder_data.items()
                 if str(Path(f).resolve()) not in seen_resolved],
                key=lambda x: x[1][0],
            )
            for folder_str, (sort_key, file_list) in sorted_remaining:
                folder_path = Path(folder_str)
                if not folder_path.exists():
                    continue
                folder_path = folder_path.resolve()
                resolved_str = str(folder_path)
                if resolved_str in seen_resolved:
                    continue
                seen_resolved.add(resolved_str)
                fid = self._allocate_folder_id()
                self.source_folders.append(folder_path)
                self._folder_ids.append(fid)
                self.source_list.addItem(resolved_str)
                folder_data[folder_str] = (main_idx, file_list)
                main_idx += 1
        else:
            # Legacy path: no ordered_folders, use symlink-derived order
            sorted_folders = sorted(folder_data.items(), key=lambda x: x[1][0])
            for actual_idx, (folder, (sort_key, file_list)) in enumerate(sorted_folders):
                folder_data[folder] = (actual_idx, file_list)

            for folder, (folder_idx, file_list) in sorted(folder_data.items(), key=lambda x: x[1][0]):
                folder_path = Path(folder)
                if folder_path.exists():
                    folder_path = folder_path.resolve()
                    fid = self._allocate_folder_id()
                    self.source_folders.append(folder_path)
                    self._folder_ids.append(fid)
                    self.source_list.addItem(str(folder_path))
                    # Apply trim/type from old DB settings (try both path forms)
                    settings_key = _resolve_lookup(folder, db_folder_settings)
                    if settings_key is None:
                        settings_key = _resolve_lookup(str(folder_path), db_folder_settings)
                    if settings_key is not None:
                        ts, te, ft = db_folder_settings[settings_key]
                        if ts > 0 or te > 0:
                            self._folder_trim_settings[fid] = (ts, te)
                        if ft != FolderType.AUTO:
                            self._folder_type_overrides[fid] = ft
                        else:
                            self._folder_type_overrides[fid] = FolderType.MAIN

        if db_transition_settings:
            self.transition_group.setChecked(db_transition_settings.enabled)
            for i in range(self.curve_combo.count()):
                if self.curve_combo.itemData(i) == db_transition_settings.blend_curve:
                    self.curve_combo.setCurrentIndex(i)
                    break
            for i in range(self.blend_format_combo.count()):
                if self.blend_format_combo.itemData(i) == db_transition_settings.output_format:
                    self.blend_format_combo.setCurrentIndex(i)
                    break
            for i in range(self.blend_method_combo.count()):
                if self.blend_method_combo.itemData(i) == db_transition_settings.blend_method:
                    self.blend_method_combo.setCurrentIndex(i)
                    break
            self.webp_method_spin.setValue(db_transition_settings.webp_method)
            self.blend_quality_spin.setValue(db_transition_settings.output_quality)
            if db_transition_settings.trans_destination:
                self._add_to_path_history(self.trans_dst_path, str(db_transition_settings.trans_destination))
            if db_transition_settings.rife_binary_path:
                self.rife_path_input.setText(str(db_transition_settings.rife_binary_path))
            # OF preset is not restored from session — the widget default (Max) always applies
            self.of_levels_spin.setValue(db_transition_settings.of_levels)
            self.of_winsize_spin.setValue(db_transition_settings.of_winsize)
            self.of_iterations_spin.setValue(db_transition_settings.of_iterations)
            for i in range(self.of_poly_n_combo.count()):
                if self.of_poly_n_combo.itemData(i) == db_transition_settings.of_poly_n:
                    self.of_poly_n_combo.setCurrentIndex(i)
                    break
            self.of_poly_sigma_spin.setValue(db_transition_settings.of_poly_sigma)
            # Update visibility of RIFE path widgets
            self._on_blend_method_changed(self.blend_method_combo.currentIndex())

        # Reconstruct removed files by comparing disk contents vs exported files
        # This recovers edits from sessions before removed_files persistence was added
        if not db_removed_files:
            exported_by_folder: dict[str, set[str]] = {}
            for folder_str, (_, file_list) in folder_data.items():
                exported_by_folder[folder_str] = {fname for _, fname in file_list}

            for i, folder_path in enumerate(self.source_folders):
                fid = self._folder_ids[i]
                folder_str = str(folder_path)
                if folder_str not in exported_by_folder:
                    continue
                exported_names = exported_by_folder[folder_str]
                # Get all supported files on disk for this folder
                disk_files = set()
                from config import SUPPORTED_EXTENSIONS
                for f in sorted(folder_path.iterdir()):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                        disk_files.add(f.name)

                # Apply trim to get the effective file list
                trim_start, trim_end = self._folder_trim_settings.get(fid, (0, 0))
                sorted_disk = sorted(disk_files)
                if trim_start > 0 or trim_end > 0:
                    end_idx = len(sorted_disk) - trim_end
                    trimmed = set(sorted_disk[trim_start:end_idx])
                else:
                    trimmed = disk_files

                # Files on disk (after trim) but not in export = removed
                removed = trimmed - exported_names
                if removed:
                    self._removed_files[fid] = removed

        self._current_session_id = session.id
        self._last_resumed_dest = session.destination

        self._sync_dual_lists()
        # If duplicate paths exist (e.g. split folders), folder_data (keyed by
        # path) can't represent them separately — fall back to _refresh_files
        # which correctly uses per-fid trim settings.
        resolved_paths = [str(f.resolve()) for f in self.source_folders]
        if len(resolved_paths) != len(set(resolved_paths)):
            self._refresh_files()
        else:
            # Restore exact files from session instead of refreshing from disk
            self._restore_files_from_session(folder_data)
        # Ensure _folder_file_counts reflects raw disk counts for ALL folders
        # (_restore_files_from_session only covers MAIN folders in folder_data
        # and stores post-removal counts; TRANSITION folders are missing entirely)
        self._scan_folder_file_counts()
        # Refresh slider now that counts are correct (the earlier call inside
        # _restore_files_from_session used stale post-removal counts)
        self._update_trim_slider_for_selected_folder()
        self._update_flow_arrows()

        total_files = self.file_list.topLevelItemCount()
        trim_count = sum(1 for ts in self._folder_trim_settings.values() if ts[0] > 0 or ts[1] > 0)
        override_count = len(self._folder_type_overrides)
        per_trans_count = len(self._per_transition_settings)
        direct_count = len(self._direct_transitions)
        msg = f"Restored session from {session.created_at.strftime('%Y-%m-%d %H:%M')}.\n"
        msg += f"Loaded {total_files} files from {len(self.source_folders)} folder(s)."
        if trim_count > 0:
            msg += f"\nRestored trim settings for {trim_count} folder(s)."
        if override_count > 0:
            msg += f"\nRestored {override_count} folder type override(s)."
        if per_trans_count > 0:
            msg += f"\nRestored {per_trans_count} per-transition overlap setting(s)."
        if direct_count > 0:
            msg += f"\nRestored {direct_count} direct interpolation setting(s)."
        removed_count = sum(len(v) for v in self._removed_files.values())
        if db_transition_settings and db_transition_settings.enabled:
            msg += f"\nRestored transition settings."
        if removed_count > 0:
            if db_removed_files:
                msg += f"\nRestored {removed_count} removed file(s)."
            else:
                msg += f"\nRecovered {removed_count} file removal(s) from export history."
        if missing_count > 0:
            msg += f"\n{missing_count} file(s) no longer exist and were skipped."

        if not silent:
            QMessageBox.information(self, "Session Restored", msg)
        else:
            pass  # Silent restore (auto-resume on startup)
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
        """Auto-save session state and clean up media player when window closes."""
        self._auto_save_session()
        self.media_player.stop()
        super().closeEvent(event)

    def _auto_save_session(self) -> None:
        """Save current state to the database so it can be restored on next launch.

        Creates or updates a session for the current destination path. This runs
        on close so that folder setup, transition settings, trim, etc. survive
        even if the user never explicitly exported.
        """
        try:
            if not self.source_folders:
                return

            dst = self.dst_path.currentText().strip()
            if not dst:
                return

            dest = str(Path(dst).resolve())

            # Always create a new session so we never overwrite a manual save
            session_id = self.db.create_session(dest, name="autosave")

            # Get current file list before clearing — don't clear if empty
            # to avoid corrupting the session
            files = self._get_files_in_order()
            if not files and self.source_folders:
                # file_list is empty but we have folders — don't overwrite
                # the session, just save settings
                return

            # Clear all stale data for this session before re-saving
            self.db.clear_session_data(session_id)

            self._save_session_settings(session_id, save_effective_types=True)

            # Save the file list so the exact sequence can be restored
            records = []
            for i, (source_dir, filename, folder_idx, file_idx, _fid) in enumerate(files):
                source_path = source_dir / filename
                ext = source_path.suffix
                link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                records.append((
                    str(source_path.resolve()),
                    str(Path(dest) / link_name),
                    filename,
                    i,
                ))
            self.db.record_symlinks_batch(session_id, records)
        except Exception:
            pass  # Best-effort save on close

    def _save_session(self) -> None:
        """Explicitly save the current session state (triggered by Save Session button)."""
        if not self.source_folders:
            QMessageBox.warning(self, "Nothing to Save", "Add at least one source folder first.")
            return

        dst = self.dst_path.currentText().strip()
        if not dst:
            QMessageBox.warning(self, "No Destination", "Set a destination path first.")
            return

        dest = str(Path(dst).resolve())

        try:
            # Always create a fresh session to avoid stale data
            session_id = self.db.create_session(dest)
            self._current_session_id = session_id

            self._save_session_settings(session_id, save_effective_types=True)

            # Save the exact file list in a single transaction
            files = self._get_files_in_order()
            records = []
            for i, (source_dir, filename, folder_idx, file_idx, _fid) in enumerate(files):
                source_path = source_dir / filename
                ext = source_path.suffix
                link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                records.append((
                    str(source_path.resolve()),
                    str(Path(dest) / link_name),
                    filename,
                    i,
                ))
            self.db.record_symlinks_batch(session_id, records)

            main_count = sum(
                1 for i, f in enumerate(self.source_folders)
                if self._get_effective_folder_type(i, f) == FolderType.MAIN
            )
            trans_count = len(self.source_folders) - main_count
            folder_info = f"{main_count} main"
            if trans_count > 0:
                folder_info += f" + {trans_count} transition"
            QMessageBox.information(
                self, "Session Saved",
                f"Saved {len(files)} files from {folder_info} folders."
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save session:\n{e}")

    def _pick_and_restore_session(self) -> None:
        """Show a dialog listing saved sessions and restore the selected one."""
        sessions = self.db.get_sessions()
        if not sessions:
            QMessageBox.information(self, "No Sessions", "No saved sessions found.")
            return

        from PyQt6.QtWidgets import (
            QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
            QTreeWidget, QTreeWidgetItem, QHeaderView, QPushButton,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Restore Session")
        dlg.resize(700, 400)

        layout = QVBoxLayout(dlg)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Date", "Destination", "Files"])
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(tree)

        def populate_tree():
            tree.clear()
            for s in self.db.get_sessions():
                date_str = s.created_at.strftime('%Y-%m-%d %H:%M')
                if s.name:
                    date_str += f"  ({s.name})"
                dest_short = s.destination
                if len(dest_short) > 60:
                    dest_short = '...' + dest_short[-57:]
                item = QTreeWidgetItem([date_str, dest_short, str(s.link_count)])
                item.setData(0, Qt.ItemDataRole.UserRole, s)
                item.setToolTip(1, s.destination)
                tree.addTopLevelItem(item)
            if tree.topLevelItemCount() > 0:
                tree.setCurrentItem(tree.topLevelItem(0))

        def delete_selected():
            selected_items = tree.selectedItems()
            if not selected_items:
                return
            count = len(selected_items)
            label = "session" if count == 1 else "sessions"
            reply = QMessageBox.question(
                dlg, "Delete Sessions",
                f"Delete {count} {label}? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            ids = []
            for item in selected_items:
                s = item.data(0, Qt.ItemDataRole.UserRole)
                if s is not None:
                    ids.append(s.id)
            if ids:
                try:
                    self.db.delete_sessions(ids)
                except Exception as e:
                    QMessageBox.critical(dlg, "Error", f"Failed to delete:\n{e}")
                    return
            populate_tree()

        populate_tree()

        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(delete_selected)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

        # Double-click to accept
        tree.itemDoubleClicked.connect(dlg.accept)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected = tree.currentItem()
        if not selected:
            return

        session = selected.data(0, Qt.ItemDataRole.UserRole)
        if session is None:
            return

        # Set destination path to match the session
        self._add_to_path_history(self.dst_path, session.destination)

        self._restore_session_by_id(session)

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom in image tab."""
        if self.preview_tabs.currentWidget() == self.image_tab:
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
        """Handle dropped folders — insert as TRANSITION when dropped on a placeholder."""
        # Check if the drop lands on a placeholder item in source_list
        insert_index = None
        local_pos = self.source_list.mapFrom(self, event.position().toPoint())
        target_item = self.source_list.itemAt(local_pos)
        if target_item is not None and self._is_placeholder_item(target_item):
            insert_index = self._get_placeholder_insert_index(target_item)

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).is_dir():
                if insert_index is not None:
                    self._add_source_folder(
                        path,
                        folder_type=FolderType.TRANSITION,
                        insert_index=insert_index,
                    )
                    # Only the first dropped folder fills the placeholder slot
                    insert_index = None
                else:
                    self._add_source_folder(path)

    def _on_folders_reordered(self) -> None:
        """Handle folder list reordering."""
        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

    def _show_folder_context_menu(self, pos: QPoint) -> None:
        """Show context menu for folder type and overlap override."""
        item = self.source_list.itemAt(pos)
        if item is None:
            return

        # Placeholder item → offer "Add Transition Folder..."
        if self._is_placeholder_item(item):
            insert_index = self._get_placeholder_insert_index(item)
            if insert_index is not None:
                menu = QMenu(self)
                add_action = menu.addAction("Add Transition Folder...")
                add_action.triggered.connect(
                    lambda: self._add_transition_from_placeholder(insert_index)
                )
                insert_clip_action = menu.addAction("Insert Clip Here...")
                insert_clip_action.triggered.connect(
                    lambda: self._insert_clip_at_position(insert_index)
                )
                menu.exec(self.source_list.mapToGlobal(pos))
            return

        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item_data is None:
            return  # Clicked on video info item

        if not isinstance(item_data, tuple) or len(item_data) != 2:
            return
        folder, fid = item_data

        idx = self._folder_ids.index(fid) if fid in self._folder_ids else -1
        if idx < 0:
            return

        menu = QMenu(self)

        current_type = self._folder_type_overrides.get(fid, FolderType.AUTO)

        auto_action = menu.addAction("Auto (position-based)")
        auto_action.setCheckable(True)
        auto_action.setChecked(current_type == FolderType.AUTO)
        auto_action.triggered.connect(lambda: self._set_folder_type(fid, FolderType.AUTO))

        main_action = menu.addAction("Main [M]")
        main_action.setCheckable(True)
        main_action.setChecked(current_type == FolderType.MAIN)
        main_action.triggered.connect(lambda: self._set_folder_type(fid, FolderType.MAIN))

        trans_action = menu.addAction("Transition [T]")
        trans_action.setCheckable(True)
        trans_action.setChecked(current_type == FolderType.TRANSITION)
        trans_action.triggered.connect(lambda: self._set_folder_type(fid, FolderType.TRANSITION))

        menu.addSeparator()

        # Only show overlap setting for transition folders
        effective_type = self._get_effective_folder_type(idx, folder)
        if effective_type == FolderType.TRANSITION:
            overlap_action = menu.addAction("Set Overlap Frames...")
            overlap_action.triggered.connect(lambda: self._show_overlap_dialog(fid, folder))

        replace_action = menu.addAction("Replace Folder...")
        replace_action.triggered.connect(lambda: self._replace_source_folder(folder, idx))

        menu.addSeparator()

        insert_before_action = menu.addAction("Insert Clip Before...")
        insert_before_action.triggered.connect(lambda: self._insert_clip_at_position(idx))

        # "After" skips past TRANSITION slot if this is a MAIN folder followed by a TRANSITION
        after_idx = idx + 1
        if effective_type == FolderType.MAIN and after_idx < len(self.source_folders):
            next_folder = self.source_folders[after_idx]
            if self._get_effective_folder_type(after_idx, next_folder) == FolderType.TRANSITION:
                after_idx = idx + 2
        insert_after_action = menu.addAction("Insert Clip After...")
        insert_after_action.triggered.connect(lambda: self._insert_clip_at_position(after_idx))

        # Split / Merge for MAIN folders
        if effective_type == FolderType.MAIN:
            menu.addSeparator()

            # Split: only if folder has >1 effective frame
            trimmed = self._get_trimmed_file_list(fid, folder)
            removed = self._removed_files.get(fid, set())
            effective_count = len([f for f in trimmed if f not in removed])
            split_action = menu.addAction("Split Sequence...")
            split_action.triggered.connect(lambda: self._show_split_dialog(fid))
            if effective_count < 2:
                split_action.setEnabled(False)

            # Merge: only if next entry points to same resolved path
            if idx + 1 < len(self.source_folders):
                next_fid = self._folder_ids[idx + 1]
                next_f = self.source_folders[idx + 1]
                if folder.resolve() == next_f.resolve():
                    merge_action = menu.addAction("Merge with Next")
                    merge_action.triggered.connect(lambda: self._merge_adjacent_folders(fid, idx))

        menu.exec(self.source_list.mapToGlobal(pos))

    def _show_file_list_context_menu(self, pos: QPoint) -> None:
        """Show context menu for file list items (split, remove)."""
        item = self.file_list.itemAt(pos)
        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return  # Separator item

        folder, filename, folder_idx, file_idx = data
        fid = item.data(0, Qt.ItemDataRole.UserRole + 2)
        if fid is None:
            return

        menu = QMenu(self)

        # Split action
        split_action = menu.addAction("Split Sequence After This Frame")
        split_action.triggered.connect(lambda: self._split_folder_at_file(fid, filename))

        # Count effective frames for this fid to decide if split is possible
        idx = self._folder_ids.index(fid) if fid in self._folder_ids else -1
        if idx >= 0:
            trimmed = self._get_trimmed_file_list(fid, self.source_folders[idx])
            removed = self._removed_files.get(fid, set())
            effective = [f for f in trimmed if f not in removed]
            if len(effective) <= 1 or filename == effective[-1]:
                split_action.setEnabled(False)

        menu.addSeparator()

        remove_action = menu.addAction("Remove Frame")
        remove_action.triggered.connect(self._remove_selected_files)

        menu.exec(self.file_list.viewport().mapToGlobal(pos))

    def _get_trimmed_file_list(self, fid: int, folder: Path) -> list[str]:
        """Get the file list for a folder after applying trims but before removal filter."""
        from config import SUPPORTED_EXTENSIONS

        if not folder.is_dir():
            return []
        dir_files = sorted(
            [item.name for item in folder.iterdir()
             if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=str.lower
        )
        total = len(dir_files)
        trim_start, trim_end = self._folder_trim_settings.get(fid, (0, 0))
        trim_start = min(trim_start, max(0, total - 1))
        trim_end = min(trim_end, max(0, total - 1 - trim_start))
        end_idx = total - trim_end
        return dir_files[trim_start:end_idx]

    def _split_folder_at_file(self, fid: int, split_after_filename: str) -> None:
        """Split a folder entry into two at the given filename boundary.

        The original entry keeps everything up to and including split_after_filename.
        A new entry is inserted after it with everything after split_after_filename.
        """
        from config import SUPPORTED_EXTENSIONS

        if fid not in self._folder_ids:
            return
        idx = self._folder_ids.index(fid)
        folder = self.source_folders[idx]

        # Get full sorted file list for this physical folder
        if not folder.is_dir():
            return
        all_files = sorted(
            [item.name for item in folder.iterdir()
             if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=str.lower
        )
        total = len(all_files)
        if total < 2:
            return

        # Current trim settings
        old_trim_start, old_trim_end = self._folder_trim_settings.get(fid, (0, 0))
        old_trim_start = min(old_trim_start, max(0, total - 1))
        old_trim_end = min(old_trim_end, max(0, total - 1 - old_trim_start))

        # Find split_after_filename position in the full (untrimmed) file list
        if split_after_filename not in all_files:
            return
        abs_pos = all_files.index(split_after_filename)

        # The split point in absolute indices:
        # Original entry: files[old_trim_start .. abs_pos] → trim_start stays, trim_end = total - abs_pos - 1
        # New entry:      files[abs_pos+1 .. total-1-old_trim_end] → trim_start = abs_pos+1, trim_end = old_trim_end
        new_orig_trim_end = total - abs_pos - 1
        new_entry_trim_start = abs_pos + 1
        new_entry_trim_end = old_trim_end

        # Validate both halves have at least one file
        orig_count = abs_pos - old_trim_start + 1
        new_count = (total - new_entry_trim_end) - new_entry_trim_start
        if orig_count < 1 or new_count < 1:
            return

        # Update original entry trim
        self._folder_trim_settings[fid] = (old_trim_start, new_orig_trim_end)

        # Partition removed files between the two halves
        old_removed = self._removed_files.get(fid, set())
        orig_removed = set()
        new_removed = set()
        for fname in old_removed:
            if fname in all_files:
                fpos = all_files.index(fname)
                if fpos <= abs_pos:
                    orig_removed.add(fname)
                else:
                    new_removed.add(fname)
        if orig_removed:
            self._removed_files[fid] = orig_removed
        elif fid in self._removed_files:
            del self._removed_files[fid]

        # Allocate new fid for the second half
        new_fid = self._allocate_folder_id()

        # Pin effective folder types for shifted positions (same pattern as _add_source_folder)
        insert_at = idx + 1
        for j in range(insert_at, len(self.source_folders)):
            jfid = self._folder_ids[j]
            if jfid not in self._folder_type_overrides:
                self._folder_type_overrides[jfid] = self._get_effective_folder_type(
                    j, self.source_folders[j]
                )

        # Insert new entry
        self.source_folders.insert(insert_at, folder)
        self._folder_ids.insert(insert_at, new_fid)

        # Set new entry settings
        self._folder_trim_settings[new_fid] = (new_entry_trim_start, new_entry_trim_end)
        self._folder_type_overrides[new_fid] = FolderType.MAIN
        if new_removed:
            self._removed_files[new_fid] = new_removed

        # Transfer direct transitions from original to new entry
        # (transitions apply to the end of a sequence, so they belong to the second half)
        if fid in self._direct_transitions:
            self._direct_transitions[new_fid] = self._direct_transitions.pop(fid)

        # Make sure original is also pinned as MAIN
        if fid not in self._folder_type_overrides:
            self._folder_type_overrides[fid] = FolderType.MAIN

        # Refresh UI
        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

    def _show_split_dialog(self, fid: int) -> None:
        """Show a dialog to choose where to split a folder sequence."""
        if fid not in self._folder_ids:
            return
        idx = self._folder_ids.index(fid)
        folder = self.source_folders[idx]

        trimmed = self._get_trimmed_file_list(fid, folder)
        removed = self._removed_files.get(fid, set())
        effective = [f for f in trimmed if f not in removed]

        if len(effective) < 2:
            QMessageBox.information(self, "Split", "Need at least 2 frames to split.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Split Sequence")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(f"Folder: {folder.name}"))
        layout.addWidget(QLabel(f"Frames: {len(effective)}"))

        form = QFormLayout()
        spin = QSpinBox()
        spin.setRange(1, len(effective) - 1)
        spin.setValue(len(effective) // 2)
        spin.setToolTip("First half will contain this many frames")
        form.addRow("Split after frame:", spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            split_idx = spin.value() - 1  # 0-based index into effective list
            split_after_filename = effective[split_idx]
            self._split_folder_at_file(fid, split_after_filename)

    def _merge_adjacent_folders(self, fid: int, idx: int) -> None:
        """Merge two adjacent folder entries that point to the same path.

        Combines the first entry (fid at idx) with the next entry (idx+1),
        undoing a previous split.
        """
        # Re-derive idx from fid in case list changed since menu was shown
        if fid not in self._folder_ids:
            return
        idx = self._folder_ids.index(fid)

        if idx + 1 >= len(self.source_folders):
            return

        next_idx = idx + 1
        next_fid = self._folder_ids[next_idx]
        next_folder = self.source_folders[next_idx]

        # Verify same physical folder
        if self.source_folders[idx].resolve() != next_folder.resolve():
            return

        # Combine trims: keep first's trim_start, second's trim_end
        first_start, _ = self._folder_trim_settings.get(fid, (0, 0))
        _, second_end = self._folder_trim_settings.get(next_fid, (0, 0))
        self._folder_trim_settings[fid] = (first_start, second_end)

        # Union removed files
        first_removed = self._removed_files.get(fid, set())
        second_removed = self._removed_files.get(next_fid, set())
        combined_removed = first_removed | second_removed
        if combined_removed:
            self._removed_files[fid] = combined_removed
        elif fid in self._removed_files:
            del self._removed_files[fid]

        # Transfer direct transitions from second to first
        if next_fid in self._direct_transitions:
            self._direct_transitions[fid] = self._direct_transitions.pop(next_fid)

        # Clean up second entry's settings
        self._folder_trim_settings.pop(next_fid, None)
        self._folder_type_overrides.pop(next_fid, None)
        self._per_transition_settings.pop(next_fid, None)
        self._removed_files.pop(next_fid, None)
        self._direct_transitions.pop(next_fid, None)

        # Remove second entry from lists
        self.source_folders.pop(next_idx)
        self._folder_ids.pop(next_idx)

        # Refresh UI
        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

    def _show_overlap_dialog(self, fid: int, folder: Path) -> None:
        """Show dialog to set per-transition overlap frames."""
        pts = self._per_transition_settings.get(fid)
        left = pts.left_overlap if pts else 16
        right = pts.right_overlap if pts else 16

        dialog = OverlapDialog(self, folder.name, left, right)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_left, new_right = dialog.get_values()
            self._per_transition_settings[fid] = PerTransitionSettings(
                trans_folder=folder,
                left_overlap=new_left,
                right_overlap=new_right
            )
            self._sync_dual_lists()
            self._update_sequence_table()

    def _show_direct_transition_dialog(self, fid_or_folder, folder: Path = None) -> None:
        """Show dialog to configure direct frame interpolation between sequences.

        Args:
            fid_or_folder: Either an int fid or a Path (for backward compat from sequence table clicks).
            folder: Optional folder path when fid is provided as first arg.
        """
        if isinstance(fid_or_folder, int):
            fid = fid_or_folder
            if folder is None:
                idx = self._folder_ids.index(fid) if fid in self._folder_ids else -1
                folder = self.source_folders[idx] if idx >= 0 else Path("unknown")
        else:
            # Called from sequence table with Path — find fid
            folder = fid_or_folder
            # Find first matching fid (legacy path)
            fid = None
            for i, f in enumerate(self.source_folders):
                if f == folder:
                    fid = self._folder_ids[i]
                    break
            if fid is None:
                return

        existing = self._direct_transitions.get(fid)
        if existing:
            frame_count = existing.frame_count
            method = existing.method
            enabled = existing.enabled
        else:
            frame_count = 16
            method = DirectInterpolationMethod.FILM
            enabled = True

        dialog = DirectTransitionDialog(
            self, folder.name, frame_count, method, enabled
        )
        result = dialog.exec()

        if dialog.was_removed():
            # User clicked Remove
            if fid in self._direct_transitions:
                del self._direct_transitions[fid]
            self._update_sequence_table()
        elif result == QDialog.DialogCode.Accepted:
            new_method, new_count, new_enabled = dialog.get_values()
            self._direct_transitions[fid] = DirectTransitionSettings(
                after_folder=folder,
                frame_count=new_count,
                method=new_method,
                enabled=new_enabled
            )
            self._update_sequence_table()

    def _set_folder_type(self, fid: int, folder_type: FolderType) -> None:
        """Set the folder type override for a folder entry."""
        if folder_type == FolderType.AUTO:
            if fid in self._folder_type_overrides:
                del self._folder_type_overrides[fid]
        else:
            self._folder_type_overrides[fid] = folder_type

        self._sync_dual_lists()
        self._update_flow_arrows()

    def _update_folder_type_indicators(self, _=None) -> None:
        """Update folder list item colors and prefixes based on folder types."""
        self._sync_dual_lists()
        self._update_flow_arrows()

    def _add_transition_from_placeholder(self, insert_index: int) -> None:
        """Open file dialog and insert the chosen folder as a TRANSITION at *insert_index*."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Transition Folder", start_dir
        )
        if path:
            self._add_source_folder(
                path, folder_type=FolderType.TRANSITION, insert_index=insert_index
            )

    def _insert_clip_at_position(self, insert_before_idx: int) -> None:
        """Open file dialog and insert the chosen folder as a MAIN clip at *insert_before_idx*."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Clip Folder to Insert", start_dir
        )
        if not path:
            return
        folder = Path(path).resolve()
        if not folder.is_dir():
            return
        # Clear direct transition spanning the insertion gap
        if insert_before_idx > 0 and insert_before_idx <= len(self.source_folders):
            prev_fid = self._folder_ids[insert_before_idx - 1]
            prev = self.source_folders[insert_before_idx - 1]
            if self._get_effective_folder_type(insert_before_idx - 1, prev) == FolderType.MAIN:
                self._direct_transitions.pop(prev_fid, None)
        self._add_source_folder(str(folder), folder_type=FolderType.MAIN, insert_index=insert_before_idx)

    def _on_source_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle click on a source list item — trigger placeholder action if applicable."""
        if self._is_placeholder_item(item):
            insert_index = self._get_placeholder_insert_index(item)
            if insert_index is not None:
                self._add_transition_from_placeholder(insert_index)

    def _get_transition_settings(self) -> TransitionSettings:
        """Get current transition settings from UI."""
        trans_dest = None
        trans_path = self.trans_dst_path.currentText().strip()
        if trans_path:
            trans_dest = Path(trans_path)

        rife_path = None
        rife_path_text = self.rife_path_input.text().strip()
        if rife_path_text:
            rife_path = Path(rife_path_text)

        return TransitionSettings(
            enabled=self.transition_group.isChecked(),
            blend_curve=self.curve_combo.currentData(),
            output_format=self.blend_format_combo.currentData(),
            webp_method=self.webp_method_spin.value(),
            output_quality=self.blend_quality_spin.value(),
            trans_destination=trans_dest,
            blend_method=self.blend_method_combo.currentData(),
            rife_binary_path=rife_path,
            rife_model=self.rife_model_combo.currentData(),
            rife_uhd=self.rife_uhd_check.isChecked(),
            rife_tta=self.rife_tta_check.isChecked(),
            practical_rife_model=self.practical_model_combo.currentData(),
            practical_rife_ensemble=self.practical_ensemble_check.isChecked(),
            of_preset=self.of_preset_combo.currentData(),
            of_levels=self.of_levels_spin.value(),
            of_winsize=self.of_winsize_spin.value(),
            of_iterations=self.of_iterations_spin.value(),
            of_poly_n=self.of_poly_n_combo.currentData(),
            of_poly_sigma=self.of_poly_sigma_spin.value()
        )

    def _scan_folder_file_counts(self) -> None:
        """Scan all source folders and set _folder_file_counts to raw disk counts.

        This ensures the trim slider always shows the true total, regardless
        of whether the file list was populated by _refresh_files (which does
        this automatically) or _restore_files_from_session (which doesn't).
        """
        from config import SUPPORTED_EXTENSIONS
        for i, folder in enumerate(self.source_folders):
            fid = self._folder_ids[i]
            if folder.is_dir():
                count = sum(
                    1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                self._folder_file_counts[fid] = count

    def _refresh_files(self, select_position: str = 'first') -> None:
        """Refresh the file list from all source folders, applying trim settings."""
        from config import SUPPORTED_EXTENSIONS

        self.file_list.clear()
        if not self.source_folders:
            self._folder_file_counts.clear()
            return

        main_folder_indices = self._get_main_folder_indices()
        sub_indices = self._get_main_folder_sub_indices()

        # Scan each folder entry individually (supports duplicate folders)
        files_by_fid: dict[int, list[str]] = {}
        for i, folder in enumerate(self.source_folders):
            fid = self._folder_ids[i]
            if not folder.is_dir():
                continue
            dir_files = sorted(
                [item.name for item in folder.iterdir()
                 if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS],
                key=str.lower
            )
            files_by_fid[fid] = dir_files

        self._folder_file_counts = {fid: len(files) for fid, files in files_by_fid.items()}

        # file_idx counter keyed by base folder_idx so split halves share
        # continuous numbering (seq01_0000..seq01_0005, seq01_0006..seq01_0010)
        base_file_counts: dict[int, int] = {}
        is_first_folder = True
        for i, folder in enumerate(self.source_folders):
            fid = self._folder_ids[i]
            if fid not in files_by_fid:
                continue

            # Skip transition folders — they only participate in blending,
            # not in the main image sequence list.
            if fid not in main_folder_indices:
                continue

            folder_files = files_by_fid[fid]
            total_in_folder = len(folder_files)

            trim_start, trim_end = self._folder_trim_settings.get(fid, (0, 0))
            trim_start = min(trim_start, max(0, total_in_folder - 1))
            trim_end = min(trim_end, max(0, total_in_folder - 1 - trim_start))

            end_idx = total_in_folder - trim_end
            trimmed_files = folder_files[trim_start:end_idx]

            # Filter out individually removed files
            removed = self._removed_files.get(fid, set())
            if removed:
                trimmed_files = [f for f in trimmed_files if f not in removed]

            if not trimmed_files:
                continue

            folder_idx = main_folder_indices[fid]
            sub_idx = sub_indices.get(fid, 0)

            # Add separator between folders (not before first)
            if not is_first_folder:
                separator = self._create_folder_separator(folder_idx, sub_idx)
                self.file_list.addTopLevelItem(separator)
            is_first_folder = False

            for file_i, filename in enumerate(trimmed_files):
                file_idx = base_file_counts.get(folder_idx, 0)
                base_file_counts[folder_idx] = file_idx + 1

                ext = Path(filename).suffix
                if sub_idx > 0:
                    seq_name = f"seq{folder_idx + 1:02d}-{sub_idx}_{file_idx:04d}{ext}"
                else:
                    seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                overall_frame = sum(base_file_counts.values())

                item = QTreeWidgetItem([seq_name, filename, str(folder), str(overall_frame)])
                item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx))
                item.setData(0, Qt.ItemDataRole.UserRole + 2, fid)

                # Bold the last frame of each sequence
                if file_i == len(trimmed_files) - 1:
                    font = item.font(0)
                    font.setBold(True)
                    for col in range(4):
                        item.setFont(col, font)

                self.file_list.addTopLevelItem(item)

        total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, total - 1))
        if total > 0 and select_position != 'none':
            if select_position == 'last':
                self.file_list.setCurrentItem(self.file_list.topLevelItem(total - 1))
            else:
                self.file_list.setCurrentItem(self.file_list.topLevelItem(0))

        self._update_trim_slider_for_selected_folder()
        self._update_sequence_table()
        self._update_export_range_max()

    def _restore_files_from_session(
        self,
        folder_data: dict[str, tuple[int, list[tuple[int, str]]]]
    ) -> None:
        """Restore file list from session data, preserving exact sequence.

        Args:
            folder_data: Dict mapping folder paths to (folder_idx, [(file_idx, filename), ...])
        """
        self.file_list.clear()
        if not folder_data:
            self._folder_file_counts.clear()
            return

        # Sort folders by their index
        sorted_folders = sorted(folder_data.items(), key=lambda x: x[1][0])

        self._folder_file_counts = {}
        is_first_folder = True
        overall_frame = 0

        # Batch UI updates for performance
        self.file_list.setUpdatesEnabled(False)

        for folder_str, (folder_idx, file_list) in sorted_folders:
            folder_path = Path(folder_str)
            if not folder_path.exists():
                continue
            folder_path = folder_path.resolve()

            # Find fid for this folder by matching to source_folders
            fid = 0
            for i, sf in enumerate(self.source_folders):
                if sf == folder_path:
                    fid = self._folder_ids[i]
                    break

            # Sort files by their sequence index
            sorted_files = sorted(file_list, key=lambda x: x[0])

            # Filter out individually removed files
            removed = self._removed_files.get(fid, set())
            if removed:
                sorted_files = [(idx, fname) for idx, fname in sorted_files if fname not in removed]

            if not sorted_files:
                continue

            self._folder_file_counts[fid] = len(sorted_files)

            # Add separator between folders (not before first)
            if not is_first_folder:
                separator = self._create_folder_separator(folder_idx)
                self.file_list.addTopLevelItem(separator)
            is_first_folder = False

            for file_i, (file_idx, filename) in enumerate(sorted_files):
                ext = Path(filename).suffix
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                overall_frame += 1

                item = QTreeWidgetItem([seq_name, filename, str(folder_path), str(overall_frame)])
                item.setData(0, Qt.ItemDataRole.UserRole, (folder_path, filename, folder_idx, file_idx))
                item.setData(0, Qt.ItemDataRole.UserRole + 2, fid)

                # Bold the last frame of each sequence
                if file_i == len(sorted_files) - 1:
                    font = item.font(0)
                    font.setBold(True)
                    for col in range(4):
                        item.setFont(col, font)

                self.file_list.addTopLevelItem(item)

        self.file_list.setUpdatesEnabled(True)

        total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, total - 1))
        if total > 0:
            self.file_list.setCurrentItem(self.file_list.topLevelItem(0))

        self._update_trim_slider_for_selected_folder()
        self._update_sequence_table()
        self._update_export_range_max()

    def _create_folder_separator(self, next_folder_idx: int, sub_idx: int = 0) -> QTreeWidgetItem:
        """Create a visual separator item between folders."""
        if sub_idx > 0:
            label = f"── Sequence {next_folder_idx + 1}-{sub_idx} (continued) ──"
        else:
            label = f"── Sequence {next_folder_idx + 1} ──"
        separator = QTreeWidgetItem(["", label, "", ""])
        separator.setData(0, Qt.ItemDataRole.UserRole, None)  # No data = separator
        # Light grey background
        grey = QColor(220, 220, 220)
        for col in range(4):
            separator.setBackground(col, grey)
        # Make it non-selectable and non-draggable
        separator.setFlags(Qt.ItemFlag.NoItemFlags)
        return separator

    def _is_separator_item(self, item: QTreeWidgetItem) -> bool:
        """Check if an item is a folder separator."""
        return item.data(0, Qt.ItemDataRole.UserRole) is None

    def _get_files_in_order(self) -> list[tuple[Path, str, int, int, int]]:
        """Get files in the current list order with sequence info.

        Returns list of (folder, filename, folder_idx, file_idx, fid) tuples.
        """
        files = []
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                fid = item.data(0, Qt.ItemDataRole.UserRole + 2) or 0
                files.append((*data, fid))
        return files

    def _recalculate_sequence_names(self) -> None:
        """Recalculate sequence names after file reordering."""
        if not self.source_folders:
            return

        main_folder_indices = self._get_main_folder_indices()
        sub_indices = self._get_main_folder_sub_indices()
        # file_idx counter keyed by base folder_idx for continuous numbering
        base_file_counts: dict[int, int] = {}
        last_folder_idx = -1

        # Collect items per fid to detect last file
        fid_items: dict[int, list[int]] = {}
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                item_fid = item.data(0, Qt.ItemDataRole.UserRole + 2) or 0
                if item_fid not in fid_items:
                    fid_items[item_fid] = []
                fid_items[item_fid].append(i)

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                source_dir = data[0]
                filename = data[1]
                item_fid = item.data(0, Qt.ItemDataRole.UserRole + 2) or 0
                folder_idx = main_folder_indices.get(item_fid, 0)
                sub_idx = sub_indices.get(item_fid, 0)
                file_idx = base_file_counts.get(folder_idx, 0)
                base_file_counts[folder_idx] = file_idx + 1

                ext = Path(filename).suffix
                if sub_idx > 0:
                    seq_name = f"seq{folder_idx + 1:02d}-{sub_idx}_{file_idx:04d}{ext}"
                else:
                    seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                overall_frame = sum(base_file_counts.values())
                item.setText(0, seq_name)
                item.setText(3, str(overall_frame))
                item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename, folder_idx, file_idx))

                # Bold the last frame of each sequence
                is_last = (fid_items.get(item_fid, [])[-1] == i)
                font = item.font(0)
                font.setBold(is_last)
                for col in range(4):
                    item.setFont(col, font)

                last_folder_idx = folder_idx
            elif self._is_separator_item(item):
                # Update separator label based on next file's folder
                next_folder_idx = last_folder_idx + 1
                next_sub_idx = 0
                for j in range(i + 1, self.file_list.topLevelItemCount()):
                    next_item = self.file_list.topLevelItem(j)
                    next_data = next_item.data(0, Qt.ItemDataRole.UserRole)
                    if next_data:
                        next_fid = next_item.data(0, Qt.ItemDataRole.UserRole + 2) or 0
                        next_folder_idx = main_folder_indices.get(next_fid, last_folder_idx + 1)
                        next_sub_idx = sub_indices.get(next_fid, 0)
                        break
                if next_sub_idx > 0:
                    item.setText(1, f"── Sequence {next_folder_idx + 1}-{next_sub_idx} (continued) ──")
                else:
                    item.setText(1, f"── Sequence {next_folder_idx + 1} ──")

        # Update the With Transitions tab to reflect the new order
        self._update_sequence_table()
        self._update_export_range_max()

    # --- Video Preview Methods ---

    def _get_videos_in_folder(self, folder: Path) -> list[Path]:
        """Get all video files in the parent folder of the source."""
        videos = []
        parent = folder.parent
        if parent.is_dir():
            for item in parent.iterdir():
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(item)
        return sorted(videos, key=lambda p: p.name.lower())

    def _get_folder_from_item(self, item: QListWidgetItem) -> Optional[Path]:
        """Extract folder path from list item."""
        if item is None:
            return None
        # Placeholder items have no real folder
        if self._is_placeholder_item(item):
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                return data[0]  # (folder, fid) tuple
            return data  # legacy: plain Path
        item_text = item.text()
        if item_text.startswith("[M] ") or item_text.startswith("[T] "):
            return Path(item_text[4:])
        if ". " in item_text:
            return Path(item_text.split(". ", 1)[1])
        return Path(item_text)

    def _get_fid_from_source_item(self, item: QListWidgetItem) -> Optional[int]:
        """Extract folder entry ID from a source list item."""
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and len(data) == 2:
            return data[1]
        return None

    def _get_current_selected_item(self) -> Optional[QListWidgetItem]:
        """Get the currently selected item from the source list."""
        selected = self.source_list.selectedItems()
        if selected:
            return selected[0]
        return None

    def _on_folder_selected(self, current, previous) -> None:
        """Handle folder selection change."""
        self._stop_video()
        self.video_combo.clear()

        if current is None:
            current = self._get_current_selected_item()
            if current is None:
                self.trim_slider.setRange(0)
                self.trim_slider.setEnabled(False)
                self.trim_label.setText("Frames: No folder selected")
                return

        folder = self._get_folder_from_item(current)
        if folder is None:
            # Reset trim slider so stale info doesn't linger on placeholders
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No folder selected")
            return

        self._update_trim_slider_for_selected_folder()

        videos = self._get_videos_in_folder(folder)

        if not videos:
            self.video_combo.addItem("No videos found")
            self.video_combo.setEnabled(False)
            return

        self.video_combo.setEnabled(True)
        for video in videos:
            self.video_combo.addItem(video.name, video)

        self.video_combo.setCurrentIndex(0)

    def _update_trim_slider_for_selected_folder(self) -> None:
        """Update the trim slider to reflect the currently selected folder."""
        current_item = self._get_current_selected_item()
        if current_item is None:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No folder selected")
            return

        folder = self._get_folder_from_item(current_item)
        if folder is None:
            return
        fid = self._get_fid_from_source_item(current_item)
        if fid is None:
            return
        total = self._folder_file_counts.get(fid, 0)

        if total == 0:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No images in folder")
            return

        trim_start, trim_end = self._folder_trim_settings.get(fid, (0, 0))

        self.trim_slider.setEnabled(True)
        self.trim_slider.setRange(total)
        self.trim_slider.setTrimStart(trim_start)
        self.trim_slider.setTrimEnd(trim_end)

        self._update_trim_label(folder, total, trim_start, trim_end)

    def _update_trim_label(self, folder: Path, total: int, trim_start: int, trim_end: int) -> None:
        """Update the trim label to show current trim range."""
        included_start = trim_start + 1
        included_end = total - trim_end
        included_count = included_end - trim_start

        if trim_start == 0 and trim_end == 0:
            self.trim_label.setText(f"Frames: All {total} included")
        elif included_count <= 0:
            self.trim_label.setText(f"Frames: None included (all {total} trimmed)")
        else:
            self.trim_label.setText(f"Frames {included_start}-{included_end} of {total} ({included_count} included)")

    def _on_trim_changed(self, trim_start: int, trim_end: int, handle: str) -> None:
        """Handle trim slider value changes (lightweight, called during drag)."""
        current_item = self._get_current_selected_item()
        if current_item is None:
            return

        folder = self._get_folder_from_item(current_item)
        if folder is None:
            return
        fid = self._get_fid_from_source_item(current_item)
        if fid is None:
            return
        total = self._folder_file_counts.get(fid, 0)

        self._folder_trim_settings[fid] = (trim_start, trim_end)
        self._update_trim_label(folder, total, trim_start, trim_end)

    def _on_trim_drag_finished(self, trim_start: int, trim_end: int, handle: str) -> None:
        """Handle trim drag release (expensive rebuild)."""
        current_item = self._get_current_selected_item()
        if current_item is None:
            return
        fid = self._get_fid_from_source_item(current_item)
        if fid is None:
            return

        self._folder_trim_settings[fid] = (trim_start, trim_end)
        self._refresh_files(select_position='none')
        self._select_folder_boundary(fid, 'first' if handle == 'left' else 'last')

    def _select_folder_boundary(self, fid: int, position: str) -> None:
        """Select the first or last file of a specific folder entry in the file list."""
        matching_indices = []

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            item_fid = item.data(0, Qt.ItemDataRole.UserRole + 2)
            if data and item_fid == fid:
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
            # Play and immediately pause to show first frame
            self.media_player.play()
            self.media_player.pause()

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

        total = self.file_list.topLevelItemCount()
        current_index = self.file_list.indexOfTopLevelItem(current)

        self.image_slider.setRange(0, max(0, total - 1))
        self.image_slider.setValue(current_index)

        self._show_image_at_index(current_index)

        # Sync source list selection so the trim slider shows this frame's folder
        fid = current.data(0, Qt.ItemDataRole.UserRole + 2)
        if fid is not None:
            self._select_folder_in_lists(fid)

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

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Cannot load image:\n{image_path}")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        self._current_pixmap = pixmap
        self._apply_zoom()

        total = self.file_list.topLevelItemCount()
        self.image_index_label.setText(f"{index + 1} / {total}")
        seq_name = item.text(0)
        self.image_name_label.setText(f"{seq_name} ({filename})")

        self.file_list.setCurrentItem(item)

    def _apply_zoom(self) -> None:
        """Apply current zoom level to the image."""
        if self._current_pixmap is None:
            return

        if self._zoom_level == 1.0:
            scaled = self._current_pixmap.scaled(
                self.image_scroll.size() * 0.95,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
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

        self.file_list.takeTopLevelItem(current_index)
        self._recalculate_sequence_names()

        new_total = self.file_list.topLevelItemCount()
        self.image_slider.setRange(0, max(0, new_total - 1))

        if new_total == 0:
            self.image_label.clear()
            self.image_name_label.setText("")
            self.image_index_label.setText("0 / 0")
            self._current_pixmap = None
        else:
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

    def _confirm_overwrite(self, *directories: Path) -> bool:
        """Check if any directory contains seq_* files and ask the user to confirm."""
        existing = []
        for d in directories:
            if d.is_dir():
                count = sum(1 for _ in d.glob("seq_*"))
                if count > 0:
                    existing.append((d, count))
        if not existing:
            return True
        lines = "\n".join(f"  {d}  ({n} files)" for d, n in existing)
        reply = QMessageBox.question(
            self, "Overwrite Existing Files?",
            f"The following directories already contain exported frames "
            f"that will be updated:\n\n{lines}\n\n"
            f"Unchanged files will be kept. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _export_sequence(self) -> None:
        """Export symlinks only (no transitions), with progress bar."""
        dst = self.dst_path.currentText()

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

        dest = Path(dst)
        copy_files = self.copy_files_check.isChecked()

        if not self._confirm_overwrite(dest):
            return

        try:
            self.manager.validate_paths(self.source_folders, dest)
        except SymlinkError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Build planned names for incremental export / orphan removal
        planned_names: set[str] = set()
        for source_dir, filename, folder_idx, file_idx, _fid in files:
            ext = Path(filename).suffix
            planned_names.add(f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}")

        try:
            session_id = self.db.create_session(str(dest))
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to create session: {e}")
            return
        self._current_session_id = session_id

        if session_id:
            self._save_session_settings(session_id, save_effective_types=True)

        total = len(files)
        link_type = "copies" if copy_files else "symlinks"
        progress = QProgressDialog(
            f"Exporting {total} files...", "Cancel", 0, total, self
        )
        progress.setWindowTitle("Export Sequence")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoReset(False)
        progress.setAutoClose(False)
        progress.setValue(0)

        import shutil

        successful = 0
        skipped = 0
        errors = []
        symlink_records = []

        for i, (source_dir, filename, folder_idx, file_idx, _fid) in enumerate(files):
            if progress.wasCanceled():
                break

            source_path = source_dir / filename
            ext = source_path.suffix
            link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
            link_path = dest / link_name

            # Throttle UI updates — label text changes are expensive
            if i % 10 == 0:
                progress.setLabelText(f"Exporting file {i + 1}/{total}: {filename}")
            progress.setValue(i)
            QApplication.processEvents()

            try:
                # Check if existing file already matches — skip if unchanged
                already_correct = False
                if link_path.exists() or link_path.is_symlink():
                    if copy_files:
                        already_correct = SymlinkManager.copy_matches(link_path, source_path)
                    else:
                        already_correct = SymlinkManager.symlink_matches(link_path, source_path)

                if already_correct:
                    skipped += 1
                    symlink_records.append((
                        str(source_path.resolve()),
                        str(link_path),
                        filename,
                        i,
                    ))
                    continue

                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()

                if copy_files:
                    shutil.copy2(source_path, link_path)
                else:
                    rel_source = Path(os.path.relpath(source_path.resolve(), dest.resolve()))
                    link_path.symlink_to(rel_source)

                successful += 1
                symlink_records.append((
                    str(source_path.resolve()),
                    str(link_path),
                    filename,
                    i,
                ))
            except Exception as e:
                errors.append(f"{filename}: {e}")

        # Remove orphan seq*/film_temp_* files not in the planned set
        try:
            SymlinkManager.remove_orphan_files(dest, planned_names)
        except CleanupError:
            pass

        # Batch DB insert — one transaction instead of per-file connections
        if symlink_records:
            try:
                self.db.record_symlinks_batch(session_id, symlink_records)
            except Exception:
                pass  # Don't fail the export over DB recording

        progress.setValue(total)
        progress.close()

        skip_note = f", skipped {skipped} unchanged" if skipped > 0 else ""

        if progress.wasCanceled():
            QMessageBox.warning(
                self, "Canceled",
                f"Export canceled.\n"
                f"Created {successful} {link_type} before cancellation{skip_note}.\n"
                f"Destination: {dst}"
            )
        elif errors:
            QMessageBox.warning(
                self, "Partial Success",
                f"Created {successful} {link_type}, {len(errors)} failed{skip_note}.\n"
                f"First error: {errors[0]}\n"
                f"Destination: {dst}"
            )
        else:
            QMessageBox.information(
                self, "Success",
                f"Created {successful} {link_type}{skip_note} to {dst}"
            )

    def _export_with_transitions(self) -> None:
        """Export with cross-dissolve transitions."""
        dst = self.dst_path.currentText()

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

        # Range is applied inside _process_with_transitions on the output
        # sequence, not on the input file list, because the output includes
        # blended/interpolated frames whose indices differ from input indices.
        export_range = None
        if self.export_options_group.isChecked():
            export_range = (self.range_start_spin.value(), self.range_end_spin.value())

        transition_settings = self._get_transition_settings()

        # Use transition destination if specified, otherwise use main destination
        trans_dst = transition_settings.trans_destination
        if trans_dst is None:
            trans_dst = Path(dst)

        try:
            copy_files = self.copy_files_check.isChecked()
            if len(self.source_folders) >= 2:
                self._process_with_transitions(Path(dst), trans_dst, files, transition_settings, copy_files, export_range)
            else:
                # Fall back to regular export if less than 2 folders
                self._export_sequence()
                return

        except SymlinkError as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))
            return

        # Trigger video encoding if enabled — all output is in trans_dest
        if self.export_options_group.isChecked() and self.video_export_check.isChecked():
            self._encode_output_video(trans_dst)

    def _update_export_range_max(self) -> None:
        """Update export range spinbox maximums based on current file count.

        When transitions are enabled, uses the full sequence frame count
        (including TRANSITION folder middle frames) instead of MAIN-only count.
        """
        if self.transition_group.isChecked() and self._sequence_frame_count > 0:
            total = self._sequence_frame_count
        else:
            total = len(self._get_files_in_order())
        max_val = max(0, total - 1)
        old_end_max = self.range_end_spin.maximum()

        # Temporarily disconnect to avoid cross-constraint issues
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)

        self.range_start_spin.setMaximum(max_val)
        self.range_end_spin.setMaximum(max_val)

        # If end was at old max, snap to new max
        if self.range_end_spin.value() == old_end_max or self.range_end_spin.value() > max_val:
            self.range_end_spin.setValue(max_val)

        # Ensure start <= end
        if self.range_start_spin.value() > self.range_end_spin.value():
            self.range_start_spin.setValue(self.range_end_spin.value())

        # Re-apply cross constraints
        self.range_end_spin.setMinimum(self.range_start_spin.value())
        self.range_start_spin.setMaximum(self.range_end_spin.value())

        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)

    def _reset_export_range(self) -> None:
        """Reset export range to cover all frames."""
        if self.transition_group.isChecked() and self._sequence_frame_count > 0:
            total = self._sequence_frame_count
        else:
            total = len(self._get_files_in_order())
        max_val = max(0, total - 1)

        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)

        self.range_start_spin.setMinimum(0)
        self.range_start_spin.setMaximum(max_val)
        self.range_end_spin.setMinimum(0)
        self.range_end_spin.setMaximum(max_val)

        self.range_start_spin.setValue(0)
        self.range_end_spin.setValue(max_val)

        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)

    def _encode_video_only(self) -> None:
        """Encode video from exported seq_* files, or directly from source images."""
        if not find_ffmpeg():
            QMessageBox.warning(
                self, "ffmpeg Not Found",
                "ffmpeg is not installed or not found in PATH.\n"
                "Install ffmpeg to use video encoding."
            )
            return

        dst = self.dst_path.currentText().strip()
        dst_dir = Path(dst) if dst else None

        # Check transition destination first (Export with Transitions writes there),
        # then fall back to main destination (Export Sequence writes there).
        trans_dst = self.trans_dst_path.currentText().strip()
        trans_dst_dir = Path(trans_dst) if trans_dst else None

        encode_dir = None
        if trans_dst_dir is not None and trans_dst_dir.is_dir() and any(trans_dst_dir.glob("seq_*")):
            encode_dir = trans_dst_dir
        elif dst_dir is not None and dst_dir.is_dir() and any(dst_dir.glob("seq_*")):
            encode_dir = dst_dir

        if encode_dir is not None:
            self._encode_output_video(encode_dir)
        else:
            # Encode directly from the current file list (no prior export needed)
            files = self._get_files_in_order()
            if not files:
                QMessageBox.warning(self, "Error", "No files in the sequence to encode!")
                return

            # Apply export range if set
            if self.export_options_group.isChecked():
                start = self.range_start_spin.value()
                end = self.range_end_spin.value()
                files = files[start:end + 1]

            file_paths = [source_dir / filename for source_dir, filename, *_ in files]

            preset_key = self.video_preset_combo.currentData()
            if preset_key is None:
                return
            preset = VIDEO_PRESETS[preset_key]
            fps = self.fps_spin.value()

            # Ask where to save the video
            default_name = f"output.{preset.container}"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", default_name,
                f"Video (*.{preset.container})"
            )
            if not save_path:
                return

            output_path = Path(save_path)
            total_frames = len(file_paths)

            progress = QProgressDialog("Encoding video...", "Cancel", 0, total_frames, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            cancelled = False

            def on_progress(current: int, total: int) -> bool:
                nonlocal cancelled
                progress.setValue(current)
                QApplication.processEvents()
                if progress.wasCanceled():
                    cancelled = True
                    return False
                return True

            success, message = encode_from_file_list(
                file_paths=file_paths,
                output_path=output_path,
                fps=fps,
                preset=preset,
                progress_callback=on_progress,
            )

            progress.close()

            if success:
                QMessageBox.information(
                    self, "Video Encoded",
                    f"Video saved to:\n{message}"
                )
            elif not cancelled:
                QMessageBox.critical(
                    self, "Encoding Failed",
                    f"Video encoding failed:\n{message}"
                )

    def _encode_output_video(self, output_dir: Path) -> None:
        """Encode the exported image sequence to video using ffmpeg."""
        if not find_ffmpeg():
            QMessageBox.warning(
                self, "ffmpeg Not Found",
                "ffmpeg is not installed or not found in PATH.\n"
                "Install ffmpeg to use video encoding."
            )
            return

        preset_key = self.video_preset_combo.currentData()
        if preset_key is None:
            return
        preset = VIDEO_PRESETS[preset_key]
        fps = self.fps_spin.value()

        # Count seq_* files and detect extension
        seq_files = sorted(output_dir.glob("seq_*"))
        if not seq_files:
            QMessageBox.warning(self, "No Frames", "No seq_* files found in output directory.")
            return
        total_frames = len(seq_files)

        output_path = output_dir / f"output.{preset.container}"

        progress = QProgressDialog("Encoding video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        cancelled = False

        def on_progress(current: int, total: int) -> bool:
            nonlocal cancelled
            progress.setValue(current)
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True
                return False
            return True

        success, message = encode_image_sequence(
            input_dir=output_dir,
            output_path=output_path,
            fps=fps,
            preset=preset,
            progress_callback=on_progress,
            total_frames=total_frames,
        )

        progress.close()

        if success:
            QMessageBox.information(
                self, "Video Encoded",
                f"Video saved to:\n{message}"
            )
        elif not cancelled:
            QMessageBox.critical(
                self, "Encoding Failed",
                f"Video encoding failed:\n{message}"
            )

    def _save_session_settings(self, session_id: int, save_effective_types: bool = False) -> None:
        """Save transition settings and folder type overrides to database.

        Args:
            session_id: The session ID.
            save_effective_types: If True, save the effective folder type for every folder
                (used by "Export with Transitions" to preserve MAIN/TRANSITION assignments).
                If False, only save explicit overrides and trim settings.
        """
        self.db.save_transition_settings(session_id, self._get_transition_settings())

        for folder_idx, folder in enumerate(self.source_folders):
            fid = self._folder_ids[folder_idx]
            trim_start, trim_end = self._folder_trim_settings.get(fid, (0, 0))
            # Always use resolved path so it matches symlink source paths in DB
            resolved_folder = str(folder.resolve())
            if save_effective_types:
                # Save effective type so restore doesn't rely on index-based auto-detection
                effective_type = self._get_effective_folder_type(folder_idx, folder)
                self.db.save_trim_settings(
                    session_id, resolved_folder, trim_start, trim_end, effective_type,
                    folder_order=folder_idx,
                )
            else:
                folder_type = self._folder_type_overrides.get(fid, FolderType.AUTO)
                self.db.save_trim_settings(
                    session_id, resolved_folder, trim_start, trim_end, folder_type,
                    folder_order=folder_idx,
                )

        for fid, pts in self._per_transition_settings.items():
            # Find folder_order (position index) for this fid
            folder_order = self._folder_ids.index(fid) if fid in self._folder_ids else 0
            folder = self.source_folders[folder_order] if folder_order < len(self.source_folders) else pts.trans_folder
            # Use resolved path so it matches folder paths in other tables
            resolved_pts = PerTransitionSettings(
                trans_folder=folder.resolve(),
                left_overlap=pts.left_overlap,
                right_overlap=pts.right_overlap,
            )
            self.db.save_per_transition_settings(session_id, resolved_pts, folder_order=folder_order)

        for fid, removed in self._removed_files.items():
            if removed:
                folder_order = self._folder_ids.index(fid) if fid in self._folder_ids else 0
                folder = self.source_folders[folder_order] if folder_order < len(self.source_folders) else Path("unknown")
                self.db.save_removed_files(session_id, str(folder.resolve()), list(removed), folder_order=folder_order)

        for fid, dt in self._direct_transitions.items():
            folder_order = self._folder_ids.index(fid) if fid in self._folder_ids else 0
            folder = self.source_folders[folder_order] if folder_order < len(self.source_folders) else dt.after_folder
            self.db.save_direct_transition(
                session_id, str(folder.resolve()), dt.frame_count, dt.method.value, dt.enabled,
                folder_order=folder_order,
            )

    def _process_with_transitions(
        self,
        symlink_dest: Path,
        trans_dest: Path,
        files: list[tuple],
        settings: TransitionSettings,
        copy_files: bool = False,
        export_range: Optional[tuple[int, int]] = None,
    ) -> None:
        """Process files with cross-dissolve transitions.

        All output (symlinks/copies AND blended frames) goes to trans_dest so
        that the main destination's Export Sequence files are never touched.

        Args:
            symlink_dest: Main destination (used only for validation/session).
            trans_dest: Where all transition output files are written.
            export_range: Optional (start, end) output frame range. Only output
                frames whose sequence number falls within this range are written.
        """
        self.manager.validate_paths(self.source_folders, symlink_dest)

        # Only write to trans_dest — never touch symlink_dest's seq* files
        trans_dest.mkdir(parents=True, exist_ok=True)
        if not self._confirm_overwrite(trans_dest):
            return

        planned_names: set[str] = set()

        session_id = self.db.create_session(str(symlink_dest))
        self._current_session_id = session_id
        self._save_session_settings(session_id, save_effective_types=True)

        # Build files_by_idx: position index → file list (supports duplicate folders)
        fid_to_pos = {fid: i for i, fid in enumerate(self._folder_ids)}
        files_by_idx: dict[int, list[str]] = {}
        for source_dir, filename, folder_idx, file_idx, fid in files:
            pi = fid_to_pos.get(fid, 0)
            if pi not in files_by_idx:
                files_by_idx[pi] = []
            files_by_idx[pi].append(filename)

        # Include TRANSITION folder files (not in file_list but needed for blending)
        from config import SUPPORTED_EXTENSIONS as _SUP_EXT
        for idx, folder in enumerate(self.source_folders):
            if idx not in files_by_idx:
                ft = self._get_effective_folder_type(idx, folder)
                if ft == FolderType.TRANSITION and folder.is_dir():
                    trans_files = sorted(
                        [item.name for item in folder.iterdir()
                         if item.is_file() and item.suffix.lower() in _SUP_EXT],
                        key=str.lower
                    )
                    # Apply trim settings to transition folders
                    fid = self._folder_ids[idx]
                    ts, te = self._folder_trim_settings.get(fid, (0, 0))
                    if ts > 0 or te > 0:
                        total_t = len(trans_files)
                        ts = min(ts, max(0, total_t - 1))
                        te = min(te, max(0, total_t - 1 - ts))
                        trans_files = trans_files[ts:total_t - te]
                    if trans_files:
                        files_by_idx[idx] = trans_files

        # Build index-keyed overrides and per-transition dicts for blender
        overrides_by_idx: dict[int, FolderType] = {}
        per_trans_by_idx: dict[int, PerTransitionSettings] = {}
        for i in range(len(self.source_folders)):
            fid = self._folder_ids[i]
            if fid in self._folder_type_overrides:
                overrides_by_idx[i] = self._folder_type_overrides[fid]
            if fid in self._per_transition_settings:
                per_trans_by_idx[i] = self._per_transition_settings[fid]

        generator = TransitionGenerator(settings)

        transitions = generator.identify_transition_boundaries(
            self.source_folders,
            files_by_idx,
            overrides_by_idx,
            per_trans_by_idx
        )

        trans_at_main_end: dict[int, TransitionSpec] = {}
        trans_at_trans_start: dict[int, TransitionSpec] = {}
        for trans in transitions:
            trans_at_main_end[trans.main_folder_idx] = trans
            trans_at_trans_start[trans.trans_folder_idx] = trans

        # Build transition boundary summary for the completion dialog
        boundary_notes: list[str] = []
        for trans in transitions:
            main_count = len(trans.main_files)
            trans_count = len(trans.trans_files)
            capped = ""
            if trans.left_overlap < 16 or trans.right_overlap < 16:
                parts = []
                if trans.left_overlap < 16:
                    parts.append(f"{trans.main_folder.name} has {main_count} files")
                if trans.right_overlap < 16:
                    parts.append(f"{trans.trans_folder.name} has {trans_count} files")
                capped = f" (capped: {', '.join(parts)})"
            boundary_notes.append(
                f"  {trans.main_folder.name} -> {trans.trans_folder.name}: "
                f"{trans.left_overlap}/{trans.right_overlap} overlap{capped}"
            )

        # Count total files including direct interpolation frames
        # Include all folders — TRANSITION folders contribute middle (non-overlap) frames
        total_files = sum(len(f) for f in files_by_idx.values())
        for fid, direct_settings in self._direct_transitions.items():
            if direct_settings.enabled:
                total_files += direct_settings.frame_count

        progress = QProgressDialog("Generating sequence...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Cross-Dissolve Generation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoReset(False)
        progress.setAutoClose(False)
        progress.setValue(0)

        import shutil

        current_op = 0
        output_seq = 0
        symlink_count = 0
        skipped = 0
        blend_count = 0
        blend_skipped_range = 0
        errors = []
        symlink_records = []

        num_folders = len(self.source_folders)

        for folder_idx, folder in enumerate(self.source_folders):
            if progress.wasCanceled():
                break

            folder_files = files_by_idx.get(folder_idx, [])
            if not folder_files:
                continue

            folder_label = folder.name
            progress.setLabelText(
                f"Processing folder {folder_idx + 1}/{num_folders}: {folder_label}..."
            )

            num_files = len(folder_files)

            trans_at_end = trans_at_main_end.get(folder_idx)
            trans_at_start = trans_at_trans_start.get(folder_idx)

            for file_idx, filename in enumerate(folder_files):
                if progress.wasCanceled():
                    break

                source_path = folder / filename

                should_blend = False
                should_skip = False
                blend_trans = None
                blend_idx_in_overlap = 0

                if trans_at_end:
                    left_overlap = trans_at_end.left_overlap
                    main_overlap_start = num_files - left_overlap
                    if file_idx >= main_overlap_start:
                        should_blend = True
                        blend_trans = trans_at_end
                        blend_idx_in_overlap = file_idx - main_overlap_start

                if trans_at_start and not should_blend:
                    right_overlap = trans_at_start.right_overlap
                    if file_idx < right_overlap:
                        should_skip = True
                        current_op += 1
                        progress.setValue(current_op)
                        continue

                # Check if this output frame is within the export range
                in_range = (
                    export_range is None
                    or (export_range[0] <= output_seq <= export_range[1])
                )

                if should_blend and blend_trans:
                    if in_range:
                        # Generate asymmetric blend frame — always regenerate
                        output_count = max(blend_trans.left_overlap, blend_trans.right_overlap)

                        # Calculate positions
                        t = blend_idx_in_overlap / (output_count - 1) if output_count > 1 else 0

                        # Get main frame
                        main_path = source_path

                        # Get trans frame position
                        trans_pos = t * (blend_trans.right_overlap - 1) if blend_trans.right_overlap > 1 else 0
                        trans_idx = round(trans_pos)
                        trans_idx = min(trans_idx, blend_trans.right_overlap - 1)
                        trans_file = blend_trans.trans_files[trans_idx]
                        trans_path = blend_trans.trans_folder / trans_file

                        factor = generator.blender.calculate_blend_factor(
                            blend_idx_in_overlap, output_count, settings.blend_curve
                        )

                        ext = f".{settings.output_format.lower()}"
                        output_name = f"seq_{output_seq:05d}{ext}"
                        output_path = trans_dest / output_name
                        planned_names.add(output_name)

                        # Always unlink stale file before regenerating blend
                        if output_path.exists() or output_path.is_symlink():
                            output_path.unlink()

                        result = generator.blender.blend_images(
                            main_path, trans_path, factor,
                            output_path, settings.output_format,
                            settings.output_quality, settings.webp_method,
                            settings.blend_method, settings.rife_binary_path,
                            settings.rife_model, settings.rife_uhd, settings.rife_tta,
                            settings.practical_rife_model, settings.practical_rife_ensemble
                        )

                        if result.success:
                            blend_count += 1
                            symlink_records.append((
                                str(main_path.resolve()),
                                str(output_path), filename, output_seq
                            ))
                        else:
                            errors.append(f"Blend {filename}: {result.error}")
                    else:
                        blend_skipped_range += 1

                    output_seq += 1
                else:
                    if in_range:
                        ext = source_path.suffix
                        link_name = f"seq_{output_seq:05d}{ext}"
                        link_path = trans_dest / link_name
                        planned_names.add(link_name)

                        # Check if existing file already matches — skip if unchanged
                        already_correct = False
                        if link_path.exists() or link_path.is_symlink():
                            if copy_files:
                                already_correct = SymlinkManager.copy_matches(link_path, source_path)
                            else:
                                already_correct = SymlinkManager.symlink_matches(link_path, source_path)

                        if already_correct:
                            skipped += 1
                            symlink_records.append((
                                str(source_path.resolve()),
                                str(link_path), filename, output_seq
                            ))
                        else:
                            try:
                                if link_path.exists() or link_path.is_symlink():
                                    link_path.unlink()

                                if copy_files:
                                    shutil.copy2(source_path, link_path)
                                else:
                                    rel_source = Path(os.path.relpath(source_path.resolve(), trans_dest.resolve()))
                                    link_path.symlink_to(rel_source)
                                symlink_count += 1
                                symlink_records.append((
                                    str(source_path.resolve()),
                                    str(link_path), filename, output_seq
                                ))
                            except Exception as e:
                                errors.append(f"Symlink {filename}: {e}")

                    output_seq += 1

                current_op += 1
                progress.setValue(current_op)
                QApplication.processEvents()

            # Check for direct interpolation after this folder
            fid = self._folder_ids[folder_idx]
            if fid in self._direct_transitions:
                direct_settings = self._direct_transitions[fid]
                if direct_settings.enabled:
                    # Find next folder and get its first frame
                    next_folder_idx = folder_idx + 1
                    if next_folder_idx < len(self.source_folders):
                        next_folder = self.source_folders[next_folder_idx]
                        next_files = files_by_idx.get(next_folder_idx, [])
                        if next_files and folder_files:
                            # Get last frame of current folder and first of next
                            last_frame = folder / folder_files[-1]
                            first_frame = next_folder / next_files[0]
                            batch_end = output_seq + direct_settings.frame_count - 1

                            # Check if any frame in this batch falls within the range
                            batch_in_range = (
                                export_range is None
                                or (output_seq <= export_range[1] and batch_end >= export_range[0])
                            )

                            if batch_in_range:
                                progress.setLabelText(
                                    f"Generating {direct_settings.method.value.upper()} frames..."
                                )

                                # Add planned names and unlink stale files before regeneration
                                out_fmt_ext = f".{settings.output_format.lower()}"
                                for di in range(direct_settings.frame_count):
                                    dname = f"seq_{(output_seq + di):05d}{out_fmt_ext}"
                                    planned_names.add(dname)
                                    dpath = trans_dest / dname
                                    if dpath.exists() or dpath.is_symlink():
                                        dpath.unlink()

                                # Generate direct interpolation frames
                                direct_results = generator.generate_direct_interpolation_frames(
                                    last_frame,
                                    first_frame,
                                    direct_settings.frame_count,
                                    direct_settings.method,
                                    trans_dest,
                                    folder_idx,
                                    output_seq,
                                    settings.practical_rife_model,
                                    settings.practical_rife_ensemble
                                )

                                for result in direct_results:
                                    if result.success:
                                        blend_count += 1
                                        symlink_records.append((
                                            str(result.source_a.resolve()),
                                            str(result.output_path),
                                            result.output_path.name,
                                            output_seq
                                        ))
                                    else:
                                        errors.append(
                                            f"Direct interp {result.output_path.name}: {result.error}"
                                        )
                                    output_seq += 1
                            else:
                                output_seq += direct_settings.frame_count

                            progress.setLabelText(
                                f"Processing folder {folder_idx + 1}/{num_folders}: {folder_label}..."
                            )

        # Remove orphan seq*/film_temp_* files not in the planned set
        try:
            SymlinkManager.remove_orphan_files(trans_dest, planned_names)
        except CleanupError:
            pass

        # Batch DB insert — one transaction instead of per-file connections
        if symlink_records:
            try:
                self.db.record_symlinks_batch(session_id, symlink_records)
            except Exception:
                pass  # Don't fail the export over DB recording

        progress.close()

        link_type = "copies" if copy_files else "symlinks"
        skip_note = f", skipped {skipped} unchanged" if skipped > 0 else ""
        range_note = ""
        if blend_skipped_range > 0:
            range_note = f"\n({blend_skipped_range} blends outside export range)"

        if progress.wasCanceled():
            QMessageBox.warning(
                self, "Canceled",
                f"Operation canceled.\n"
                f"Created {symlink_count} {link_type}, {blend_count} blended frames{skip_note}."
                f"{range_note}"
            )
        elif errors:
            QMessageBox.warning(
                self, "Partial Success",
                f"Created {symlink_count} {link_type}, {blend_count} blended frames{skip_note}.\n"
                f"{len(errors)} errors occurred.\n"
                f"First error: {errors[0] if errors else 'N/A'}\n"
                f"Output: {trans_dest}"
                f"{range_note}"
            )
        else:
            QMessageBox.information(
                self, "Success",
                f"Created {symlink_count} {link_type} and {blend_count} blended frames{skip_note}.\n"
                f"Output: {trans_dest}"
                f"{range_note}"
            )
