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
    TransitionSpec,
    SymlinkError,
    DatabaseManager,
    TransitionGenerator,
    RifeDownloader,
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

                        # Draw time label on right of column 1 (right edge)
                        painter.drawText(viewport_width - text_width - 6, y_center + metrics.ascent() // 2, time_str)

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
        self.left_spin.setToolTip("Frames consumed from the end of the Main folder")
        form_layout.addRow("Left overlap (Main end):", self.left_spin)

        self.right_spin = QSpinBox()
        self.right_spin.setRange(1, 120)
        self.right_spin.setValue(right_overlap)
        self.right_spin.setToolTip("Frames consumed from the start of the Transition folder")
        form_layout.addRow("Right overlap (Trans start):", self.right_spin)

        layout.addLayout(form_layout)

        # Explanation
        explain = QLabel(
            "Left overlap: frames from Main folder end that are blended.\n"
            "Right overlap: frames from Transition folder start that are blended.\n"
            "Output frames = max(left, right). Asymmetric values interpolate."
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
        self.last_directory: Optional[str] = None
        self._last_resumed_dest: Optional[str] = None
        self._folder_trim_settings: dict[Path, tuple[int, int]] = {}
        self._folder_file_counts: dict[Path, int] = {}
        self._folder_type_overrides: dict[Path, FolderType] = {}
        self._transition_settings = TransitionSettings()
        self._per_transition_settings: dict[Path, PerTransitionSettings] = {}
        self._direct_transitions: dict[Path, DirectTransitionSettings] = {}
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

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle('Video Montage Linker')
        self.setMinimumSize(1000, 700)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Source folders panel - side panel with single unified list
        self.source_panel = QWidget()
        self.source_panel.setMinimumWidth(250)
        self.source_panel.setMaximumWidth(400)

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
        self.file_list.setHeaderLabels(["Sequence Name", "Original Filename", "Source Folder"])
        self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setRootIsDecorated(False)
        self.file_list.header().setStretchLastSection(True)
        self.file_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
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
        self.sequence_table.setHeaderLabels(["Main Frame", "Transition Frame"])
        self.sequence_table.setColumnCount(2)
        self.sequence_table.setRootIsDecorated(False)
        self.sequence_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sequence_table.setAlternatingRowColors(True)
        self.sequence_table.header().setStretchLastSection(True)
        self.sequence_table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sequence_table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

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
        self.of_preset_combo.setCurrentIndex(1)  # Default to Balanced
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

        # File list action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.remove_files_btn)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()

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
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.addWidget(file_list_panel)
        self.content_splitter.addWidget(self.preview_tabs)
        self.content_splitter.setSizes([350, 450])

        # Export buttons layout
        export_layout = QHBoxLayout()
        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.export_trans_btn)

        # Assemble right panel
        right_layout.addLayout(dst_layout)
        right_layout.addLayout(trans_dst_layout)
        right_layout.addWidget(self.transition_group)
        right_layout.addWidget(self.content_splitter, 1)
        right_layout.addLayout(export_layout)

        # === MAIN SPLITTER: Source Panel | Main Content ===
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.source_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([250, 750])
        self.main_splitter.setStretchFactor(0, 0)  # Source panel doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Main content stretches

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
        self.export_btn.clicked.connect(self._export_sequence)
        self.export_trans_btn.clicked.connect(self._export_with_transitions)

        # Connect reorder signals
        self.file_list.model().rowsMoved.connect(self._recalculate_sequence_names)

        # Context menu for folder type and overlap override
        self.source_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.source_list.customContextMenuRequested.connect(self._show_folder_context_menu)

        # Update folder indicators when transition setting changes
        self.transition_group.toggled.connect(self._update_folder_type_indicators)

        # Connect folder selection to update video list
        self.source_list.currentItemChanged.connect(self._on_folder_selected)

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
        self.sequence_table.clear()

        if not self.source_folders:
            self._update_timeline_display()
            return

        files = self._get_files_in_order()
        if not files:
            self._update_timeline_display()
            return

        # Group files by folder
        files_by_folder: dict[Path, list[str]] = {}
        for source_dir, filename, folder_idx, file_idx in files:
            if source_dir not in files_by_folder:
                files_by_folder[source_dir] = []
            files_by_folder[source_dir].append(filename)

        # Check if transitions are enabled
        if not self.transition_group.isChecked():
            # Just show symlinks in Main column only
            for source_dir, filename, folder_idx, file_idx in files:
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}"
                item = QTreeWidgetItem([f"{seq_name} ({filename})", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename, folder_idx, file_idx, 'symlink'))
                self.sequence_table.addTopLevelItem(item)
            self._update_timeline_display()
            return

        # Get transition specs
        settings = self._get_transition_settings()
        generator = TransitionGenerator(settings)
        transitions = generator.identify_transition_boundaries(
            self.source_folders,
            files_by_folder,
            self._folder_type_overrides,
            self._per_transition_settings
        )

        # Build lookup for transitions
        trans_at_main_end: dict[Path, TransitionSpec] = {}
        trans_at_trans_start: dict[Path, TransitionSpec] = {}
        for trans in transitions:
            trans_at_main_end[trans.main_folder] = trans
            trans_at_trans_start[trans.trans_folder] = trans

        # Find consecutive MAIN folders (for direct interpolation)
        consecutive_main_pairs: list[tuple[int, int]] = []
        for i in range(len(self.source_folders) - 1):
            folder_a = self.source_folders[i]
            folder_b = self.source_folders[i + 1]
            type_a = self._get_effective_folder_type(i, folder_a)
            type_b = self._get_effective_folder_type(i + 1, folder_b)
            # Two consecutive MAIN folders with no transition between them
            if type_a == FolderType.MAIN and type_b == FolderType.MAIN:
                if folder_a not in trans_at_main_end:
                    consecutive_main_pairs.append((i, i + 1))

        # Process each folder
        for folder_idx, folder in enumerate(self.source_folders):
            folder_files = files_by_folder.get(folder, [])
            if not folder_files:
                continue

            num_files = len(folder_files)
            trans_at_end = trans_at_main_end.get(folder)
            trans_at_start = trans_at_trans_start.get(folder)
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
                if trans_at_start:
                    right_overlap = trans_at_start.right_overlap
                    if file_idx < right_overlap:
                        # These frames are consumed by the blend - skip them
                        continue

                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}"

                if should_blend and blend_trans:
                    # Calculate which trans frame this blends with
                    output_count = max(blend_trans.left_overlap, blend_trans.right_overlap)
                    t = blend_idx_in_overlap / (output_count - 1) if output_count > 1 else 0
                    trans_pos = t * (blend_trans.right_overlap - 1) if blend_trans.right_overlap > 1 else 0
                    trans_idx = min(int(trans_pos), blend_trans.right_overlap - 1)
                    trans_file = blend_trans.trans_files[trans_idx]

                    # Outgoing frame with [B] marker, incoming frame with arrow
                    main_text = f"[B] {seq_name} ({filename})"
                    trans_text = f"→ {trans_file}"

                    item = QTreeWidgetItem([main_text, trans_text])
                    item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'blend'))
                    item.setData(1, Qt.ItemDataRole.UserRole, (blend_trans.trans_folder, trans_file))
                    # Blue color for blend frames
                    item.setForeground(0, QColor(100, 150, 255))
                    item.setForeground(1, QColor(100, 150, 255))
                elif folder_type == FolderType.TRANSITION:
                    # Transition folder files go in Transition column only
                    item = QTreeWidgetItem(["", f"{seq_name} ({filename})"])
                    item.setData(1, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'symlink'))
                else:
                    # Main folder files go in Main column only
                    item = QTreeWidgetItem([f"{seq_name} ({filename})", ""])
                    item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx, 'symlink'))

                self.sequence_table.addTopLevelItem(item)

            # Check if this folder starts a direct interpolation gap
            # (current MAIN followed by another MAIN with no transition)
            for pair_idx_a, pair_idx_b in consecutive_main_pairs:
                if folder_idx == pair_idx_a:
                    # Add direct interpolation row after this folder's files
                    self._add_direct_interpolation_row(folder, pair_idx_b)

        # Update timeline display after rebuilding sequence table
        self._update_timeline_display()

    def _add_direct_interpolation_row(self, after_folder: Path, next_folder_idx: int) -> None:
        """Add a clickable direct interpolation row between MAIN sequences.

        Args:
            after_folder: The folder after which interpolation occurs.
            next_folder_idx: Index of the next MAIN folder.
        """
        direct_settings = self._direct_transitions.get(after_folder)

        if direct_settings and direct_settings.enabled:
            # Configured: show green row with settings + placeholder frames
            method_name = direct_settings.method.value.upper()
            frame_count = direct_settings.frame_count

            # Header row (clickable to edit)
            header_text = f"  [{method_name}: {frame_count} frames] (click to edit)"
            header_item = QTreeWidgetItem([header_text, ""])
            header_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_header', after_folder))
            header_item.setForeground(0, QColor(50, 180, 100))  # Green
            header_item.setFlags(header_item.flags() & ~Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.sequence_table.addTopLevelItem(header_item)

            # Add placeholder rows for each interpolated frame
            for i in range(frame_count):
                placeholder_text = f"    [{method_name} {i + 1}/{frame_count}]"
                placeholder_item = QTreeWidgetItem([placeholder_text, ""])
                placeholder_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_placeholder', after_folder, i))
                placeholder_item.setForeground(0, QColor(100, 180, 220))  # Light blue
                # Make placeholders non-selectable
                placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.sequence_table.addTopLevelItem(placeholder_item)
        else:
            # Unconfigured: show grey "+" row
            add_text = "  [+ Add RIFE/FILM transition] (click to configure)"
            add_item = QTreeWidgetItem([add_text, ""])
            add_item.setData(0, Qt.ItemDataRole.UserRole, ('direct_add', after_folder))
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
                after_folder = data[1]
                frame_index = data[2]
                self._show_direct_interpolation_preview(after_folder, frame_index)
                return

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
                after_folder = data[1]
                frame_index = data[2]
                self._show_direct_interpolation_preview(after_folder, frame_index)

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
            seq_name = f"seq{data0[2] + 1:02d}_{data0[3]:04d}"
            self.image_name_label.setText(f"[B] {seq_name} ({main_file} + {trans_file}) @ {factor:.0%}")

        except Exception as e:
            self.image_label.setText(f"Error generating blend preview:\n{e}")
            self.image_name_label.setText("")
            self._current_pixmap = None

    def _show_direct_interpolation_preview(self, after_folder: Path, frame_index: int) -> None:
        """Generate and show a preview for a direct interpolation placeholder frame.

        For RIFE: Generates one frame at a time (RIFE handles arbitrary timesteps well).
        For FILM: Generates ALL frames at once on first click (FILM works best this way),
                  then caches all frames for instant subsequent access.

        Args:
            after_folder: The folder after which the interpolation occurs.
            frame_index: The index of the interpolated frame (0-based).
        """
        from PIL import Image
        from PIL.ImageQt import ImageQt
        from core import ImageBlender

        # Get direct transition settings
        direct_settings = self._direct_transitions.get(after_folder)
        if not direct_settings or not direct_settings.enabled:
            self.image_label.setText("Direct interpolation not configured")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        # Find the folder index and next folder
        try:
            folder_idx = self.source_folders.index(after_folder)
        except ValueError:
            self.image_label.setText("Folder not found in sequence")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        if folder_idx >= len(self.source_folders) - 1:
            self.image_label.setText("No next folder for interpolation")
            self.image_name_label.setText("")
            self._current_pixmap = None
            return

        next_folder = self.source_folders[folder_idx + 1]

        # Get files for both folders
        files = self._get_files_in_order()
        files_by_folder: dict[Path, list[str]] = {}
        for source_dir, filename, f_idx, file_idx in files:
            if source_dir not in files_by_folder:
                files_by_folder[source_dir] = []
            files_by_folder[source_dir].append(filename)

        after_files = files_by_folder.get(after_folder, [])
        next_files = files_by_folder.get(next_folder, [])

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
        cache_key = f"direct|{after_folder}|{frame_index}|{direct_settings.method.value}|{frame_count}"

        try:
            # Check cache first
            if cache_key in self._blend_preview_cache:
                pixmap = self._blend_preview_cache[cache_key]
            elif direct_settings.method == DirectInterpolationMethod.FILM and FilmEnv.is_setup():
                # FILM: Generate ALL frames at once for better quality
                # Check if we need to generate (first frame not cached means none are)
                first_cache_key = f"direct|{after_folder}|0|{direct_settings.method.value}|{frame_count}"
                if first_cache_key not in self._blend_preview_cache:
                    # Generate all frames at once
                    error_msg = self._generate_all_film_preview_frames(
                        after_folder, last_frame_path, first_frame_path, frame_count
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
                    item_data[1] == after_folder and
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
        after_folder: Path,
        last_frame_path: Path,
        first_frame_path: Path,
        frame_count: int
    ) -> Optional[str]:
        """Generate all FILM preview frames at once and cache them.

        FILM works best when generating all frames at once using its
        recursive approach. This method generates all frames and stores
        them in the preview cache.

        Args:
            after_folder: The folder after which the interpolation occurs.
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
                        cache_key = f"direct|{after_folder}|{i}|film|{frame_count}"
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
        folder_type: Optional[FolderType] = None
    ) -> None:
        """Add a source folder via file dialog or direct path."""
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
            folder = Path(path)
            if folder.is_dir() and folder not in self.source_folders:
                self.source_folders.append(folder)
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

        for i, folder in enumerate(self.source_folders):
            folder_type = self._get_effective_folder_type(i, folder)

            # Get per-transition settings if available
            pts = self._per_transition_settings.get(folder)
            overlap_text = ""
            if pts and folder_type == FolderType.TRANSITION:
                overlap_text = f" [L:{pts.left_overlap} R:{pts.right_overlap}]"

            # Type indicator
            type_tag = "[M]" if folder_type == FolderType.MAIN else "[T]"

            # Compress path if common prefix exists
            if common_prefix:
                display_path = "[...]" + str(folder)[len(common_prefix):]
            else:
                display_path = str(folder)

            # Show path with index and type
            display_name = f"{i+1}. {type_tag} {display_path}{overlap_text}"
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            item.setToolTip(str(folder))  # Full path on hover

            # Color and alignment: Main = left/default, Transition = right/purple
            if folder_type == FolderType.TRANSITION:
                item.setForeground(QColor(155, 89, 182))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self.source_list.addItem(item)

    def _get_effective_folder_type(self, index: int, folder: Path) -> FolderType:
        """Get the effective folder type considering overrides."""
        if folder in self._folder_type_overrides:
            override = self._folder_type_overrides[folder]
            if override != FolderType.AUTO:
                return override
        return FolderType.MAIN if index % 2 == 0 else FolderType.TRANSITION

    def _update_flow_arrows(self) -> None:
        """Update visual indicators."""
        pass

    def _get_selected_folder(self) -> Optional[tuple[Path, int]]:
        """Get the currently selected folder and its index."""
        selected = self.source_list.selectedItems()
        if selected:
            folder = selected[0].data(Qt.ItemDataRole.UserRole)
            if folder is not None and folder in self.source_folders:
                return folder, self.source_folders.index(folder)
        return None

    def _move_folder_up(self) -> None:
        """Move the selected folder up in the sequence."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result
        if idx > 0:
            self.source_folders[idx], self.source_folders[idx - 1] = \
                self.source_folders[idx - 1], self.source_folders[idx]
            self._sync_dual_lists()
            self._refresh_files()
            self._update_flow_arrows()
            self._select_folder_in_lists(folder)

    def _move_folder_down(self) -> None:
        """Move the selected folder down in the sequence."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result
        if idx < len(self.source_folders) - 1:
            self.source_folders[idx], self.source_folders[idx + 1] = \
                self.source_folders[idx + 1], self.source_folders[idx]
            self._sync_dual_lists()
            self._refresh_files()
            self._update_flow_arrows()
            self._select_folder_in_lists(folder)

    def _select_folder_in_lists(self, folder: Path) -> None:
        """Select a folder in the source list widget."""
        for i in range(self.source_list.count()):
            item = self.source_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == folder:
                self.source_list.setCurrentItem(item)
                return

    def _remove_source_folder(self) -> None:
        """Remove selected source folder(s), preserving sequence order of remaining files."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result

        if folder in self._folder_type_overrides:
            del self._folder_type_overrides[folder]
        if folder in self._per_transition_settings:
            del self._per_transition_settings[folder]
        if folder in self._folder_trim_settings:
            del self._folder_trim_settings[folder]
        if folder in self._folder_file_counts:
            del self._folder_file_counts[folder]

        del self.source_folders[idx]

        self._sync_dual_lists()

        # Remove only files from the deleted folder, preserving order of others
        self._remove_files_from_folder(folder)

        # Renumber sequence names to reflect new folder indices
        self._recalculate_sequence_names()

        # Update the sequence table (With Transitions tab)
        self._update_sequence_table()

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
        """Try to resume a previous session for the given destination."""
        dest = Path(dest_path).resolve()
        dest_str = str(dest)

        self._last_resumed_dest = dest_str

        sessions = self.db.get_sessions_by_destination(dest_str)

        if not sessions:
            return False

        latest_session = sessions[0]
        symlinks = self.db.get_symlinks_by_session(latest_session.id)

        if not symlinks:
            return False

        db_trim_settings = self.db.get_all_trim_settings(latest_session.id)
        db_folder_types = self.db.get_folder_type_overrides(latest_session.id)
        db_transition_settings = self.db.get_transition_settings(latest_session.id)
        db_per_trans_settings = self.db.get_all_per_transition_settings(latest_session.id)

        new_pattern = re.compile(r'seq(\d+)_(\d+)')
        old_pattern = re.compile(r'seq_(\d+)')

        folder_data: dict[str, tuple[int, list[tuple[int, str]]]] = {}
        missing_count = 0

        for link in symlinks:
            source_path = Path(link.source_path)
            if not source_path.exists():
                missing_count += 1
                continue

            folder = str(source_path.parent)
            link_name = Path(link.link_path).stem

            match = new_pattern.match(link_name)
            if match:
                folder_idx = int(match.group(1)) - 1
                file_idx = int(match.group(2))
            else:
                match = old_pattern.match(link_name)
                if match:
                    folder_idx = 0
                    file_idx = int(match.group(1))
                else:
                    folder_idx = 0
                    file_idx = link.sequence_number

            if folder not in folder_data:
                folder_data[folder] = (folder_idx, [])
            folder_data[folder][1].append((file_idx, link.original_filename))

        if not folder_data:
            return False

        sorted_folders = sorted(folder_data.items(), key=lambda x: x[1][0])

        self.source_folders.clear()
        self.source_list.clear()
        self._folder_trim_settings.clear()
        self._folder_type_overrides.clear()
        self._per_transition_settings.clear()

        for folder, (folder_idx, file_list) in sorted_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                self.source_folders.append(folder_path)
                self.source_list.addItem(folder)
                if folder in db_trim_settings:
                    self._folder_trim_settings[folder_path] = db_trim_settings[folder]
                if folder in db_folder_types and db_folder_types[folder] != FolderType.AUTO:
                    self._folder_type_overrides[folder_path] = db_folder_types[folder]
                if folder in db_per_trans_settings:
                    self._per_transition_settings[folder_path] = db_per_trans_settings[folder]

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
            # Restore optical flow settings
            for i in range(self.of_preset_combo.count()):
                if self.of_preset_combo.itemData(i) == db_transition_settings.of_preset:
                    self.of_preset_combo.setCurrentIndex(i)
                    break
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

        self._current_session_id = latest_session.id

        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

        total_files = self.file_list.topLevelItemCount()
        trim_count = sum(1 for ts in self._folder_trim_settings.values() if ts[0] > 0 or ts[1] > 0)
        override_count = len(self._folder_type_overrides)
        per_trans_count = len(self._per_transition_settings)
        msg = f"Resumed session from {latest_session.created_at.strftime('%Y-%m-%d %H:%M')}.\n"
        msg += f"Loaded {total_files} files from {len(self.source_folders)} folder(s)."
        if trim_count > 0:
            msg += f"\nRestored trim settings for {trim_count} folder(s)."
        if override_count > 0:
            msg += f"\nRestored {override_count} folder type override(s)."
        if per_trans_count > 0:
            msg += f"\nRestored {per_trans_count} per-transition overlap setting(s)."
        if db_transition_settings and db_transition_settings.enabled:
            msg += f"\nRestored transition settings."
        if missing_count > 0:
            msg += f"\n{missing_count} file(s) no longer exist and were skipped."

        QMessageBox.information(self, "Session Resumed", msg)
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
        """Clean up media player when window closes."""
        self.media_player.stop()
        super().closeEvent(event)

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
        """Handle dropped folders."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).is_dir():
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

        folder = item.data(Qt.ItemDataRole.UserRole)
        if folder is None:
            return  # Clicked on video info item

        idx = self.source_folders.index(folder) if folder in self.source_folders else -1
        if idx < 0:
            return

        menu = QMenu(self)

        current_type = self._folder_type_overrides.get(folder, FolderType.AUTO)

        auto_action = menu.addAction("Auto (position-based)")
        auto_action.setCheckable(True)
        auto_action.setChecked(current_type == FolderType.AUTO)
        auto_action.triggered.connect(lambda: self._set_folder_type(folder, FolderType.AUTO))

        main_action = menu.addAction("Main [M]")
        main_action.setCheckable(True)
        main_action.setChecked(current_type == FolderType.MAIN)
        main_action.triggered.connect(lambda: self._set_folder_type(folder, FolderType.MAIN))

        trans_action = menu.addAction("Transition [T]")
        trans_action.setCheckable(True)
        trans_action.setChecked(current_type == FolderType.TRANSITION)
        trans_action.triggered.connect(lambda: self._set_folder_type(folder, FolderType.TRANSITION))

        menu.addSeparator()

        # Only show overlap setting for transition folders
        effective_type = self._get_effective_folder_type(idx, folder)
        if effective_type == FolderType.TRANSITION:
            overlap_action = menu.addAction("Set Overlap Frames...")
            overlap_action.triggered.connect(lambda: self._show_overlap_dialog(folder))

        menu.exec(self.source_list.mapToGlobal(pos))

    def _show_overlap_dialog(self, folder: Path) -> None:
        """Show dialog to set per-transition overlap frames."""
        pts = self._per_transition_settings.get(folder)
        left = pts.left_overlap if pts else 16
        right = pts.right_overlap if pts else 16

        dialog = OverlapDialog(self, folder.name, left, right)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_left, new_right = dialog.get_values()
            self._per_transition_settings[folder] = PerTransitionSettings(
                trans_folder=folder,
                left_overlap=new_left,
                right_overlap=new_right
            )
            self._sync_dual_lists()
            self._update_sequence_table()

    def _show_direct_transition_dialog(self, after_folder: Path) -> None:
        """Show dialog to configure direct frame interpolation between sequences."""
        existing = self._direct_transitions.get(after_folder)
        if existing:
            frame_count = existing.frame_count
            method = existing.method
            enabled = existing.enabled
        else:
            frame_count = 16
            method = DirectInterpolationMethod.FILM
            enabled = True

        dialog = DirectTransitionDialog(
            self, after_folder.name, frame_count, method, enabled
        )
        result = dialog.exec()

        if dialog.was_removed():
            # User clicked Remove
            if after_folder in self._direct_transitions:
                del self._direct_transitions[after_folder]
            self._update_sequence_table()
        elif result == QDialog.DialogCode.Accepted:
            new_method, new_count, new_enabled = dialog.get_values()
            self._direct_transitions[after_folder] = DirectTransitionSettings(
                after_folder=after_folder,
                frame_count=new_count,
                method=new_method,
                enabled=new_enabled
            )
            self._update_sequence_table()

    def _set_folder_type(self, folder: Path, folder_type: FolderType) -> None:
        """Set the folder type override for a folder."""
        if folder_type == FolderType.AUTO:
            if folder in self._folder_type_overrides:
                del self._folder_type_overrides[folder]
        else:
            self._folder_type_overrides[folder] = folder_type

        self._sync_dual_lists()
        self._update_flow_arrows()

    def _update_folder_type_indicators(self, _=None) -> None:
        """Update folder list item colors and prefixes based on folder types."""
        self._sync_dual_lists()
        self._update_flow_arrows()

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

    def _refresh_files(self, select_position: str = 'first') -> None:
        """Refresh the file list from all source folders, applying trim settings."""
        self.file_list.clear()
        if not self.source_folders:
            self._folder_file_counts.clear()
            return

        folder_to_index = {folder: i for i, folder in enumerate(self.source_folders)}
        all_files = self.manager.get_supported_files(self.source_folders)

        files_by_folder: dict[Path, list[str]] = {}
        for source_dir, filename in all_files:
            if source_dir not in files_by_folder:
                files_by_folder[source_dir] = []
            files_by_folder[source_dir].append(filename)

        self._folder_file_counts = {folder: len(files) for folder, files in files_by_folder.items()}

        folder_file_counts: dict[Path, int] = {}
        is_first_folder = True
        for folder in self.source_folders:
            if folder not in files_by_folder:
                continue

            folder_files = files_by_folder[folder]
            total_in_folder = len(folder_files)

            trim_start, trim_end = self._folder_trim_settings.get(folder, (0, 0))
            trim_start = min(trim_start, max(0, total_in_folder - 1))
            trim_end = min(trim_end, max(0, total_in_folder - 1 - trim_start))

            end_idx = total_in_folder - trim_end
            trimmed_files = folder_files[trim_start:end_idx]

            if not trimmed_files:
                continue

            folder_idx = folder_to_index.get(folder, 0)

            # Add separator between folders (not before first)
            if not is_first_folder:
                separator = self._create_folder_separator(folder_idx)
                self.file_list.addTopLevelItem(separator)
            is_first_folder = False

            for filename in trimmed_files:
                file_idx = folder_file_counts.get(folder, 0)
                folder_file_counts[folder] = file_idx + 1

                ext = Path(filename).suffix
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"

                item = QTreeWidgetItem([seq_name, filename, str(folder)])
                item.setData(0, Qt.ItemDataRole.UserRole, (folder, filename, folder_idx, file_idx))
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

    def _create_folder_separator(self, next_folder_idx: int) -> QTreeWidgetItem:
        """Create a visual separator item between folders."""
        separator = QTreeWidgetItem(["", f"── Sequence {next_folder_idx + 1} ──", ""])
        separator.setData(0, Qt.ItemDataRole.UserRole, None)  # No data = separator
        # Light grey background
        grey = QColor(220, 220, 220)
        for col in range(3):
            separator.setBackground(col, grey)
        # Make it non-selectable and non-draggable
        separator.setFlags(Qt.ItemFlag.NoItemFlags)
        return separator

    def _is_separator_item(self, item: QTreeWidgetItem) -> bool:
        """Check if an item is a folder separator."""
        return item.data(0, Qt.ItemDataRole.UserRole) is None

    def _get_files_in_order(self) -> list[tuple[Path, str, int, int]]:
        """Get files in the current list order with sequence info."""
        files = []
        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                files.append(data)
        return files

    def _recalculate_sequence_names(self) -> None:
        """Recalculate sequence names after file reordering."""
        if not self.source_folders:
            return

        folder_to_index = {folder: i for i, folder in enumerate(self.source_folders)}
        folder_file_counts: dict[Path, int] = {}
        last_folder_idx = -1

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                source_dir = data[0]
                filename = data[1]
                folder_idx = folder_to_index.get(source_dir, 0)
                file_idx = folder_file_counts.get(source_dir, 0)
                folder_file_counts[source_dir] = file_idx + 1

                ext = Path(filename).suffix
                seq_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                item.setText(0, seq_name)
                item.setData(0, Qt.ItemDataRole.UserRole, (source_dir, filename, folder_idx, file_idx))
                last_folder_idx = folder_idx
            elif self._is_separator_item(item):
                # Update separator label based on next file's folder
                # Look ahead to find the next file's folder index
                next_folder_idx = last_folder_idx + 1
                for j in range(i + 1, self.file_list.topLevelItemCount()):
                    next_item = self.file_list.topLevelItem(j)
                    next_data = next_item.data(0, Qt.ItemDataRole.UserRole)
                    if next_data:
                        next_folder_idx = folder_to_index.get(next_data[0], last_folder_idx + 1)
                        break
                item.setText(1, f"── Sequence {next_folder_idx + 1} ──")

        # Update the With Transitions tab to reflect the new order
        self._update_sequence_table()

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
        folder = item.data(Qt.ItemDataRole.UserRole)
        if folder is not None:
            return folder
        item_text = item.text()
        if item_text.startswith("[M] ") or item_text.startswith("[T] "):
            return Path(item_text[4:])
        if ". " in item_text:
            return Path(item_text.split(". ", 1)[1])
        return Path(item_text)

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
        total = self._folder_file_counts.get(folder, 0)

        if total == 0:
            self.trim_slider.setRange(0)
            self.trim_slider.setEnabled(False)
            self.trim_label.setText("Frames: No images in folder")
            return

        trim_start, trim_end = self._folder_trim_settings.get(folder, (0, 0))

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
        """Handle trim slider value changes."""
        current_item = self._get_current_selected_item()
        if current_item is None:
            return

        folder = self._get_folder_from_item(current_item)
        if folder is None:
            return
        total = self._folder_file_counts.get(folder, 0)

        self._folder_trim_settings[folder] = (trim_start, trim_end)
        self._update_trim_label(folder, total, trim_start, trim_end)
        self._refresh_files(select_position='none')
        self._select_folder_boundary(folder, 'first' if handle == 'left' else 'last')

    def _select_folder_boundary(self, folder: Path, position: str) -> None:
        """Select the first or last file of a specific folder in the file list."""
        folder_str = str(folder)
        matching_indices = []

        for i in range(self.file_list.topLevelItemCount()):
            item = self.file_list.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and str(data[0]) == folder_str:
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

    def _export_sequence(self) -> None:
        """Export symlinks only (no transitions)."""
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

        try:
            results, session_id = self.manager.create_sequence_links(
                sources=self.source_folders,
                dest=Path(dst),
                files=files,
                trim_settings=self._folder_trim_settings
            )

            self._current_session_id = session_id

            if session_id:
                self._save_session_settings(session_id)

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

        transition_settings = self._get_transition_settings()

        # Use transition destination if specified, otherwise use main destination
        trans_dst = transition_settings.trans_destination
        if trans_dst is None:
            trans_dst = Path(dst)

        try:
            if len(self.source_folders) >= 2:
                self._process_with_transitions(Path(dst), trans_dst, files, transition_settings)
            else:
                # Fall back to regular export if less than 2 folders
                self._export_sequence()

        except SymlinkError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))

    def _save_session_settings(self, session_id: int) -> None:
        """Save transition settings and folder type overrides to database."""
        self.db.save_transition_settings(session_id, self._get_transition_settings())

        for folder in self.source_folders:
            trim_start, trim_end = self._folder_trim_settings.get(folder, (0, 0))
            folder_type = self._folder_type_overrides.get(folder, FolderType.AUTO)
            if trim_start > 0 or trim_end > 0 or folder_type != FolderType.AUTO:
                self.db.save_trim_settings(
                    session_id, str(folder), trim_start, trim_end, folder_type
                )

        for folder, pts in self._per_transition_settings.items():
            self.db.save_per_transition_settings(session_id, pts)

    def _process_with_transitions(
        self,
        symlink_dest: Path,
        trans_dest: Path,
        files: list[tuple],
        settings: TransitionSettings
    ) -> None:
        """Process files with cross-dissolve transitions."""
        self.manager.validate_paths(self.source_folders, symlink_dest)
        self.manager.cleanup_old_links(symlink_dest)

        # Also clean transition destination if different
        if trans_dest != symlink_dest:
            trans_dest.mkdir(parents=True, exist_ok=True)
            self.manager.cleanup_old_links(trans_dest)

        session_id = self.db.create_session(str(symlink_dest))
        self._current_session_id = session_id
        self._save_session_settings(session_id)

        files_by_folder: dict[Path, list[str]] = {}
        for source_dir, filename, folder_idx, file_idx in files:
            if source_dir not in files_by_folder:
                files_by_folder[source_dir] = []
            files_by_folder[source_dir].append(filename)

        generator = TransitionGenerator(settings)

        transitions = generator.identify_transition_boundaries(
            self.source_folders,
            files_by_folder,
            self._folder_type_overrides,
            self._per_transition_settings
        )

        trans_at_main_end: dict[Path, TransitionSpec] = {}
        trans_at_trans_start: dict[Path, TransitionSpec] = {}
        for trans in transitions:
            trans_at_main_end[trans.main_folder] = trans
            trans_at_trans_start[trans.trans_folder] = trans

        # Count total files including direct interpolation frames
        total_files = sum(len(f) for f in files_by_folder.values())
        for folder, direct_settings in self._direct_transitions.items():
            if direct_settings.enabled:
                total_files += direct_settings.frame_count

        progress = QProgressDialog("Generating sequence...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Cross-Dissolve Generation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        current_op = 0
        output_seq = 0
        symlink_count = 0
        blend_count = 0
        errors = []

        for folder_idx, folder in enumerate(self.source_folders):
            if progress.wasCanceled():
                break

            folder_files = files_by_folder.get(folder, [])
            if not folder_files:
                continue

            num_files = len(folder_files)

            trans_at_end = trans_at_main_end.get(folder)
            trans_at_start = trans_at_trans_start.get(folder)

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

                if trans_at_start:
                    right_overlap = trans_at_start.right_overlap
                    if file_idx < right_overlap:
                        should_skip = True
                        current_op += 1
                        progress.setValue(current_op)
                        continue

                if should_blend and blend_trans:
                    # Generate asymmetric blend frame
                    output_count = max(blend_trans.left_overlap, blend_trans.right_overlap)

                    # Calculate positions
                    t = blend_idx_in_overlap / (output_count - 1) if output_count > 1 else 0

                    # Get main frame
                    main_path = source_path

                    # Get trans frame position
                    trans_pos = t * (blend_trans.right_overlap - 1) if blend_trans.right_overlap > 1 else 0
                    trans_idx = int(trans_pos)
                    trans_idx = min(trans_idx, blend_trans.right_overlap - 1)
                    trans_file = blend_trans.trans_files[trans_idx]
                    trans_path = blend_trans.trans_folder / trans_file

                    factor = generator.blender.calculate_blend_factor(
                        blend_idx_in_overlap, output_count, settings.blend_curve
                    )

                    ext = f".{settings.output_format.lower()}"
                    output_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                    output_path = trans_dest / output_name

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
                        self.db.record_symlink(
                            session_id, str(main_path.resolve()),
                            str(output_path), filename, output_seq
                        )
                    else:
                        errors.append(f"Blend {filename}: {result.error}")

                    output_seq += 1
                else:
                    ext = source_path.suffix
                    link_name = f"seq{folder_idx + 1:02d}_{file_idx:04d}{ext}"
                    link_path = symlink_dest / link_name

                    rel_source = Path(os.path.relpath(source_path.resolve(), symlink_dest.resolve()))

                    try:
                        link_path.symlink_to(rel_source)
                        symlink_count += 1
                        self.db.record_symlink(
                            session_id, str(source_path.resolve()),
                            str(link_path), filename, output_seq
                        )
                    except OSError as e:
                        errors.append(f"Symlink {filename}: {e}")

                    output_seq += 1

                current_op += 1
                progress.setValue(current_op)

            # Check for direct interpolation after this folder
            if folder in self._direct_transitions:
                direct_settings = self._direct_transitions[folder]
                if direct_settings.enabled:
                    # Find next folder and get its first frame
                    next_folder_idx = folder_idx + 1
                    if next_folder_idx < len(self.source_folders):
                        next_folder = self.source_folders[next_folder_idx]
                        next_files = files_by_folder.get(next_folder, [])
                        if next_files and folder_files:
                            # Get last frame of current folder and first of next
                            last_frame = folder / folder_files[-1]
                            first_frame = next_folder / next_files[0]

                            progress.setLabelText(
                                f"Generating {direct_settings.method.value.upper()} frames..."
                            )

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
                                    self.db.record_symlink(
                                        session_id,
                                        str(result.source_a.resolve()),
                                        str(result.output_path),
                                        result.output_path.name,
                                        output_seq
                                    )
                                else:
                                    errors.append(
                                        f"Direct interp {result.output_path.name}: {result.error}"
                                    )
                                output_seq += 1

                            progress.setLabelText("Generating sequence...")

        progress.close()

        if progress.wasCanceled():
            QMessageBox.warning(
                self, "Canceled",
                f"Operation canceled.\n"
                f"Created {symlink_count} symlinks, {blend_count} blended frames."
            )
        elif errors:
            QMessageBox.warning(
                self, "Partial Success",
                f"Created {symlink_count} symlinks, {blend_count} blended frames.\n"
                f"{len(errors)} errors occurred.\n"
                f"First error: {errors[0] if errors else 'N/A'}\n"
                f"Symlinks: {symlink_dest}\n"
                f"Blends: {trans_dest}"
            )
        else:
            QMessageBox.information(
                self, "Success",
                f"Created {symlink_count} symlinks and {blend_count} blended frames.\n"
                f"Symlinks: {symlink_dest}\n"
                f"Blends: {trans_dest}"
            )
