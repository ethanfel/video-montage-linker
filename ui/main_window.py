"""Main window UI for Video Montage Linker."""

import os
import re
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, QEvent, QPoint
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor
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
)
from PyQt6.QtGui import QPixmap

from config import VIDEO_EXTENSIONS
from core import (
    BlendCurve,
    BlendMethod,
    FolderType,
    TransitionSettings,
    PerTransitionSettings,
    TransitionSpec,
    SymlinkError,
    DatabaseManager,
    TransitionGenerator,
    RifeDownloader,
    SymlinkManager,
)
from .widgets import TrimSlider


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
        self._current_session_id: Optional[int] = None
        self.db = DatabaseManager()
        self.manager = SymlinkManager(self.db)
        self._setup_window()
        self._create_widgets()
        self._create_layout()
        self._connect_signals()
        self.setAcceptDrops(True)

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

        # Destination - now with two paths
        self.dst_label = QLabel("Destination Folder:")
        self.dst_path = QLineEdit(placeholderText="Select destination folder for symlinks")
        self.dst_btn = QPushButton("Browse")

        self.trans_dst_label = QLabel("Transition Destination:")
        self.trans_dst_path = QLineEdit(placeholderText="Select destination for transition output (optional)")
        self.trans_dst_btn = QPushButton("Browse")

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

        # Trim slider
        self.trim_slider = TrimSlider()
        self.trim_label = QLabel("Frames: All included")
        self.trim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Sequence table (2-column: Main Frame | Transition Frame)
        self.sequence_table = QTreeWidget()
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
        self.blend_method_combo.addItem("RIFE (AI)", BlendMethod.RIFE)
        self.blend_method_combo.setToolTip(
            "Blending method:\n"
            "- Cross-Dissolve: Simple alpha blend (fast, may ghost)\n"
            "- Optical Flow: Motion-compensated blend (slower, less ghosting)\n"
            "- RIFE: AI frame interpolation (best quality, requires rife-ncnn-vulkan)"
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

        # Tab 2: With Transitions (2-column view)
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
        self.dst_path.editingFinished.connect(self._on_destination_changed)
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
        self.rife_path_btn.clicked.connect(self._browse_rife_binary)
        self.rife_download_btn.clicked.connect(self._download_rife_binary)

        # Sequence table selection - show image
        self.sequence_table.currentItemChanged.connect(self._on_sequence_table_selected)

        # Update sequence table when transitions setting changes
        self.transition_group.toggled.connect(self._update_sequence_table)

        # Update sequence table when switching to "With Transitions" tab
        self.sequence_tabs.currentChanged.connect(self._on_sequence_tab_changed)

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
        is_rife = (method == BlendMethod.RIFE)
        self.rife_path_label.setVisible(is_rife)
        self.rife_path_input.setVisible(is_rife)
        self.rife_path_btn.setVisible(is_rife)
        self.rife_download_btn.setVisible(is_rife)

        if is_rife:
            self._update_rife_download_button()

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

    def _on_sequence_tab_changed(self, index: int) -> None:
        """Handle sequence tab change to update the With Transitions view."""
        if index == 1:  # "With Transitions" tab
            self._update_sequence_table()

    def _update_sequence_table(self, _=None) -> None:
        """Update the 2-column sequence table showing Main/Transition frame pairing."""
        self.sequence_table.clear()

        if not self.source_folders:
            return

        files = self._get_files_in_order()
        if not files:
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

            # Blend images using selected method
            if settings.blend_method == BlendMethod.OPTICAL_FLOW:
                blended = ImageBlender.optical_flow_blend(img_a, img_b, factor)
            elif settings.blend_method == BlendMethod.RIFE:
                blended = ImageBlender.rife_blend(img_a, img_b, factor, settings.rife_binary_path)
            else:
                blended = Image.blend(img_a, img_b, factor)

            # Convert to QPixmap
            qim = ImageQt(blended.convert('RGBA'))
            pixmap = QPixmap.fromImage(qim)

            self._current_pixmap = pixmap
            self._apply_zoom()

            # Update labels
            self.image_index_label.setText(f"{row_idx + 1} / {total}")
            seq_name = f"seq{data0[2] + 1:02d}_{data0[3]:04d}"
            self.image_name_label.setText(f"[B] {seq_name} ({main_file} + {trans_file}) @ {factor:.0%}")

            img_a.close()
            img_b.close()

        except Exception as e:
            self.image_label.setText(f"Error generating blend preview:\n{e}")
            self.image_name_label.setText("")
            self._current_pixmap = None

    def _browse_trans_destination(self) -> None:
        """Select transition destination folder via file dialog."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Transition Destination Folder", start_dir
        )
        if path:
            self.trans_dst_path.setText(path)
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
        """Remove selected source folder(s)."""
        result = self._get_selected_folder()
        if result is None:
            return

        folder, idx = result

        if folder in self._folder_type_overrides:
            del self._folder_type_overrides[folder]
        if folder in self._per_transition_settings:
            del self._per_transition_settings[folder]

        del self.source_folders[idx]

        self._sync_dual_lists()
        self._refresh_files()
        self._update_flow_arrows()

    def _remove_selected_files(self) -> None:
        """Remove selected files from the file list."""
        selected = self.file_list.selectedItems()
        if not selected:
            return

        rows = sorted([self.file_list.indexOfTopLevelItem(item) for item in selected], reverse=True)
        for row in rows:
            self.file_list.takeTopLevelItem(row)

    def _browse_destination(self) -> None:
        """Select destination folder via file dialog."""
        start_dir = self.last_directory or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", start_dir
        )
        if path:
            self.dst_path.setText(path)
            self.last_directory = str(Path(path).parent)
            self._try_resume_session(path)

    def _on_destination_changed(self) -> None:
        """Handle destination path text field changes."""
        path = self.dst_path.text().strip()
        if path and Path(path).is_dir():
            resolved = str(Path(path).resolve())
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
                self.trans_dst_path.setText(str(db_transition_settings.trans_destination))
            if db_transition_settings.rife_binary_path:
                self.rife_path_input.setText(str(db_transition_settings.rife_binary_path))
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
        trans_path = self.trans_dst_path.text().strip()
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
            rife_binary_path=rife_path
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

            folder_idx = folder_to_index.get(folder, 0)

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
        dst = self.dst_path.text()

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
        dst = self.dst_path.text()

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

        total_files = sum(len(f) for f in files_by_folder.values())
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
                        settings.blend_method, settings.rife_binary_path
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
