"""Image blending and transition generation for Video Montage Linker."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models import (
    BlendCurve,
    BlendMethod,
    FolderType,
    TransitionSettings,
    PerTransitionSettings,
    BlendResult,
    TransitionSpec,
)


# Cache directory for downloaded binaries
CACHE_DIR = Path.home() / '.cache' / 'video-montage-linker'
RIFE_GITHUB_API = 'https://api.github.com/repos/nihui/rife-ncnn-vulkan/releases/latest'


class RifeDownloader:
    """Handles automatic download and caching of rife-ncnn-vulkan binary."""

    @staticmethod
    def get_cache_dir() -> Path:
        """Get the cache directory, creating it if needed."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR

    @staticmethod
    def get_platform_identifier() -> Optional[str]:
        """Get the platform identifier for downloading the correct binary.

        Returns:
            Platform string like 'ubuntu', 'windows', 'macos', or None if unsupported.
        """
        system = platform.system().lower()
        if system == 'linux':
            return 'ubuntu'
        elif system == 'windows':
            return 'windows'
        elif system == 'darwin':
            return 'macos'
        return None

    @staticmethod
    def get_cached_binary() -> Optional[Path]:
        """Get the path to a cached RIFE binary if it exists.

        Returns:
            Path to the binary, or None if not cached.
        """
        cache_dir = RifeDownloader.get_cache_dir()
        rife_dir = cache_dir / 'rife-ncnn-vulkan'

        if not rife_dir.exists():
            return None

        # Look for the binary
        system = platform.system().lower()
        if system == 'windows':
            binary_name = 'rife-ncnn-vulkan.exe'
        else:
            binary_name = 'rife-ncnn-vulkan'

        binary_path = rife_dir / binary_name
        if binary_path.exists():
            # Ensure it's executable on Unix
            if system != 'windows':
                binary_path.chmod(0o755)
            return binary_path

        return None

    @staticmethod
    def get_latest_release_info() -> Optional[dict]:
        """Fetch the latest release info from GitHub.

        Returns:
            Dict with 'tag_name' and 'assets' list, or None on error.
        """
        try:
            req = urllib.request.Request(
                RIFE_GITHUB_API,
                headers={'User-Agent': 'video-montage-linker'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception:
            return None

    @staticmethod
    def find_asset_url(release_info: dict, platform_id: str) -> Optional[str]:
        """Find the download URL for the platform-specific asset.

        Args:
            release_info: Release info dict from GitHub API.
            platform_id: Platform identifier (ubuntu, windows, macos).

        Returns:
            Download URL or None if not found.
        """
        assets = release_info.get('assets', [])
        for asset in assets:
            name = asset.get('name', '').lower()
            # Match patterns like rife-ncnn-vulkan-20221029-ubuntu.zip
            if platform_id in name and name.endswith('.zip'):
                return asset.get('browser_download_url')
        return None

    @staticmethod
    def download_and_extract(url: str, progress_callback=None, cancelled_check=None) -> Optional[Path]:
        """Download and extract the RIFE binary.

        Args:
            url: URL to download from.
            progress_callback: Optional callback(downloaded, total) for progress.
            cancelled_check: Optional callable that returns True if cancelled.

        Returns:
            Path to the extracted binary, or None on error/cancel.
        """
        cache_dir = RifeDownloader.get_cache_dir()
        rife_dir = cache_dir / 'rife-ncnn-vulkan'

        try:
            # Download to temp file
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'video-montage-linker'}
            )

            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192

                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    while True:
                        # Check for cancellation
                        if cancelled_check and cancelled_check():
                            tmp_path.unlink(missing_ok=True)
                            return None

                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # Remove old installation if exists
            if rife_dir.exists():
                shutil.rmtree(rife_dir)

            # Extract
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                # Find the root directory in the zip
                names = zf.namelist()
                if names:
                    # Most zips have a root folder like rife-ncnn-vulkan-20221029-ubuntu/
                    root_in_zip = names[0].split('/')[0]

                    # Extract to temp location
                    extract_tmp = cache_dir / 'extract_tmp'
                    if extract_tmp.exists():
                        shutil.rmtree(extract_tmp)
                    zf.extractall(extract_tmp)

                    # Move the extracted folder to final location
                    extracted_dir = extract_tmp / root_in_zip
                    if extracted_dir.exists():
                        shutil.move(str(extracted_dir), str(rife_dir))

                    # Cleanup
                    if extract_tmp.exists():
                        shutil.rmtree(extract_tmp)

            # Cleanup temp zip
            tmp_path.unlink(missing_ok=True)

            # Return the binary path
            return RifeDownloader.get_cached_binary()

        except Exception as e:
            # Cleanup on error
            try:
                if 'tmp_path' in locals():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    @staticmethod
    def ensure_binary(progress_callback=None) -> Optional[Path]:
        """Ensure RIFE binary is available, downloading if needed.

        Args:
            progress_callback: Optional callback(downloaded, total) for progress.

        Returns:
            Path to the binary, or None if unavailable.
        """
        # Check if already cached
        cached = RifeDownloader.get_cached_binary()
        if cached:
            return cached

        # Check system PATH
        system_binary = shutil.which('rife-ncnn-vulkan')
        if system_binary:
            return Path(system_binary)

        # Need to download
        platform_id = RifeDownloader.get_platform_identifier()
        if not platform_id:
            return None

        release_info = RifeDownloader.get_latest_release_info()
        if not release_info:
            return None

        asset_url = RifeDownloader.find_asset_url(release_info, platform_id)
        if not asset_url:
            return None

        return RifeDownloader.download_and_extract(asset_url, progress_callback)

    @staticmethod
    def get_version_info() -> Optional[str]:
        """Get the version of the cached binary.

        Returns:
            Version string or None.
        """
        binary = RifeDownloader.get_cached_binary()
        if not binary:
            return None

        # The version is typically in the parent directory name
        # e.g., rife-ncnn-vulkan-20221029-ubuntu
        try:
            result = subprocess.run(
                [str(binary), '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse version from help output if available
            return "installed"
        except Exception:
            return None


class ImageBlender:
    """Handles image blending operations for cross-dissolve transitions."""

    @staticmethod
    def calculate_blend_factor(frame_idx: int, total: int, curve: BlendCurve) -> float:
        """Calculate blend factor based on curve type.

        Args:
            frame_idx: Current frame index within the overlap (0 to total-1).
            total: Total number of overlap frames.
            curve: The blend curve type.

        Returns:
            Blend factor from 0.0 (100% image A) to 1.0 (100% image B).
        """
        if total <= 1:
            return 1.0

        t = frame_idx / (total - 1)

        if curve == BlendCurve.LINEAR:
            return t
        elif curve == BlendCurve.EASE_IN:
            return t * t
        elif curve == BlendCurve.EASE_OUT:
            return 1 - (1 - t) ** 2
        elif curve == BlendCurve.EASE_IN_OUT:
            # Smooth S-curve using smoothstep
            return t * t * (3 - 2 * t)
        else:
            return t

    @staticmethod
    def interpolate_frame(frames: list, position: float) -> Image.Image:
        """Get an interpolated frame at a fractional position.

        When position is fractional, blends between adjacent frames.

        Args:
            frames: List of PIL Image objects.
            position: Position in the frame list (can be fractional).

        Returns:
            The interpolated PIL Image.
        """
        if len(frames) == 1:
            return frames[0]

        # Clamp position to valid range
        position = max(0, min(position, len(frames) - 1))

        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, len(frames) - 1)

        if lower_idx == upper_idx:
            return frames[lower_idx]

        # Fractional part determines blend
        frac = position - lower_idx
        return Image.blend(frames[lower_idx], frames[upper_idx], frac)

    @staticmethod
    def optical_flow_blend(img_a: Image.Image, img_b: Image.Image, t: float) -> Image.Image:
        """Blend using OpenCV optical flow for motion compensation.

        Uses Farneback dense optical flow to warp frames and reduce ghosting
        artifacts compared to simple alpha blending.

        Args:
            img_a: First PIL Image (source frame).
            img_b: Second PIL Image (target frame).
            t: Interpolation factor 0.0 (100% A) to 1.0 (100% B).

        Returns:
            Motion-compensated blended PIL Image.
        """
        try:
            import cv2
        except ImportError:
            # Fall back to alpha blend if OpenCV not available
            return Image.blend(img_a, img_b, t)

        # Convert PIL to numpy (RGB)
        arr_a = np.array(img_a.convert('RGB'))
        arr_b = np.array(img_b.convert('RGB'))

        # Calculate dense optical flow (A -> B)
        gray_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        h, w = flow.shape[:2]

        # Create coordinate grids
        x_coords = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        y_coords = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

        # Warp A forward by t * flow
        flow_t = flow * t
        map_x_a = x_coords + flow_t[..., 0]
        map_y_a = y_coords + flow_t[..., 1]
        warped_a = cv2.remap(arr_a, map_x_a, map_y_a, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

        # Warp B backward by (1-t) * flow
        flow_back = -flow * (1 - t)
        map_x_b = x_coords + flow_back[..., 0]
        map_y_b = y_coords + flow_back[..., 1]
        warped_b = cv2.remap(arr_b, map_x_b, map_y_b, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

        # Blend the aligned frames
        result = cv2.addWeighted(warped_a, 1 - t, warped_b, t, 0)

        return Image.fromarray(result)

    @staticmethod
    def rife_blend(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float,
        binary_path: Optional[Path] = None,
        auto_download: bool = True
    ) -> Image.Image:
        """Blend using RIFE AI frame interpolation.

        Attempts to use rife-ncnn-vulkan binary, auto-downloading if needed,
        then falls back to optical flow if unavailable.

        Args:
            img_a: First PIL Image (source frame).
            img_b: Second PIL Image (target frame).
            t: Interpolation factor 0.0 (100% A) to 1.0 (100% B).
            binary_path: Optional path to rife-ncnn-vulkan binary.
            auto_download: Whether to auto-download RIFE if not found.

        Returns:
            AI-interpolated blended PIL Image.
        """
        # Try NCNN binary first (specified path)
        if binary_path and binary_path.exists():
            result = ImageBlender._rife_ncnn(img_a, img_b, t, binary_path)
            if result is not None:
                return result

        # Try to find rife-ncnn-vulkan in PATH
        ncnn_path = shutil.which('rife-ncnn-vulkan')
        if ncnn_path:
            result = ImageBlender._rife_ncnn(img_a, img_b, t, Path(ncnn_path))
            if result is not None:
                return result

        # Try cached binary
        cached = RifeDownloader.get_cached_binary()
        if cached:
            result = ImageBlender._rife_ncnn(img_a, img_b, t, cached)
            if result is not None:
                return result

        # Auto-download if enabled
        if auto_download:
            downloaded = RifeDownloader.ensure_binary()
            if downloaded:
                result = ImageBlender._rife_ncnn(img_a, img_b, t, downloaded)
                if result is not None:
                    return result

        # Fall back to optical flow if RIFE not available
        return ImageBlender.optical_flow_blend(img_a, img_b, t)

    @staticmethod
    def _rife_ncnn(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float,
        binary: Path
    ) -> Optional[Image.Image]:
        """Use rife-ncnn-vulkan binary for interpolation.

        Args:
            img_a: First PIL Image.
            img_b: Second PIL Image.
            t: Interpolation timestep (0.0 to 1.0).
            binary: Path to rife-ncnn-vulkan binary.

        Returns:
            Interpolated PIL Image, or None if failed.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_a = tmp / 'a.png'
                input_b = tmp / 'b.png'
                output_file = tmp / 'out.png'

                # Save input images
                img_a.convert('RGB').save(input_a)
                img_b.convert('RGB').save(input_b)

                # Run NCNN binary
                # Note: rife-ncnn-vulkan uses -n for timestep count, not direct timestep
                # We generate a single frame at position t
                cmd = [
                    str(binary),
                    '-0', str(input_a),
                    '-1', str(input_b),
                    '-o', str(output_file),
                ]

                # Some versions support -s for timestep
                # Try with timestep first, fall back to simple interpolation
                try:
                    result = subprocess.run(
                        cmd + ['-s', str(t)],
                        check=True,
                        capture_output=True,
                        timeout=30
                    )
                except subprocess.CalledProcessError:
                    # Try without timestep (generates middle frame at t=0.5)
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=30
                    )

                if output_file.exists():
                    return Image.open(output_file).copy()

        except (subprocess.SubprocessError, OSError, IOError):
            pass

        return None

    @staticmethod
    def blend_images(
        img_a_path: Path,
        img_b_path: Path,
        factor: float,
        output_path: Path,
        output_format: str = 'png',
        output_quality: int = 95,
        webp_method: int = 4,
        blend_method: BlendMethod = BlendMethod.ALPHA,
        rife_binary_path: Optional[Path] = None
    ) -> BlendResult:
        """Blend two images together.

        Args:
            img_a_path: Path to first image (main sequence).
            img_b_path: Path to second image (transition sequence).
            factor: Blend factor 0.0 (100% A) to 1.0 (100% B).
            output_path: Where to save the blended image.
            output_format: Output format (png, jpeg, webp).
            output_quality: Quality for JPEG output (1-100).
            webp_method: WebP compression method (0-6, higher = smaller but slower).
            blend_method: The blending method to use (alpha, optical_flow, or rife).
            rife_binary_path: Optional path to rife-ncnn-vulkan binary.

        Returns:
            BlendResult with operation status.
        """
        try:
            img_a = Image.open(img_a_path)
            img_b = Image.open(img_b_path)

            # Handle different sizes - resize B to match A
            if img_a.size != img_b.size:
                img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

            # Normalize to RGBA for consistent blending
            if img_a.mode != 'RGBA':
                img_a = img_a.convert('RGBA')
            if img_b.mode != 'RGBA':
                img_b = img_b.convert('RGBA')

            # Blend images using selected method
            if blend_method == BlendMethod.OPTICAL_FLOW:
                blended = ImageBlender.optical_flow_blend(img_a, img_b, factor)
            elif blend_method == BlendMethod.RIFE:
                blended = ImageBlender.rife_blend(img_a, img_b, factor, rife_binary_path)
            else:
                # Default: simple alpha blend
                blended = Image.blend(img_a, img_b, factor)

            # Convert back to RGB if saving to JPEG
            if output_format.lower() in ('jpg', 'jpeg'):
                blended = blended.convert('RGB')

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with appropriate options
            save_kwargs = {}
            if output_format.lower() in ('jpg', 'jpeg'):
                save_kwargs['quality'] = output_quality
            elif output_format.lower() == 'webp':
                # WebP is always lossless with method setting
                save_kwargs['lossless'] = True
                save_kwargs['method'] = webp_method
            elif output_format.lower() == 'png':
                save_kwargs['compress_level'] = 6

            blended.save(output_path, **save_kwargs)

            return BlendResult(
                output_path=output_path,
                source_a=img_a_path,
                source_b=img_b_path,
                blend_factor=factor,
                success=True
            )

        except Exception as e:
            return BlendResult(
                output_path=output_path,
                source_a=img_a_path,
                source_b=img_b_path,
                blend_factor=factor,
                success=False,
                error=str(e)
            )

    def blend_images_pil(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        factor: float,
        output_path: Path,
        output_format: str = 'png',
        output_quality: int = 95,
        webp_method: int = 4,
        blend_method: BlendMethod = BlendMethod.ALPHA,
        rife_binary_path: Optional[Path] = None
    ) -> BlendResult:
        """Blend two PIL Image objects together.

        Args:
            img_a: First PIL Image (main sequence).
            img_b: Second PIL Image (transition sequence).
            factor: Blend factor 0.0 (100% A) to 1.0 (100% B).
            output_path: Where to save the blended image.
            output_format: Output format (png, jpeg, webp).
            output_quality: Quality for JPEG output (1-100).
            webp_method: WebP compression method (0-6).
            blend_method: The blending method to use (alpha, optical_flow, or rife).
            rife_binary_path: Optional path to rife-ncnn-vulkan binary.

        Returns:
            BlendResult with operation status.
        """
        try:
            # Handle different sizes - resize B to match A
            if img_a.size != img_b.size:
                img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

            # Normalize to RGBA for consistent blending
            if img_a.mode != 'RGBA':
                img_a = img_a.convert('RGBA')
            if img_b.mode != 'RGBA':
                img_b = img_b.convert('RGBA')

            # Blend images using selected method
            if blend_method == BlendMethod.OPTICAL_FLOW:
                blended = ImageBlender.optical_flow_blend(img_a, img_b, factor)
            elif blend_method == BlendMethod.RIFE:
                blended = ImageBlender.rife_blend(img_a, img_b, factor, rife_binary_path)
            else:
                # Default: simple alpha blend
                blended = Image.blend(img_a, img_b, factor)

            # Convert back to RGB if saving to JPEG
            if output_format.lower() in ('jpg', 'jpeg'):
                blended = blended.convert('RGB')

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with appropriate options
            save_kwargs = {}
            if output_format.lower() in ('jpg', 'jpeg'):
                save_kwargs['quality'] = output_quality
            elif output_format.lower() == 'webp':
                save_kwargs['lossless'] = True
                save_kwargs['method'] = webp_method
            elif output_format.lower() == 'png':
                save_kwargs['compress_level'] = 6

            blended.save(output_path, **save_kwargs)

            return BlendResult(
                output_path=output_path,
                source_a=Path("memory"),
                source_b=Path("memory"),
                blend_factor=factor,
                success=True
            )

        except Exception as e:
            return BlendResult(
                output_path=output_path,
                source_a=Path("memory"),
                source_b=Path("memory"),
                blend_factor=factor,
                success=False,
                error=str(e)
            )


class TransitionGenerator:
    """Generates cross-dissolve transitions between folder sequences."""

    def __init__(self, settings: TransitionSettings):
        """Initialize the transition generator.

        Args:
            settings: Transition settings to use.
        """
        self.settings = settings
        self.blender = ImageBlender()

    def get_folder_type(
        self,
        index: int,
        overrides: Optional[dict[Path, FolderType]] = None,
        folder: Optional[Path] = None
    ) -> FolderType:
        """Determine folder type based on position or override.

        Args:
            index: 0-based position of folder in list.
            overrides: Optional dict of folder path to FolderType overrides.
            folder: The folder path for checking overrides.

        Returns:
            FolderType.MAIN for odd positions (1, 3, 5...), TRANSITION for even.
        """
        if overrides and folder and folder in overrides:
            override = overrides[folder]
            if override != FolderType.AUTO:
                return override

        # Position-based: index 0, 2, 4... are MAIN; 1, 3, 5... are TRANSITION
        return FolderType.MAIN if index % 2 == 0 else FolderType.TRANSITION

    def identify_transition_boundaries(
        self,
        folders: list[Path],
        files_by_folder: dict[Path, list[str]],
        folder_overrides: Optional[dict[Path, FolderType]] = None,
        per_transition_settings: Optional[dict[Path, PerTransitionSettings]] = None
    ) -> list[TransitionSpec]:
        """Identify boundaries where transitions should occur.

        Transitions happen at boundaries where folder types change
        (MAIN->TRANSITION or TRANSITION->MAIN).

        Args:
            folders: List of folders in order.
            files_by_folder: Dict mapping folders to their file lists.
            folder_overrides: Optional folder type overrides.
            per_transition_settings: Optional per-transition overlap settings.

        Returns:
            List of TransitionSpec objects describing each transition.
        """
        if len(folders) < 2:
            return []

        transitions = []
        cumulative_idx = 0
        folder_start_indices = {}

        # Calculate start indices for each folder
        for folder in folders:
            folder_start_indices[folder] = cumulative_idx
            cumulative_idx += len(files_by_folder.get(folder, []))

        # Look for transition boundaries (MAIN->TRANSITION and TRANSITION->MAIN)
        for i in range(len(folders) - 1):
            folder_a = folders[i]
            folder_b = folders[i + 1]

            type_a = self.get_folder_type(i, folder_overrides, folder_a)
            type_b = self.get_folder_type(i + 1, folder_overrides, folder_b)

            # Create transition when types differ (MAIN->TRANS or TRANS->MAIN)
            if type_a != type_b:
                files_a = files_by_folder.get(folder_a, [])
                files_b = files_by_folder.get(folder_b, [])

                if not files_a or not files_b:
                    continue

                # Get per-transition overlap settings if available
                # Use folder_b as the key (the "incoming" folder)
                if per_transition_settings and folder_b in per_transition_settings:
                    pts = per_transition_settings[folder_b]
                    left_overlap = pts.left_overlap
                    right_overlap = pts.right_overlap
                else:
                    # Use default of 16 for both
                    left_overlap = 16
                    right_overlap = 16

                # Cap overlaps by available files
                left_overlap = min(left_overlap, len(files_a))
                right_overlap = min(right_overlap, len(files_b))

                if left_overlap < 1 or right_overlap < 1:
                    continue

                transitions.append(TransitionSpec(
                    main_folder=folder_a,
                    trans_folder=folder_b,
                    main_files=files_a,
                    trans_files=files_b,
                    left_overlap=left_overlap,
                    right_overlap=right_overlap,
                    main_start_idx=folder_start_indices[folder_a],
                    trans_start_idx=folder_start_indices[folder_b]
                ))

        return transitions

    def generate_asymmetric_blend_frames(
        self,
        spec: TransitionSpec,
        dest: Path,
        folder_idx_main: int,
        base_file_idx: int
    ) -> list[BlendResult]:
        """Generate blended frames for an asymmetric transition.

        For asymmetric overlap, left_overlap != right_overlap. The blend
        creates max(left, right) output frames, with source frames interpolated
        to match the longer sequence.

        Args:
            spec: TransitionSpec describing the transition.
            dest: Destination directory for blended frames.
            folder_idx_main: Folder index for sequence naming.
            base_file_idx: Starting file index for sequence naming.

        Returns:
            List of BlendResult objects.
        """
        results = []
        left_overlap = spec.left_overlap
        right_overlap = spec.right_overlap
        output_count = max(left_overlap, right_overlap)

        # Get the frames to use
        main_start = len(spec.main_files) - left_overlap
        main_frames_paths = [
            spec.main_folder / spec.main_files[main_start + i]
            for i in range(left_overlap)
        ]
        trans_frames_paths = [
            spec.trans_folder / spec.trans_files[i]
            for i in range(right_overlap)
        ]

        # Load all frames into memory for interpolation
        main_frames = [Image.open(p) for p in main_frames_paths]
        trans_frames = [Image.open(p) for p in trans_frames_paths]

        # Normalize all frames to RGBA
        main_frames = [f.convert('RGBA') if f.mode != 'RGBA' else f for f in main_frames]
        trans_frames = [f.convert('RGBA') if f.mode != 'RGBA' else f for f in trans_frames]

        # Resize trans frames to match main frame size if needed
        target_size = main_frames[0].size
        trans_frames = [
            f.resize(target_size, Image.Resampling.LANCZOS) if f.size != target_size else f
            for f in trans_frames
        ]

        for i in range(output_count):
            # Calculate position in each source (0.0 to 1.0)
            t = i / (output_count - 1) if output_count > 1 else 0

            # Map to source frame indices
            main_pos = t * (left_overlap - 1) if left_overlap > 1 else 0
            trans_pos = t * (right_overlap - 1) if right_overlap > 1 else 0

            # Get source frames (interpolate if fractional)
            main_frame = self.blender.interpolate_frame(main_frames, main_pos)
            trans_frame = self.blender.interpolate_frame(trans_frames, trans_pos)

            # Calculate blend factor with curve
            factor = self.blender.calculate_blend_factor(
                i, output_count, self.settings.blend_curve
            )

            # Generate output filename
            ext = f".{self.settings.output_format.lower()}"
            file_idx = base_file_idx + i
            output_name = f"seq{folder_idx_main + 1:02d}_{file_idx:04d}{ext}"
            output_path = dest / output_name

            result = self.blender.blend_images_pil(
                main_frame,
                trans_frame,
                factor,
                output_path,
                self.settings.output_format,
                self.settings.output_quality,
                self.settings.webp_method,
                self.settings.blend_method,
                self.settings.rife_binary_path
            )
            results.append(result)

        # Close loaded images
        for f in main_frames:
            f.close()
        for f in trans_frames:
            f.close()

        return results

    def generate_transition_frames(
        self,
        spec: TransitionSpec,
        dest: Path,
        folder_idx_main: int,
        base_file_idx: int
    ) -> list[BlendResult]:
        """Generate blended frames for a transition.

        Uses asymmetric blend if left_overlap != right_overlap.

        Args:
            spec: TransitionSpec describing the transition.
            dest: Destination directory for blended frames.
            folder_idx_main: Folder index for sequence naming.
            base_file_idx: Starting file index for sequence naming.

        Returns:
            List of BlendResult objects.
        """
        # Use asymmetric blend for all cases (handles symmetric too)
        return self.generate_asymmetric_blend_frames(
            spec, dest, folder_idx_main, base_file_idx
        )
