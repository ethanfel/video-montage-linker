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
    DirectInterpolationMethod,
    DirectTransitionSettings,
)


# Cache directory for downloaded binaries
CACHE_DIR = Path.home() / '.cache' / 'video-montage-linker'
RIFE_GITHUB_API = 'https://api.github.com/repos/nihui/rife-ncnn-vulkan/releases/latest'
PRACTICAL_RIFE_VENV_DIR = CACHE_DIR / 'venv-rife'

# Optical flow presets
OPTICAL_FLOW_PRESETS = {
    'fast': {'levels': 2, 'winsize': 11, 'iterations': 2, 'poly_n': 5, 'poly_sigma': 1.1},
    'balanced': {'levels': 3, 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2},
    'quality': {'levels': 5, 'winsize': 21, 'iterations': 5, 'poly_n': 7, 'poly_sigma': 1.5},
    'max': {'levels': 7, 'winsize': 31, 'iterations': 10, 'poly_n': 7, 'poly_sigma': 1.5},
}


class PracticalRifeEnv:
    """Manages isolated Python environment for Practical-RIFE."""

    VENV_DIR = PRACTICAL_RIFE_VENV_DIR
    MODEL_CACHE_DIR = CACHE_DIR / 'practical-rife'
    REQUIRED_PACKAGES = ['torch', 'torchvision', 'numpy']

    # Available Practical-RIFE models
    AVAILABLE_MODELS = ['v4.26', 'v4.25', 'v4.22', 'v4.20', 'v4.18', 'v4.15']

    @classmethod
    def get_venv_python(cls) -> Optional[Path]:
        """Get path to venv Python executable."""
        if cls.VENV_DIR.exists():
            if sys.platform == 'win32':
                return cls.VENV_DIR / 'Scripts' / 'python.exe'
            return cls.VENV_DIR / 'bin' / 'python'
        return None

    @classmethod
    def is_setup(cls) -> bool:
        """Check if venv exists and has required packages."""
        python = cls.get_venv_python()
        if not python or not python.exists():
            return False
        # Check if torch is importable
        result = subprocess.run(
            [str(python), '-c', 'import torch; print(torch.__version__)'],
            capture_output=True
        )
        return result.returncode == 0

    @classmethod
    def get_torch_version(cls) -> Optional[str]:
        """Get installed torch version in venv."""
        python = cls.get_venv_python()
        if not python or not python.exists():
            return None
        result = subprocess.run(
            [str(python), '-c', 'import torch; print(torch.__version__)'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    @classmethod
    def setup_venv(cls, progress_callback=None, cancelled_check=None) -> bool:
        """Create venv and install PyTorch.

        Args:
            progress_callback: Optional callback(message, percent) for progress.
            cancelled_check: Optional callable that returns True if cancelled.

        Returns:
            True if setup was successful.
        """
        import venv

        try:
            # 1. Create venv
            if progress_callback:
                progress_callback("Creating virtual environment...", 10)
            if cancelled_check and cancelled_check():
                return False

            # Remove old venv if exists
            if cls.VENV_DIR.exists():
                shutil.rmtree(cls.VENV_DIR)

            venv.create(cls.VENV_DIR, with_pip=True)

            # 2. Get pip path
            python = cls.get_venv_python()
            if not python:
                return False

            # 3. Upgrade pip
            if progress_callback:
                progress_callback("Upgrading pip...", 20)
            if cancelled_check and cancelled_check():
                return False

            subprocess.run(
                [str(python), '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                check=True
            )

            # 4. Install PyTorch (this is the big download)
            if progress_callback:
                progress_callback("Installing PyTorch (this may take a while)...", 30)
            if cancelled_check and cancelled_check():
                return False

            # Try to install with CUDA support first, fall back to CPU
            # Use pip index to get the right version
            result = subprocess.run(
                [str(python), '-m', 'pip', 'install', 'torch', 'torchvision'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Try CPU-only version
                subprocess.run(
                    [str(python), '-m', 'pip', 'install',
                     'torch', 'torchvision',
                     '--index-url', 'https://download.pytorch.org/whl/cpu'],
                    capture_output=True,
                    check=True
                )

            if progress_callback:
                progress_callback("Installing additional dependencies...", 90)
            if cancelled_check and cancelled_check():
                return False

            # Install numpy (usually a dependency of torch) and gdown (for Google Drive downloads)
            subprocess.run(
                [str(python), '-m', 'pip', 'install', 'numpy', 'gdown'],
                capture_output=True
            )

            if progress_callback:
                progress_callback("Setup complete!", 100)

            return cls.is_setup()

        except Exception as e:
            # Cleanup on error
            if cls.VENV_DIR.exists():
                try:
                    shutil.rmtree(cls.VENV_DIR)
                except Exception:
                    pass
            return False

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return list of available model versions."""
        return cls.AVAILABLE_MODELS.copy()

    @classmethod
    def get_worker_script(cls) -> Path:
        """Get path to the RIFE worker script."""
        return Path(__file__).parent / 'rife_worker.py'

    @classmethod
    def run_interpolation(
        cls,
        img_a_path: Path,
        img_b_path: Path,
        output_path: Path,
        t: float,
        model: str = 'v4.25',
        ensemble: bool = False
    ) -> tuple[bool, str]:
        """Run RIFE interpolation via subprocess in venv.

        Args:
            img_a_path: Path to first input image.
            img_b_path: Path to second input image.
            output_path: Path to output image.
            t: Timestep for interpolation (0.0 to 1.0).
            model: Model version to use.
            ensemble: Enable ensemble mode.

        Returns:
            Tuple of (success, error_message).
        """
        python = cls.get_venv_python()
        if not python or not python.exists():
            return False, "venv python not found"

        script = cls.get_worker_script()
        if not script.exists():
            return False, f"worker script not found: {script}"

        cmd = [
            str(python), str(script),
            '--input0', str(img_a_path),
            '--input1', str(img_b_path),
            '--output', str(output_path),
            '--timestep', str(t),
            '--model', model,
            '--model-dir', str(cls.MODEL_CACHE_DIR)
        ]

        if ensemble:
            cmd.append('--ensemble')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per frame
            )
            if result.returncode == 0 and output_path.exists():
                return True, ""
            else:
                error = result.stderr.strip() if result.stderr else f"returncode={result.returncode}"
                return False, error
        except subprocess.TimeoutExpired:
            return False, "timeout (120s)"
        except Exception as e:
            return False, str(e)


class FilmEnv:
    """Manages FILM frame interpolation using shared venv with RIFE."""

    VENV_DIR = PRACTICAL_RIFE_VENV_DIR  # Share venv with RIFE
    MODEL_CACHE_DIR = CACHE_DIR / 'film'
    MODEL_FILENAME = 'film_net_fp32.pt'
    MODEL_URL = 'https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt'

    # Keep REPO_DIR for backward compat (but unused now - model is downloaded directly)
    REPO_DIR = CACHE_DIR / 'frame-interpolation-pytorch'

    @classmethod
    def get_venv_python(cls) -> Optional[Path]:
        """Get path to venv Python executable."""
        if cls.VENV_DIR.exists():
            if sys.platform == 'win32':
                return cls.VENV_DIR / 'Scripts' / 'python.exe'
            return cls.VENV_DIR / 'bin' / 'python'
        return None

    @classmethod
    def get_model_path(cls) -> Path:
        """Get path to the FILM TorchScript model."""
        return cls.MODEL_CACHE_DIR / cls.MODEL_FILENAME

    @classmethod
    def is_setup(cls) -> bool:
        """Check if venv exists and FILM model is downloaded."""
        python = cls.get_venv_python()
        if not python or not python.exists():
            return False
        # Check if model is downloaded
        return cls.get_model_path().exists()

    @classmethod
    def setup_film(cls, progress_callback=None, cancelled_check=None) -> bool:
        """Download FILM model and ensure venv is ready.

        Args:
            progress_callback: Optional callback(message, percent) for progress.
            cancelled_check: Optional callable that returns True if cancelled.

        Returns:
            True if setup was successful.
        """
        python = cls.get_venv_python()
        if not python or not python.exists():
            # Need to set up base venv first via PracticalRifeEnv
            return False

        try:
            model_path = cls.get_model_path()

            if not model_path.exists():
                if progress_callback:
                    progress_callback("Downloading FILM model (~380MB)...", 30)
                if cancelled_check and cancelled_check():
                    return False

                # Download the pre-trained TorchScript model
                cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(cls.MODEL_URL, model_path)

            if progress_callback:
                progress_callback("FILM setup complete!", 100)

            return cls.is_setup()

        except Exception as e:
            print(f"[FILM] Setup error: {e}", file=sys.stderr)
            return False

    @classmethod
    def get_worker_script(cls) -> Path:
        """Get path to the FILM worker script."""
        return Path(__file__).parent / 'film_worker.py'

    @classmethod
    def run_interpolation(
        cls,
        img_a_path: Path,
        img_b_path: Path,
        output_path: Path,
        t: float
    ) -> tuple[bool, str]:
        """Run FILM interpolation via subprocess in venv.

        Args:
            img_a_path: Path to first input image.
            img_b_path: Path to second input image.
            output_path: Path to output image.
            t: Timestep for interpolation (0.0 to 1.0).

        Returns:
            Tuple of (success, error_message).
        """
        python = cls.get_venv_python()
        if not python or not python.exists():
            return False, "venv python not found"

        script = cls.get_worker_script()
        if not script.exists():
            return False, f"worker script not found: {script}"

        cmd = [
            str(python), str(script),
            '--input0', str(img_a_path),
            '--input1', str(img_b_path),
            '--output', str(output_path),
            '--timestep', str(t),
            '--repo-dir', str(cls.REPO_DIR),
            '--model-dir', str(cls.MODEL_CACHE_DIR)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout per frame (FILM is slower)
            )
            if result.returncode == 0 and output_path.exists():
                return True, ""
            else:
                error = result.stderr.strip() if result.stderr else f"returncode={result.returncode}"
                return False, error
        except subprocess.TimeoutExpired:
            return False, "timeout (180s)"
        except Exception as e:
            return False, str(e)

    @classmethod
    def run_batch_interpolation(
        cls,
        img_a_path: Path,
        img_b_path: Path,
        output_dir: Path,
        frame_count: int,
        output_pattern: str = 'frame_{:04d}.png'
    ) -> tuple[bool, str, list[Path]]:
        """Run FILM batch interpolation via subprocess in venv.

        Generates all frames at once using FILM's recursive approach,
        which produces better results than generating frames independently.

        Args:
            img_a_path: Path to first input image.
            img_b_path: Path to second input image.
            output_dir: Directory to save output frames.
            frame_count: Number of frames to generate.
            output_pattern: Filename pattern for output frames.

        Returns:
            Tuple of (success, error_message, list_of_output_paths).
        """
        python = cls.get_venv_python()
        if not python or not python.exists():
            return False, "venv python not found", []

        script = cls.get_worker_script()
        if not script.exists():
            return False, f"worker script not found: {script}", []

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(python), str(script),
            '--input0', str(img_a_path),
            '--input1', str(img_b_path),
            '--output-dir', str(output_dir),
            '--frame-count', str(frame_count),
            '--output-pattern', output_pattern,
            '--repo-dir', str(cls.REPO_DIR),
            '--model-dir', str(cls.MODEL_CACHE_DIR)
        ]

        try:
            # Longer timeout for batch - scale with frame count
            timeout = max(300, frame_count * 30)  # At least 5 min, +30s per frame

            print(f"[FILM] Running batch interpolation: {frame_count} frames", file=sys.stderr)
            print(f"[FILM] Command: {' '.join(cmd)}", file=sys.stderr)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Collect output paths
            output_paths = [
                output_dir / output_pattern.format(i)
                for i in range(frame_count)
            ]
            existing_paths = [p for p in output_paths if p.exists()]

            if result.returncode == 0 and len(existing_paths) == frame_count:
                print(f"[FILM] Success: generated {len(existing_paths)} frames", file=sys.stderr)
                return True, "", output_paths
            else:
                # Combine stdout and stderr for better error reporting
                error_parts = []
                if result.returncode != 0:
                    error_parts.append(f"returncode={result.returncode}")
                if result.stdout and result.stdout.strip():
                    error_parts.append(f"stdout: {result.stdout.strip()}")
                if result.stderr and result.stderr.strip():
                    error_parts.append(f"stderr: {result.stderr.strip()}")
                if len(existing_paths) != frame_count:
                    error_parts.append(f"expected {frame_count} frames, got {len(existing_paths)}")

                error = "; ".join(error_parts) if error_parts else "unknown error"
                print(f"[FILM] Failed: {error}", file=sys.stderr)
                return False, error, existing_paths

        except subprocess.TimeoutExpired:
            print(f"[FILM] Timeout after {timeout}s", file=sys.stderr)
            return False, f"timeout ({timeout}s)", []
        except Exception as e:
            print(f"[FILM] Exception: {e}", file=sys.stderr)
            return False, str(e), []


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
    def optical_flow_blend(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2
    ) -> Image.Image:
        """Blend using OpenCV optical flow for motion compensation.

        Uses Farneback dense optical flow to warp frames and reduce ghosting
        artifacts compared to simple alpha blending.

        Args:
            img_a: First PIL Image (source frame).
            img_b: Second PIL Image (target frame).
            t: Interpolation factor 0.0 (100% A) to 1.0 (100% B).
            levels: Pyramid levels for optical flow (1-7).
            winsize: Window size for optical flow (5-51, odd).
            iterations: Number of iterations (1-10).
            poly_n: Polynomial neighborhood size (5 or 7).
            poly_sigma: Gaussian sigma for polynomial expansion (0.5-2.0).

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
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
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
        auto_download: bool = True,
        model: str = 'rife-v4.6',
        uhd: bool = False,
        tta: bool = False
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
            model: RIFE model to use (e.g., 'rife-v4.6', 'rife-anime').
            uhd: Enable UHD mode for high resolution images.
            tta: Enable TTA mode for better quality (slower).

        Returns:
            AI-interpolated blended PIL Image.
        """
        # Try NCNN binary first (specified path)
        if binary_path and binary_path.exists():
            result = ImageBlender._rife_ncnn(img_a, img_b, t, binary_path, model, uhd, tta)
            if result is not None:
                return result

        # Try to find rife-ncnn-vulkan in PATH
        ncnn_path = shutil.which('rife-ncnn-vulkan')
        if ncnn_path:
            result = ImageBlender._rife_ncnn(img_a, img_b, t, Path(ncnn_path), model, uhd, tta)
            if result is not None:
                return result

        # Try cached binary
        cached = RifeDownloader.get_cached_binary()
        if cached:
            result = ImageBlender._rife_ncnn(img_a, img_b, t, cached, model, uhd, tta)
            if result is not None:
                return result

        # Auto-download if enabled
        if auto_download:
            downloaded = RifeDownloader.ensure_binary()
            if downloaded:
                result = ImageBlender._rife_ncnn(img_a, img_b, t, downloaded, model, uhd, tta)
                if result is not None:
                    return result

        # Fall back to optical flow if RIFE not available
        return ImageBlender.optical_flow_blend(img_a, img_b, t)

    @staticmethod
    def _rife_ncnn(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float,
        binary: Path,
        model: str = 'rife-v4.6',
        uhd: bool = False,
        tta: bool = False
    ) -> Optional[Image.Image]:
        """Use rife-ncnn-vulkan binary for interpolation.

        Args:
            img_a: First PIL Image.
            img_b: Second PIL Image.
            t: Interpolation timestep (0.0 to 1.0).
            binary: Path to rife-ncnn-vulkan binary.
            model: RIFE model to use.
            uhd: Enable UHD mode.
            tta: Enable TTA mode.

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

                # Add model path (models are in same directory as binary)
                model_path = binary.parent / model
                if model_path.exists():
                    cmd.extend(['-m', str(model_path)])

                # Add UHD mode flag
                if uhd:
                    cmd.append('-u')

                # Add TTA mode flag (spatial)
                if tta:
                    cmd.append('-x')

                # Some versions support -s for timestep
                # Try with timestep first, fall back to simple interpolation
                try:
                    result = subprocess.run(
                        cmd + ['-s', str(t)],
                        check=True,
                        capture_output=True,
                        timeout=60  # Increased timeout for TTA mode
                    )
                except subprocess.CalledProcessError:
                    # Try without timestep (generates middle frame at t=0.5)
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=60
                    )

                if output_file.exists():
                    return Image.open(output_file).copy()

        except (subprocess.SubprocessError, OSError, IOError):
            pass

        return None

    @staticmethod
    def practical_rife_blend(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float,
        model: str = 'v4.25',
        ensemble: bool = False
    ) -> Image.Image:
        """Blend using Practical-RIFE Python/PyTorch implementation.

        Runs RIFE interpolation via subprocess in an isolated venv.
        Falls back to ncnn RIFE or optical flow if unavailable.

        Args:
            img_a: First PIL Image (source frame).
            img_b: Second PIL Image (target frame).
            t: Interpolation factor 0.0 (100% A) to 1.0 (100% B).
            model: Practical-RIFE model version (e.g., 'v4.25', 'v4.26').
            ensemble: Enable ensemble mode for better quality (slower).

        Returns:
            AI-interpolated blended PIL Image.
        """
        if not PracticalRifeEnv.is_setup():
            print("[Practical-RIFE] Venv not set up, falling back to ncnn RIFE", file=sys.stderr)
            return ImageBlender.rife_blend(img_a, img_b, t)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_a = tmp / 'a.png'
                input_b = tmp / 'b.png'
                output_file = tmp / 'out.png'

                # Save input images
                img_a.convert('RGB').save(input_a)
                img_b.convert('RGB').save(input_b)

                # Run Practical-RIFE via subprocess
                success, error_msg = PracticalRifeEnv.run_interpolation(
                    input_a, input_b, output_file, t, model, ensemble
                )

                if success and output_file.exists():
                    return Image.open(output_file).copy()
                else:
                    print(f"[Practical-RIFE] Interpolation failed: {error_msg}, falling back to ncnn RIFE", file=sys.stderr)

        except Exception as e:
            print(f"[Practical-RIFE] Exception: {e}, falling back to ncnn RIFE", file=sys.stderr)

        # Fall back to ncnn RIFE or optical flow
        return ImageBlender.rife_blend(img_a, img_b, t)

    @staticmethod
    def film_blend(
        img_a: Image.Image,
        img_b: Image.Image,
        t: float
    ) -> Image.Image:
        """Blend using FILM for large motion interpolation.

        FILM (Frame Interpolation for Large Motion) is Google Research's
        high-quality frame interpolation model, better for large motion.

        Args:
            img_a: First PIL Image (source frame).
            img_b: Second PIL Image (target frame).
            t: Interpolation factor 0.0 (100% A) to 1.0 (100% B).

        Returns:
            AI-interpolated blended PIL Image.
        """
        if not FilmEnv.is_setup():
            print("[FILM] Not set up, falling back to Practical-RIFE", file=sys.stderr)
            return ImageBlender.practical_rife_blend(img_a, img_b, t)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_a = tmp / 'a.png'
                input_b = tmp / 'b.png'
                output_file = tmp / 'out.png'

                # Save input images
                img_a.convert('RGB').save(input_a)
                img_b.convert('RGB').save(input_b)

                # Run FILM via subprocess
                success, error_msg = FilmEnv.run_interpolation(
                    input_a, input_b, output_file, t
                )

                if success and output_file.exists():
                    return Image.open(output_file).copy()
                else:
                    print(f"[FILM] Interpolation failed: {error_msg}, falling back to Practical-RIFE", file=sys.stderr)

        except Exception as e:
            print(f"[FILM] Exception: {e}, falling back to Practical-RIFE", file=sys.stderr)

        # Fall back to Practical-RIFE
        return ImageBlender.practical_rife_blend(img_a, img_b, t)

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
        rife_binary_path: Optional[Path] = None,
        rife_model: str = 'rife-v4.6',
        rife_uhd: bool = False,
        rife_tta: bool = False,
        practical_rife_model: str = 'v4.25',
        practical_rife_ensemble: bool = False,
        of_levels: int = 3,
        of_winsize: int = 15,
        of_iterations: int = 3,
        of_poly_n: int = 5,
        of_poly_sigma: float = 1.2
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
            blend_method: The blending method to use (alpha, optical_flow, rife, rife_practical).
            rife_binary_path: Optional path to rife-ncnn-vulkan binary.
            rife_model: RIFE ncnn model to use (e.g., 'rife-v4.6').
            rife_uhd: Enable RIFE ncnn UHD mode.
            rife_tta: Enable RIFE ncnn TTA mode.
            practical_rife_model: Practical-RIFE model version (e.g., 'v4.25').
            practical_rife_ensemble: Enable Practical-RIFE ensemble mode.
            of_levels: Optical flow pyramid levels (1-7).
            of_winsize: Optical flow window size (5-51, odd).
            of_iterations: Optical flow iterations (1-10).
            of_poly_n: Optical flow polynomial neighborhood (5 or 7).
            of_poly_sigma: Optical flow gaussian sigma (0.5-2.0).

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
                blended = ImageBlender.optical_flow_blend(
                    img_a, img_b, factor,
                    levels=of_levels,
                    winsize=of_winsize,
                    iterations=of_iterations,
                    poly_n=of_poly_n,
                    poly_sigma=of_poly_sigma
                )
            elif blend_method == BlendMethod.RIFE:
                blended = ImageBlender.rife_blend(
                    img_a, img_b, factor, rife_binary_path, True, rife_model, rife_uhd, rife_tta
                )
            elif blend_method == BlendMethod.RIFE_PRACTICAL:
                blended = ImageBlender.practical_rife_blend(
                    img_a, img_b, factor, practical_rife_model, practical_rife_ensemble
                )
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
        rife_binary_path: Optional[Path] = None,
        rife_model: str = 'rife-v4.6',
        rife_uhd: bool = False,
        rife_tta: bool = False,
        practical_rife_model: str = 'v4.25',
        practical_rife_ensemble: bool = False,
        of_levels: int = 3,
        of_winsize: int = 15,
        of_iterations: int = 3,
        of_poly_n: int = 5,
        of_poly_sigma: float = 1.2
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
            blend_method: The blending method to use (alpha, optical_flow, rife, rife_practical).
            rife_binary_path: Optional path to rife-ncnn-vulkan binary.
            rife_model: RIFE ncnn model to use (e.g., 'rife-v4.6').
            rife_uhd: Enable RIFE ncnn UHD mode.
            rife_tta: Enable RIFE ncnn TTA mode.
            practical_rife_model: Practical-RIFE model version (e.g., 'v4.25').
            practical_rife_ensemble: Enable Practical-RIFE ensemble mode.
            of_levels: Optical flow pyramid levels (1-7).
            of_winsize: Optical flow window size (5-51, odd).
            of_iterations: Optical flow iterations (1-10).
            of_poly_n: Optical flow polynomial neighborhood (5 or 7).
            of_poly_sigma: Optical flow gaussian sigma (0.5-2.0).

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
                blended = ImageBlender.optical_flow_blend(
                    img_a, img_b, factor,
                    levels=of_levels,
                    winsize=of_winsize,
                    iterations=of_iterations,
                    poly_n=of_poly_n,
                    poly_sigma=of_poly_sigma
                )
            elif blend_method == BlendMethod.RIFE:
                blended = ImageBlender.rife_blend(
                    img_a, img_b, factor, rife_binary_path, True, rife_model, rife_uhd, rife_tta
                )
            elif blend_method == BlendMethod.RIFE_PRACTICAL:
                blended = ImageBlender.practical_rife_blend(
                    img_a, img_b, factor, practical_rife_model, practical_rife_ensemble
                )
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
        overrides: Optional[dict[int, FolderType]] = None,
    ) -> FolderType:
        """Determine folder type based on position or override.

        Args:
            index: 0-based position of folder in list.
            overrides: Optional dict of position index to FolderType overrides.

        Returns:
            FolderType.MAIN for even positions (0, 2, 4...), TRANSITION for odd.
        """
        if overrides and index in overrides:
            override = overrides[index]
            if override != FolderType.AUTO:
                return override

        # Position-based: index 0, 2, 4... are MAIN; 1, 3, 5... are TRANSITION
        return FolderType.MAIN if index % 2 == 0 else FolderType.TRANSITION

    def identify_transition_boundaries(
        self,
        folders: list[Path],
        files_by_idx: dict[int, list[str]],
        folder_overrides: Optional[dict[int, FolderType]] = None,
        per_transition_settings: Optional[dict[int, PerTransitionSettings]] = None
    ) -> list[TransitionSpec]:
        """Identify boundaries where transitions should occur.

        Transitions happen at boundaries where folder types change
        (MAIN->TRANSITION or TRANSITION->MAIN).

        Args:
            folders: List of folders in order.
            files_by_idx: Dict mapping position index to file lists.
            folder_overrides: Optional position-index-keyed folder type overrides.
            per_transition_settings: Optional position-index-keyed per-transition overlap settings.

        Returns:
            List of TransitionSpec objects describing each transition.
        """
        if len(folders) < 2:
            return []

        transitions = []
        cumulative_idx = 0
        folder_start_indices: dict[int, int] = {}

        # Calculate start indices for each folder position
        for i in range(len(folders)):
            folder_start_indices[i] = cumulative_idx
            cumulative_idx += len(files_by_idx.get(i, []))

        # Look for transition boundaries (MAIN->TRANSITION and TRANSITION->MAIN)
        for i in range(len(folders) - 1):
            folder_a = folders[i]
            folder_b = folders[i + 1]

            type_a = self.get_folder_type(i, folder_overrides)
            type_b = self.get_folder_type(i + 1, folder_overrides)

            # Create transition when types differ (MAIN->TRANS or TRANS->MAIN)
            if type_a != type_b:
                files_a = files_by_idx.get(i, [])
                files_b = files_by_idx.get(i + 1, [])

                if not files_a or not files_b:
                    continue

                # Get per-transition overlap settings if available
                # Use i+1 as the key (the "incoming" folder position)
                if per_transition_settings and (i + 1) in per_transition_settings:
                    pts = per_transition_settings[i + 1]
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
                    main_start_idx=folder_start_indices[i],
                    trans_start_idx=folder_start_indices[i + 1],
                    main_folder_idx=i,
                    trans_folder_idx=i + 1,
                ))

        return transitions

    def generate_asymmetric_blend_frames(
        self,
        spec: TransitionSpec,
        dest: Path,
        folder_idx_main: int,
        base_seq_num: int
    ) -> list[BlendResult]:
        """Generate blended frames for an asymmetric transition.

        For asymmetric overlap, left_overlap != right_overlap. The blend
        creates max(left, right) output frames, with source frames interpolated
        to match the longer sequence.

        Args:
            spec: TransitionSpec describing the transition.
            dest: Destination directory for blended frames.
            folder_idx_main: Folder index (unused, kept for compatibility).
            base_seq_num: Starting sequence number for continuous naming.

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
            seq_num = base_seq_num + i
            output_name = f"seq_{seq_num:05d}{ext}"
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
                self.settings.rife_binary_path,
                self.settings.rife_model,
                self.settings.rife_uhd,
                self.settings.rife_tta,
                self.settings.practical_rife_model,
                self.settings.practical_rife_ensemble,
                self.settings.of_levels,
                self.settings.of_winsize,
                self.settings.of_iterations,
                self.settings.of_poly_n,
                self.settings.of_poly_sigma
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
        base_seq_num: int
    ) -> list[BlendResult]:
        """Generate blended frames for a transition.

        Uses asymmetric blend if left_overlap != right_overlap.

        Args:
            spec: TransitionSpec describing the transition.
            dest: Destination directory for blended frames.
            folder_idx_main: Folder index (unused, kept for compatibility).
            base_seq_num: Starting sequence number for continuous naming.

        Returns:
            List of BlendResult objects.
        """
        # Use asymmetric blend for all cases (handles symmetric too)
        return self.generate_asymmetric_blend_frames(
            spec, dest, folder_idx_main, base_seq_num
        )

    def generate_direct_interpolation_frames(
        self,
        img_a_path: Path,
        img_b_path: Path,
        frame_count: int,
        method: DirectInterpolationMethod,
        dest: Path,
        folder_idx: int,
        base_seq_num: int,
        practical_rife_model: str = 'v4.25',
        practical_rife_ensemble: bool = False
    ) -> list[BlendResult]:
        """Generate AI-interpolated frames between two images.

        Used for direct transitions between MAIN sequences without
        a transition folder.

        For FILM: Uses batch mode to generate all frames at once (better quality).
        For RIFE: Generates frames one at a time (RIFE handles arbitrary timesteps well).

        Args:
            img_a_path: Path to last frame of first sequence.
            img_b_path: Path to first frame of second sequence.
            frame_count: Number of interpolated frames to generate.
            method: Interpolation method (RIFE or FILM).
            dest: Destination directory for generated frames.
            folder_idx: Folder index (unused, kept for compatibility).
            base_seq_num: Starting sequence number for continuous naming.
            practical_rife_model: Practical-RIFE model version.
            practical_rife_ensemble: Enable Practical-RIFE ensemble mode.

        Returns:
            List of BlendResult objects.
        """
        results = []
        dest.mkdir(parents=True, exist_ok=True)

        # For FILM, use batch mode to generate all frames at once
        if method == DirectInterpolationMethod.FILM and FilmEnv.is_setup():
            return self._generate_film_frames_batch(
                img_a_path, img_b_path, frame_count, dest, base_seq_num
            )

        # For RIFE (or FILM fallback), generate frames one at a time
        # Load source images
        img_a = Image.open(img_a_path)
        img_b = Image.open(img_b_path)

        # Handle different sizes - resize B to match A
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)

        # Normalize to RGBA
        if img_a.mode != 'RGBA':
            img_a = img_a.convert('RGBA')
        if img_b.mode != 'RGBA':
            img_b = img_b.convert('RGBA')

        for i in range(frame_count):
            # Evenly space t values between 0 and 1 (exclusive)
            t = (i + 1) / (frame_count + 1)

            # Generate interpolated frame
            if method == DirectInterpolationMethod.FILM:
                blended = ImageBlender.film_blend(img_a, img_b, t)
            else:  # RIFE
                blended = ImageBlender.practical_rife_blend(
                    img_a, img_b, t,
                    practical_rife_model, practical_rife_ensemble
                )

            # Generate output filename
            ext = f".{self.settings.output_format.lower()}"
            seq_num = base_seq_num + i
            output_name = f"seq_{seq_num:05d}{ext}"
            output_path = dest / output_name

            # Save the blended frame
            try:
                # Convert back to RGB if saving to JPEG
                if self.settings.output_format.lower() in ('jpg', 'jpeg'):
                    blended = blended.convert('RGB')

                # Save with appropriate options
                save_kwargs = {}
                if self.settings.output_format.lower() in ('jpg', 'jpeg'):
                    save_kwargs['quality'] = self.settings.output_quality
                elif self.settings.output_format.lower() == 'webp':
                    save_kwargs['lossless'] = True
                    save_kwargs['method'] = self.settings.webp_method
                elif self.settings.output_format.lower() == 'png':
                    save_kwargs['compress_level'] = 6

                blended.save(output_path, **save_kwargs)

                results.append(BlendResult(
                    output_path=output_path,
                    source_a=img_a_path,
                    source_b=img_b_path,
                    blend_factor=t,
                    success=True
                ))
            except Exception as e:
                results.append(BlendResult(
                    output_path=output_path,
                    source_a=img_a_path,
                    source_b=img_b_path,
                    blend_factor=t,
                    success=False,
                    error=str(e)
                ))

        # Close loaded images
        img_a.close()
        img_b.close()

        return results

    def _generate_film_frames_batch(
        self,
        img_a_path: Path,
        img_b_path: Path,
        frame_count: int,
        dest: Path,
        base_seq_num: int
    ) -> list[BlendResult]:
        """Generate FILM frames using batch mode for better quality.

        FILM works best when generating all frames at once using its
        recursive approach, rather than generating arbitrary timesteps.

        Args:
            img_a_path: Path to last frame of first sequence.
            img_b_path: Path to first frame of second sequence.
            frame_count: Number of interpolated frames to generate.
            dest: Destination directory for generated frames.
            base_seq_num: Starting sequence number for continuous naming.

        Returns:
            List of BlendResult objects.
        """
        results = []

        # Generate frames using FILM batch mode
        # Use a temp pattern, then rename to final names
        temp_pattern = 'film_temp_{:04d}.png'

        success, error, temp_paths = FilmEnv.run_batch_interpolation(
            img_a_path,
            img_b_path,
            dest,
            frame_count,
            temp_pattern
        )

        if not success:
            # Return error results for all frames
            for i in range(frame_count):
                t = (i + 1) / (frame_count + 1)
                ext = f".{self.settings.output_format.lower()}"
                seq_num = base_seq_num + i
                output_name = f"seq_{seq_num:05d}{ext}"
                output_path = dest / output_name

                results.append(BlendResult(
                    output_path=output_path,
                    source_a=img_a_path,
                    source_b=img_b_path,
                    blend_factor=t,
                    success=False,
                    error=error
                ))
            return results

        # Rename temp files to final names and convert format if needed
        for i, temp_path in enumerate(temp_paths):
            t = (i + 1) / (frame_count + 1)
            ext = f".{self.settings.output_format.lower()}"
            seq_num = base_seq_num + i
            output_name = f"seq_{seq_num:05d}{ext}"
            output_path = dest / output_name

            try:
                if temp_path.exists():
                    # Load the temp frame
                    frame = Image.open(temp_path)

                    # Convert format if needed
                    if self.settings.output_format.lower() in ('jpg', 'jpeg'):
                        frame = frame.convert('RGB')

                    # Save with appropriate options
                    save_kwargs = {}
                    if self.settings.output_format.lower() in ('jpg', 'jpeg'):
                        save_kwargs['quality'] = self.settings.output_quality
                    elif self.settings.output_format.lower() == 'webp':
                        save_kwargs['lossless'] = True
                        save_kwargs['method'] = self.settings.webp_method
                    elif self.settings.output_format.lower() == 'png':
                        save_kwargs['compress_level'] = 6

                    frame.save(output_path, **save_kwargs)
                    frame.close()

                    # Remove temp file if different from output
                    if temp_path != output_path:
                        temp_path.unlink(missing_ok=True)

                    results.append(BlendResult(
                        output_path=output_path,
                        source_a=img_a_path,
                        source_b=img_b_path,
                        blend_factor=t,
                        success=True
                    ))
                else:
                    results.append(BlendResult(
                        output_path=output_path,
                        source_a=img_a_path,
                        source_b=img_b_path,
                        blend_factor=t,
                        success=False,
                        error=f"Temp file not found: {temp_path}"
                    ))
            except Exception as e:
                results.append(BlendResult(
                    output_path=output_path,
                    source_a=img_a_path,
                    source_b=img_b_path,
                    blend_factor=t,
                    success=False,
                    error=str(e)
                ))

        return results
