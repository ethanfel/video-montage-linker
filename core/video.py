"""Video encoding utilities wrapping ffmpeg."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .models import VideoPreset


def find_ffmpeg() -> Optional[Path]:
    """Find the ffmpeg binary on the system PATH."""
    result = shutil.which('ffmpeg')
    return Path(result) if result else None


def encode_image_sequence(
    input_dir: Path,
    output_path: Path,
    fps: int,
    preset: VideoPreset,
    input_pattern: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    total_frames: Optional[int] = None,
) -> tuple[bool, str]:
    """Encode an image sequence directory to a video file using ffmpeg.

    Args:
        input_dir: Directory containing sequentially named image files.
        output_path: Output video file path.
        fps: Frames per second.
        preset: VideoPreset with codec settings.
        input_pattern: ffmpeg input pattern (e.g. 'seq_%06d.png').
            Auto-detected from first seq_* file if not provided.
        progress_callback: Called with (current_frame, total_frames).
            Return False to cancel encoding.
        total_frames: Total number of frames for progress reporting.
            Auto-counted from input_dir if not provided.

    Returns:
        (success, message) — message is output_path on success or error text on failure.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found. Install ffmpeg to encode video."

    # Auto-detect input pattern from first seq_* file
    if input_pattern is None:
        input_pattern = _detect_input_pattern(input_dir)
        if input_pattern is None:
            return False, f"No seq_* image files found in {input_dir}"

    # Auto-count frames
    if total_frames is None:
        ext = Path(input_pattern).suffix
        total_frames = len(list(input_dir.glob(f"seq_*{ext}")))
        if total_frames == 0:
            return False, f"No matching frames found in {input_dir}"

    # Build ffmpeg command
    cmd = [
        str(ffmpeg), '-y',
        '-framerate', str(fps),
        '-i', str(input_dir / input_pattern),
        '-c:v', preset.codec,
        '-q:v' if preset.codec == 'libtheora' else '-crf', str(preset.crf),
        '-pix_fmt', preset.pixel_format,
    ]

    # Add speed preset for x264/x265
    if preset.codec in ('libx264', 'libx265'):
        cmd += ['-preset', preset.preset]

    # Add downscale filter if max_height is set
    if preset.max_height is not None:
        cmd += ['-vf', f'scale=-2:{preset.max_height}']

    # Add any extra codec-specific args
    if preset.extra_args:
        cmd += preset.extra_args

    # Progress parsing via -progress pipe:1
    cmd += ['-progress', 'pipe:1']

    cmd.append(str(output_path))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        cancelled = False
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                m = re.match(r'^frame=(\d+)', line)
                if m and progress_callback is not None:
                    current = int(m.group(1))
                    if not progress_callback(current, total_frames):
                        cancelled = True
                        proc.terminate()
                        proc.wait()
                        break

        proc.wait()

        if cancelled:
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            return False, "Encoding cancelled by user."

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            return False, f"ffmpeg exited with code {proc.returncode}:\n{stderr}"

        return True, str(output_path)

    except FileNotFoundError:
        return False, "ffmpeg binary not found."
    except Exception as e:
        return False, f"Encoding error: {e}"


def _detect_input_pattern(input_dir: Path) -> Optional[str]:
    """Detect the ffmpeg input pattern from seq_* files in a directory.

    Looks for files like seq_000000.png and returns a pattern like seq_%06d.png.
    """
    for f in sorted(input_dir.iterdir()):
        m = re.match(r'^(seq_)(\d+)(\.\w+)$', f.name)
        if m:
            prefix = m.group(1)
            digits = m.group(2)
            ext = m.group(3)
            width = len(digits)
            return f"{prefix}%0{width}d{ext}"
    return None


def encode_from_file_list(
    file_paths: list[Path],
    output_path: Path,
    fps: int,
    preset: VideoPreset,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> tuple[bool, str]:
    """Encode a video from an explicit list of image file paths.

    Uses ffmpeg's concat demuxer so files can be scattered across directories.

    Args:
        file_paths: Ordered list of image file paths.
        output_path: Output video file path.
        fps: Frames per second.
        preset: VideoPreset with codec settings.
        progress_callback: Called with (current_frame, total_frames).
            Return False to cancel encoding.

    Returns:
        (success, message) — message is output_path on success or error text on failure.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found. Install ffmpeg to encode video."

    if not file_paths:
        return False, "No files provided."

    total_frames = len(file_paths)
    frame_duration = f"{1.0 / fps:.10f}"

    # Write a concat-demuxer file listing each image with its duration
    try:
        concat_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='vml_concat_'
        )
        concat_path = Path(concat_file.name)
        for p in file_paths:
            # Escape single quotes for ffmpeg concat format
            escaped = str(p.resolve()).replace("'", "'\\''")
            concat_file.write(f"file '{escaped}'\n")
            concat_file.write(f"duration {frame_duration}\n")
        # Repeat last file so the last frame displays for its full duration
        escaped = str(file_paths[-1].resolve()).replace("'", "'\\''")
        concat_file.write(f"file '{escaped}'\n")
        concat_file.close()
    except OSError as e:
        return False, f"Failed to create concat file: {e}"

    cmd = [
        str(ffmpeg), '-y',
        '-f', 'concat', '-safe', '0',
        '-i', str(concat_path),
        '-c:v', preset.codec,
        '-q:v' if preset.codec == 'libtheora' else '-crf', str(preset.crf),
        '-pix_fmt', preset.pixel_format,
    ]

    if preset.codec in ('libx264', 'libx265'):
        cmd += ['-preset', preset.preset]

    if preset.max_height is not None:
        cmd += ['-vf', f'scale=-2:{preset.max_height}']

    if preset.extra_args:
        cmd += preset.extra_args

    cmd += ['-progress', 'pipe:1']
    cmd.append(str(output_path))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        cancelled = False
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                m = re.match(r'^frame=(\d+)', line)
                if m and progress_callback is not None:
                    current = int(m.group(1))
                    if not progress_callback(current, total_frames):
                        cancelled = True
                        proc.terminate()
                        proc.wait()
                        break

        proc.wait()

        if cancelled:
            if output_path.exists():
                output_path.unlink()
            return False, "Encoding cancelled by user."

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            return False, f"ffmpeg exited with code {proc.returncode}:\n{stderr}"

        return True, str(output_path)

    except FileNotFoundError:
        return False, "ffmpeg binary not found."
    except Exception as e:
        return False, f"Encoding error: {e}"
    finally:
        try:
            concat_path.unlink(missing_ok=True)
        except OSError:
            pass
