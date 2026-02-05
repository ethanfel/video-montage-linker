#!/usr/bin/env python
"""FILM interpolation worker - runs in isolated venv with PyTorch.

This script is executed via subprocess from the main application.
It handles frame interpolation using Google Research's FILM model
(Frame Interpolation for Large Motion) via the frame-interpolation-pytorch repo.

FILM is better than RIFE for large motion and scene gaps, but slower.

Supports two modes:
1. Single frame: --output with --timestep
2. Batch mode: --output-dir with --frame-count (generates all frames at once)
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Model download URL
FILM_MODEL_URL = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt"
FILM_MODEL_FILENAME = "film_net_fp32.pt"


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """Load image as tensor.

    Args:
        path: Path to image file.
        device: Device to load tensor to.

    Returns:
        Image tensor (1, 3, H, W) normalized to [0, 1].
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor as image.

    Args:
        tensor: Image tensor (1, 3, H, W) or (3, H, W) normalized to [0, 1].
        path: Output path.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# Global model cache
_model_cache: dict = {}


def download_model(model_dir: Path) -> Path:
    """Download FILM model if not present.

    Args:
        model_dir: Directory to store the model.

    Returns:
        Path to the downloaded model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / FILM_MODEL_FILENAME

    if not model_path.exists():
        print(f"Downloading FILM model to {model_path}...", file=sys.stderr)
        urllib.request.urlretrieve(FILM_MODEL_URL, model_path)
        print("Download complete.", file=sys.stderr)

    return model_path


def get_model(model_dir: Path, device: torch.device):
    """Get or load FILM model (cached).

    Args:
        model_dir: Model cache directory (for model downloads).
        device: Device to run on.

    Returns:
        FILM TorchScript model instance.
    """
    cache_key = f"film_{device}"
    if cache_key not in _model_cache:
        # Download model if needed
        model_path = download_model(model_dir)

        # Load pre-trained TorchScript model
        print(f"Loading FILM model from {model_path}...", file=sys.stderr)
        model = torch.jit.load(str(model_path), map_location='cpu')
        model.eval()
        model.to(device)
        _model_cache[cache_key] = model
        print("Model loaded.", file=sys.stderr)

    return _model_cache[cache_key]


@torch.no_grad()
def interpolate_single(model, img0: torch.Tensor, img1: torch.Tensor, t: float) -> torch.Tensor:
    """Perform single frame interpolation using FILM.

    Args:
        model: FILM TorchScript model instance.
        img0: First frame tensor (1, 3, H, W) normalized to [0, 1].
        img1: Second frame tensor (1, 3, H, W) normalized to [0, 1].
        t: Interpolation timestep (0.0 to 1.0).

    Returns:
        Interpolated frame tensor.
    """
    # FILM TorchScript model expects dt as tensor of shape (1, 1)
    dt = img0.new_full((1, 1), t)

    result = model(img0, img1, dt)

    if isinstance(result, tuple):
        result = result[0]

    return result.clamp(0, 1)


@torch.no_grad()
def interpolate_batch(model, img0: torch.Tensor, img1: torch.Tensor, frame_count: int) -> list[torch.Tensor]:
    """Generate multiple interpolated frames using FILM's recursive approach.

    FILM works best when generating frames recursively - it first generates
    the middle frame, then fills in the gaps. This produces more consistent
    results than generating arbitrary timesteps independently.

    Args:
        model: FILM model instance.
        img0: First frame tensor (1, 3, H, W) normalized to [0, 1].
        img1: Second frame tensor (1, 3, H, W) normalized to [0, 1].
        frame_count: Number of frames to generate between img0 and img1.

    Returns:
        List of interpolated frame tensors in order.
    """
    # Calculate timesteps for evenly spaced frames
    timesteps = [(i + 1) / (frame_count + 1) for i in range(frame_count)]

    # Try to use the model's batch/recursive interpolation if available
    try:
        # Some implementations have an interpolate_recursively method
        if hasattr(model, 'interpolate_recursively'):
            # This generates 2^n - 1 frames, so we need to handle arbitrary counts
            results = model.interpolate_recursively(img0, img1, frame_count)
            if len(results) >= frame_count:
                return results[:frame_count]
    except (AttributeError, TypeError):
        pass

    # Fall back to recursive binary interpolation for better quality
    # This mimics FILM's natural recursive approach
    frames = {}  # timestep -> tensor

    def recursive_interpolate(t_left: float, t_right: float, img_left: torch.Tensor, img_right: torch.Tensor, depth: int = 0):
        """Recursively interpolate to fill the gap."""
        if depth > 10:  # Prevent infinite recursion
            return

        t_mid = (t_left + t_right) / 2

        # Check if we need a frame near t_mid
        need_frame = False
        for t in timesteps:
            if t not in frames and abs(t - t_mid) < 0.5 / (frame_count + 1):
                need_frame = True
                break

        if not need_frame:
            # Check if any remaining timesteps are in this range
            remaining = [t for t in timesteps if t not in frames and t_left < t < t_right]
            if not remaining:
                return

        # Generate middle frame
        mid_frame = interpolate_single(model, img_left, img_right, 0.5)

        # Assign to nearest needed timestep
        for t in timesteps:
            if t not in frames and abs(t - t_mid) < 0.5 / (frame_count + 1):
                frames[t] = mid_frame
                break

        # Recurse into left and right halves
        recursive_interpolate(t_left, t_mid, img_left, mid_frame, depth + 1)
        recursive_interpolate(t_mid, t_right, mid_frame, img_right, depth + 1)

    # Start recursive interpolation
    recursive_interpolate(0.0, 1.0, img0, img1)

    # Fill any remaining timesteps with direct interpolation
    for t in timesteps:
        if t not in frames:
            frames[t] = interpolate_single(model, img0, img1, t)

    # Return frames in order
    return [frames[t] for t in timesteps]


def main():
    parser = argparse.ArgumentParser(description='FILM frame interpolation worker')
    parser.add_argument('--input0', required=True, help='Path to first input image')
    parser.add_argument('--input1', required=True, help='Path to second input image')
    parser.add_argument('--output', help='Path to output image (single frame mode)')
    parser.add_argument('--output-dir', help='Output directory (batch mode)')
    parser.add_argument('--output-pattern', default='frame_{:04d}.png',
                        help='Output filename pattern for batch mode')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Interpolation timestep 0-1 (single frame mode)')
    parser.add_argument('--frame-count', type=int,
                        help='Number of frames to generate (batch mode)')
    parser.add_argument('--repo-dir', help='Unused (kept for backward compat)')
    parser.add_argument('--model-dir', required=True, help='Model cache directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    # Validate arguments
    batch_mode = args.output_dir is not None and args.frame_count is not None
    single_mode = args.output is not None

    if not batch_mode and not single_mode:
        print("Error: Must specify either --output (single) or --output-dir + --frame-count (batch)",
              file=sys.stderr)
        return 1

    try:
        # Select device
        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load model
        model_dir = Path(args.model_dir)
        model = get_model(model_dir, device)

        # Load images
        img0 = load_image(Path(args.input0), device)
        img1 = load_image(Path(args.input1), device)

        if batch_mode:
            # Batch mode - generate all frames at once
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Generating {args.frame_count} frames...", file=sys.stderr)
            frames = interpolate_batch(model, img0, img1, args.frame_count)

            for i, frame in enumerate(frames):
                output_path = output_dir / args.output_pattern.format(i)
                save_image(frame, output_path)
                print(f"Saved {output_path.name}", file=sys.stderr)

            print(f"Success: Generated {len(frames)} frames", file=sys.stderr)
        else:
            # Single frame mode
            result = interpolate_single(model, img0, img1, args.timestep)
            save_image(result, Path(args.output))
            print("Success", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
