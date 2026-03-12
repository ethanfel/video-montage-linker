#!/usr/bin/env python
"""BiM-VFI interpolation worker - runs in isolated venv with PyTorch.

This script is executed via subprocess from the main application.
It handles frame interpolation using KAIST VICLab's BiM-VFI model
(Bidirectional Motion Field-Guided Frame Interpolation).

BiM-VFI is designed for non-uniform motions and produces high-quality
results, especially for complex scenes (CVPR 2025).

Supports two modes:
1. Single frame: --output with --timestep
2. Batch mode: --output-dir with --frame-count (generates all frames at once)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Checkpoint filename
BIM_VFI_CHECKPOINT = "bim_vfi.pth"


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


_model_cache: dict = {}


def get_model(repo_dir: Path, model_dir: Path, device: torch.device):
    """Get or load BiM-VFI model (cached).

    Args:
        repo_dir: Path to the cloned BiM-VFI repository.
        model_dir: Directory containing the checkpoint.
        device: Device to run on.

    Returns:
        BiM-VFI model instance.
    """
    cache_key = f"bim_vfi_{device}"
    if cache_key not in _model_cache:
        # Add repo to sys.path so we can import BiM-VFI modules
        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        checkpoint_path = model_dir / BIM_VFI_CHECKPOINT

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"BiM-VFI checkpoint not found at {checkpoint_path}. "
                "Please download it from Google Drive and place it there."
            )

        print(f"Loading BiM-VFI model from {checkpoint_path}...", file=sys.stderr)

        # Import from the package __init__ so the @register decorator fires
        from modules.components import make_components

        # Create model with the trained config (pyr_level=3, feat_channels=32)
        cfg = {'name': 'bim_vfi', 'args': {'pyr_level': 3, 'feat_channels': 32}}
        model = make_components(cfg)

        # Load checkpoint
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])

        model.eval()
        model.to(device)
        _model_cache[cache_key] = model
        print("Model loaded.", file=sys.stderr)

    return _model_cache[cache_key]


@torch.no_grad()
def interpolate_single(
    model, img0: torch.Tensor, img1: torch.Tensor, t: float
) -> torch.Tensor:
    """Perform single frame interpolation using BiM-VFI.

    Uses the model's trained pyr_level (set at construction time).
    The model handles input padding internally via InputPadder.

    Args:
        model: BiM-VFI model instance.
        img0: First frame tensor (1, 3, H, W) normalized to [0, 1].
        img1: Second frame tensor (1, 3, H, W) normalized to [0, 1].
        t: Interpolation timestep (0.0 to 1.0).

    Returns:
        Interpolated frame tensor.
    """
    time_step = torch.tensor([t]).view(1, 1, 1, 1).to(img0.device)

    results_dict = model(
        img0=img0, img1=img1,
        time_step=time_step,
    )

    return results_dict['imgt_pred'].clamp(0, 1)


@torch.no_grad()
def interpolate_batch(
    model, img0: torch.Tensor, img1: torch.Tensor, frame_count: int
) -> list[torch.Tensor]:
    """Generate multiple interpolated frames using BiM-VFI.

    Args:
        model: BiM-VFI model instance.
        img0: First frame tensor (1, 3, H, W) normalized to [0, 1].
        img1: Second frame tensor (1, 3, H, W) normalized to [0, 1].
        frame_count: Number of frames to generate between img0 and img1.

    Returns:
        List of interpolated frame tensors in order.
    """
    frames = []
    for i in range(frame_count):
        t = (i + 1) / (frame_count + 1)
        frame = interpolate_single(model, img0, img1, t)
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description='BiM-VFI frame interpolation worker')
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
    parser.add_argument('--repo-dir', required=True, help='Path to BiM-VFI repo clone')
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
        repo_dir = Path(args.repo_dir)
        model_dir = Path(args.model_dir)
        model = get_model(repo_dir, model_dir, device)

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
