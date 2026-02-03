#!/usr/bin/env python
"""RIFE interpolation worker - runs in isolated venv with PyTorch.

This script is executed via subprocess from the main application.
It handles loading Practical-RIFE models and performing frame interpolation.

Note: The Practical-RIFE models require the IFNet architecture from the
Practical-RIFE repository. This script downloads and uses the model weights
with a simplified inference implementation.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1./scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1./scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale*2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    backwarp_tenGrid = {}
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


class IFNet(nn.Module):
    """IFNet architecture for RIFE v4.x models."""

    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+16, c=192)
        self.block1 = IFBlock(8+4+16, c=128)
        self.block2 = IFBlock(8+4+16, c=96)
        self.block3 = IFBlock(8+4+16, c=64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 4, 2, 1)
        )

    def forward(self, img0, img1, timestep=0.5, scale_list=[8, 4, 2, 1]):
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0 = block[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1),
                    flow, scale=scale_list[i])
                flow = flow + fd
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask_final = torch.sigmoid(mask)
        merged_final = warped_img0 * mask_final + warped_img1 * (1 - mask_final)
        return merged_final


# Model URLs for downloading
MODEL_URLS = {
    'v4.26': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.26/flownet.pkl',
    'v4.25': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.25/flownet.pkl',
    'v4.22': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.22/flownet.pkl',
    'v4.20': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.20/flownet.pkl',
    'v4.18': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.18/flownet.pkl',
    'v4.15': 'https://github.com/hzwer/Practical-RIFE/raw/main/train_log_v4.15/flownet.pkl',
}


def download_model(version: str, model_dir: Path) -> Path:
    """Download model if not already cached.

    Args:
        version: Model version (e.g., 'v4.25').
        model_dir: Directory to store models.

    Returns:
        Path to the downloaded model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'flownet_{version}.pkl'

    if model_path.exists():
        return model_path

    url = MODEL_URLS.get(version)
    if not url:
        raise ValueError(f"Unknown model version: {version}")

    print(f"Downloading RIFE model {version}...", file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'video-montage-linker'})
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(model_path, 'wb') as f:
                f.write(response.read())
        print(f"Model downloaded to {model_path}", file=sys.stderr)
        return model_path
    except Exception as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model: {e}")


def load_model(model_path: Path, device: torch.device) -> IFNet:
    """Load IFNet model from state dict.

    Args:
        model_path: Path to flownet.pkl file.
        device: Device to load model to.

    Returns:
        Loaded IFNet model.
    """
    model = IFNet()
    state_dict = torch.load(model_path, map_location='cpu')

    # Handle different state dict formats
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        # Handle flownet. prefix
        if k.startswith('flownet.'):
            k = k[8:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def pad_image(img: torch.Tensor, padding: int = 64) -> tuple:
    """Pad image to be divisible by padding.

    Args:
        img: Input tensor (B, C, H, W).
        padding: Padding divisor.

    Returns:
        Tuple of (padded image, (original H, original W)).
    """
    _, _, h, w = img.shape
    ph = ((h - 1) // padding + 1) * padding
    pw = ((w - 1) // padding + 1) * padding
    pad_h = ph - h
    pad_w = pw - w
    padded = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
    return padded, (h, w)


@torch.no_grad()
def inference(model: IFNet, img0: torch.Tensor, img1: torch.Tensor,
              timestep: float = 0.5, ensemble: bool = False) -> torch.Tensor:
    """Perform frame interpolation.

    Args:
        model: Loaded IFNet model.
        img0: First frame tensor (B, C, H, W) normalized to [0, 1].
        img1: Second frame tensor (B, C, H, W) normalized to [0, 1].
        timestep: Interpolation timestep (0.0 to 1.0).
        ensemble: Enable ensemble mode for better quality.

    Returns:
        Interpolated frame tensor.
    """
    # Pad images
    img0_padded, orig_size = pad_image(img0)
    img1_padded, _ = pad_image(img1)
    h, w = orig_size

    # Create timestep tensor
    timestep_tensor = torch.full((1, 1, img0_padded.shape[2], img0_padded.shape[3]),
                                  timestep, device=img0.device)

    if ensemble:
        # Ensemble: average of forward and reverse
        result1 = model(img0_padded, img1_padded, timestep_tensor)
        result2 = model(img1_padded, img0_padded, 1 - timestep_tensor)
        result = (result1 + result2) / 2
    else:
        result = model(img0_padded, img1_padded, timestep_tensor)

    # Crop back to original size
    result = result[:, :, :h, :w]
    return result.clamp(0, 1)


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
        tensor: Image tensor (1, 3, H, W) normalized to [0, 1].
        path: Output path.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# Global model cache
_model_cache: dict = {}


def get_model(version: str, model_dir: Path, device: torch.device) -> IFNet:
    """Get or load model (cached).

    Args:
        version: Model version.
        model_dir: Model cache directory.
        device: Device to run on.

    Returns:
        IFNet model instance.
    """
    cache_key = f"{version}_{device}"
    if cache_key not in _model_cache:
        model_path = download_model(version, model_dir)
        _model_cache[cache_key] = load_model(model_path, device)
    return _model_cache[cache_key]


def main():
    parser = argparse.ArgumentParser(description='RIFE frame interpolation worker')
    parser.add_argument('--input0', required=True, help='Path to first input image')
    parser.add_argument('--input1', required=True, help='Path to second input image')
    parser.add_argument('--output', required=True, help='Path to output image')
    parser.add_argument('--timestep', type=float, default=0.5, help='Interpolation timestep (0-1)')
    parser.add_argument('--model', default='v4.25', help='Model version')
    parser.add_argument('--model-dir', required=True, help='Model cache directory')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    try:
        # Select device
        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load model
        model_dir = Path(args.model_dir)
        model = get_model(args.model, model_dir, device)

        # Load images
        img0 = load_image(Path(args.input0), device)
        img1 = load_image(Path(args.input1), device)

        # Interpolate
        result = inference(model, img0, img1, args.timestep, args.ensemble)

        # Save result
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
