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
import shutil
import sys
import tempfile
import urllib.request
import zipfile
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
    """IFNet architecture for Practical-RIFE v4.25/v4.26 models."""

    def __init__(self):
        super(IFNet, self).__init__()
        # v4.25/v4.26 architecture:
        # block0 input: img0(3) + img1(3) + f0(4) + f1(4) + timestep(1) = 15
        # block1+ input: img0(3) + img1(3) + wf0(4) + wf1(4) + f0(4) + f1(4) + timestep(1) + mask(1) + flow(4) = 28
        self.block0 = IFBlock(3+3+4+4+1, c=192)
        self.block1 = IFBlock(3+3+4+4+4+4+1+1+4, c=128)
        self.block2 = IFBlock(3+3+4+4+4+4+1+1+4, c=96)
        self.block3 = IFBlock(3+3+4+4+4+4+1+1+4, c=64)
        # Encode produces 4-channel features
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1)
        )

    def forward(self, img0, img1, timestep=0.5, scale_list=[8, 4, 2, 1]):
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
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
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, f0, f1, timestep, mask), 1),
                    flow, scale=scale_list[i])
                flow = flow + fd
                mask = mask + m0
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        mask_final = torch.sigmoid(mask)
        merged_final = warped_img0 * mask_final + warped_img1 * (1 - mask_final)
        return merged_final


# Model URLs for downloading (Google Drive direct download links)
# File IDs extracted from official Practical-RIFE repository
MODEL_URLS = {
    'v4.26': 'https://drive.google.com/uc?export=download&id=1gViYvvQrtETBgU1w8axZSsr7YUuw31uy',
    'v4.25': 'https://drive.google.com/uc?export=download&id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg',
    'v4.22': 'https://drive.google.com/uc?export=download&id=1qh2DSA9a1eZUTtZG9U9RQKO7N7OaUJ0_',
    'v4.20': 'https://drive.google.com/uc?export=download&id=11n3YR7-qCRZm9RDdwtqOTsgCJUHPuexA',
    'v4.18': 'https://drive.google.com/uc?export=download&id=1octn-UVuEjXa_HlsIUbNeLTTvYCKbC_s',
    'v4.15': 'https://drive.google.com/uc?export=download&id=1xlem7cfKoMaiLzjoeum8KIQTYO-9iqG5',
}


def download_model(version: str, model_dir: Path) -> Path:
    """Download model if not already cached.

    Google Drive links distribute zip files containing the model.
    This function downloads and extracts the flownet.pkl file.

    Args:
        version: Model version (e.g., 'v4.25').
        model_dir: Directory to store models.

    Returns:
        Path to the downloaded model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'flownet_{version}.pkl'

    if model_path.exists():
        # Verify it's not a zip file (from previous failed attempt)
        with open(model_path, 'rb') as f:
            header = f.read(4)
        if header == b'PK\x03\x04':  # ZIP magic number
            print(f"Removing corrupted zip file at {model_path}", file=sys.stderr)
            model_path.unlink()
        else:
            return model_path

    url = MODEL_URLS.get(version)
    if not url:
        raise ValueError(f"Unknown model version: {version}")

    print(f"Downloading RIFE model {version}...", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / 'download'

        # Try using gdown for Google Drive (handles confirmations automatically)
        downloaded = False
        try:
            import gdown
            file_id = url.split('id=')[1] if 'id=' in url else None
            if file_id:
                gdown_url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(gdown_url, str(tmp_path), quiet=False)
                downloaded = tmp_path.exists()
        except ImportError:
            print("gdown not available, trying direct download...", file=sys.stderr)
        except Exception as e:
            print(f"gdown failed: {e}, trying direct download...", file=sys.stderr)

        # Fallback: direct download
        if not downloaded:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=300) as response:
                    data = response.read()
                    if data[:100].startswith(b'<!') or b'<html' in data[:500].lower():
                        raise RuntimeError("Google Drive returned HTML - install gdown: pip install gdown")
                    with open(tmp_path, 'wb') as f:
                        f.write(data)
                downloaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        if not downloaded or not tmp_path.exists():
            raise RuntimeError("Download failed - no file received")

        # Check if downloaded file is a zip archive
        with open(tmp_path, 'rb') as f:
            header = f.read(4)

        if header == b'PK\x03\x04':  # ZIP magic number
            print(f"Extracting model from zip archive...", file=sys.stderr)
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                # Find flownet.pkl in the archive
                pkl_files = [n for n in zf.namelist() if n.endswith('flownet.pkl')]
                if not pkl_files:
                    raise RuntimeError(f"No flownet.pkl found in zip. Contents: {zf.namelist()}")
                # Extract the pkl file
                pkl_name = pkl_files[0]
                with zf.open(pkl_name) as src, open(model_path, 'wb') as dst:
                    dst.write(src.read())
        else:
            # Already a pkl file, just move it
            shutil.move(str(tmp_path), str(model_path))

    print(f"Model saved to {model_path}", file=sys.stderr)
    return model_path


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
