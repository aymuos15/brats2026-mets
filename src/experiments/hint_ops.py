"""Hand-crafted salience operators for the hint channel (torch-only, no nnU-Net dep).

v2: multi-scale Laplacian-of-Gaussian (LoG) blob detector — the operator brain-mets
detection frameworks actually use for small-lesion candidate selection (arXiv:1908.04701,
arXiv:2105.13406). LoG = Laplacian after Gaussian pre-smoothing at scale sigma:
  - the Gaussian smooth removes the raw Laplacian's voxel-noise hypersensitivity;
  - a small set of scales (sigmas) catches lesions from ~2mm dots to ~15mm nodules;
  - scale-normalised (x sigma^2) so responses are comparable across scales.

Kept standalone so it imports on any box (laptop for tests, Spark for training) and
so the tested code IS the shipped code.
"""
import math

import torch
import torch.nn.functional as F

# default scales in voxels (Dataset007/011 is ~1mm isotropic -> ~1,2,4mm)
DEFAULT_SIGMAS = (1.0, 2.0, 4.0)


def laplacian3d(x: torch.Tensor) -> torch.Tensor:
    """6-neighbourhood 3D Laplacian of a single-channel volume, reflect-padded.

    x: (B, 1, D, H, W) -> (B, 1, D, H, W).
    """
    kernel = x.new_zeros((1, 1, 3, 3, 3))
    kernel[0, 0, 1, 1, 1] = 6.0
    for dz, dy, dx in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
        kernel[0, 0, dz, dy, dx] = -1.0
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="reflect")
    return F.conv3d(xp, kernel)


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(t ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur (cheap: three 1D convs), reflect-padded.

    x: (B, 1, D, H, W) -> (B, 1, D, H, W).
    """
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    # dim 2 = D, dim 3 = H, dim 4 = W; F.pad order is (Wl,Wr,Hl,Hr,Dl,Dr)
    specs = (
        (k.view(1, 1, -1, 1, 1), (0, 0, 0, 0, r, r)),  # along D
        (k.view(1, 1, 1, -1, 1), (0, 0, r, r, 0, 0)),  # along H
        (k.view(1, 1, 1, 1, -1), (r, r, 0, 0, 0, 0)),  # along W
    )
    for ker, pad in specs:
        x = F.conv3d(F.pad(x, pad, mode="reflect"), ker)
    return x


def log3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Scale-normalised Laplacian-of-Gaussian at one scale. x:(B,1,D,H,W)."""
    return (sigma ** 2) * laplacian3d(gaussian_blur3d(x, sigma))


def multiscale_log(x: torch.Tensor, sigmas=DEFAULT_SIGMAS) -> torch.Tensor:
    """Stack LoG responses at several scales -> (B, len(sigmas), D, H, W)."""
    return torch.cat([log3d(x, s) for s in sigmas], dim=1)
