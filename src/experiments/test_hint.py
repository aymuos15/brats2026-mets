"""Self-contained dry run for the hint mechanism (torch only, no nnU-Net data/managers).

Validates before we ever burn GPU:
  1. Multi-scale LoG highlights a small bright blob vs smooth background.
  2. Scale-selectivity: a SMALL blob peaks at a SMALL sigma, a LARGE blob at a LARGE
     sigma — i.e. the scales are doing distinct work (the point of going multi-scale).
  3. The full 4->(4+S) channel input runs forward+backward through a conv net, no NaN.
Imports the shipped operators directly so the tested code IS the shipped code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from hint_ops import multiscale_log, log3d, DEFAULT_SIGMAS

torch.manual_seed(0)


def _planted(D, radius, amp=5.0, B=2):
    vol = torch.randn(B, 1, D, D, D) * 0.05
    c = D // 2
    vol[:, :, c - radius:c + radius, c - radius:c + radius, c - radius:c + radius] += amp
    return vol, c


def test_multiscale_highlights_blob():
    vol, c = _planted(32, radius=2)
    hint = multiscale_log(vol, DEFAULT_SIGMAS)
    assert hint.shape == (2, len(DEFAULT_SIGMAS), 32, 32, 32), hint.shape
    assert torch.isfinite(hint).all()
    blob = hint[:, :, c - 3:c + 3, c - 3:c + 3, c - 3:c + 3].abs().mean()
    bg = hint[:, :, :6, :6, :6].abs().mean()
    ratio = (blob / (bg + 1e-6)).item()
    print(f"  multiscale LoG blob/bg ratio = {ratio:.1f} (want >> 1)")
    assert ratio > 5.0


def test_scale_selectivity():
    # small blob (r=1) vs large blob (r=5); compare LoG response at small vs large sigma
    sig_small, sig_large = DEFAULT_SIGMAS[0], DEFAULT_SIGMAS[-1]

    def peak(vol, c, sigma):
        return log3d(vol, sigma)[:, :, c - 6:c + 6, c - 6:c + 6, c - 6:c + 6].abs().amax()

    vs, c = _planted(48, radius=1)
    vl, _ = _planted(48, radius=6)
    small_at_small = peak(vs, c, sig_small)
    small_at_large = peak(vs, c, sig_large)
    large_at_small = peak(vl, c, sig_small)
    large_at_large = peak(vl, c, sig_large)
    print(f"  small blob: sigma{sig_small}={small_at_small:.2f}  sigma{sig_large}={small_at_large:.2f}")
    print(f"  large blob: sigma{sig_small}={large_at_small:.2f}  sigma{sig_large}={large_at_large:.2f}")
    # small blob prefers the small scale; large blob prefers the large scale (relatively)
    assert small_at_small > small_at_large, "small blob should peak at small sigma"
    assert (large_at_large / large_at_small) > (small_at_large / small_at_small), \
        "large blob should shift preference toward the large sigma"


def test_full_channel_forward_backward():
    B, S = 2, len(DEFAULT_SIGMAS)
    data4 = torch.randn(B, 4, 16, 16, 16)
    hint = multiscale_log(data4[:, 0:1].float(), DEFAULT_SIGMAS)
    data = torch.cat([data4, hint], dim=1).clone().requires_grad_(True)
    assert data.shape[1] == 4 + S, data.shape
    net = nn.Sequential(
        nn.Conv3d(4 + S, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.LeakyReLU(),
        nn.Conv3d(8, 6, 1),  # 6 region heads (BackSplit)
    )
    out = net(data)
    assert out.shape == (B, 6, 16, 16, 16), out.shape
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).all() and torch.isfinite(data.grad).all()
    print(f"  {4+S}ch forward ok, out {tuple(out.shape)}, loss {loss.item():.4f}, grad finite")


if __name__ == "__main__":
    print(f"scales = {DEFAULT_SIGMAS}")
    print("test_multiscale_highlights_blob:"); test_multiscale_highlights_blob()
    print("test_scale_selectivity:"); test_scale_selectivity()
    print("test_full_channel_forward_backward:"); test_full_channel_forward_backward()
    print("\nALL PASS")
