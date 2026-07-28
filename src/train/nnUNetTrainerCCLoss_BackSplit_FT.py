"""CC/blob-loss FINE-TUNE trainer — region-based BraTS-METS BackSplit (Dataset011).

Goal (2026-07-07): the leaderboard gap is small-lesion DETECTION (lesion-wise F1), not DSC/NSD.
Literature (arXiv:2604.24276, multi-class CC/blob loss on BraTS-METS 2025: FG Dice 0.59->0.64,
better Panoptic Quality) shows a per-connected-component ("blob") loss weights each lesion
INSTANCE equally regardless of size -> protects small mets that voxel-mean losses drown out.

This is a FINE-TUNE, not from-scratch: launch with `-pretrained_weights <completion ckpt>`,
100 ep, lr 1e-3 (low), so a converged DSC-strong parent is reshaped toward instance-awareness
in the final stretch (no 1000-ep wash-out of the effect). Blob loss is added as a SMALL-lambda
AUXILIARY to the existing Dice+TopK-BCE (NOT a replacement) so the DSC we already have is protected,
and it is applied ONLY at the full-resolution deep-supervision head (downsampled scales merge
components anyway) -> bounds the CC cost that killed blob-loss-standalone before.

One-vs-rest over the 4 REAL region channels [WT,TC,ET,RC] (channels 0..3 of the 6-ch BackSplit
output; aux NETC/SNFH excluded). Per channel: label GT connected components (26-conn), and for
each component compute a soft-Dice of the prediction against that component within its (padded)
bounding box. Uniform average over channels-with-foreground (paper's "uniform class averaging").
Components capped/subsampled per (sample,channel) to bound cost; caps are logged via s/ep at smoke.
"""
import numpy as np
import torch
from torch import nn
from scipy import ndimage

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import (
    nnUNetTrainerDiceTopK10BCE,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

STRUCT = ndimage.generate_binary_structure(3, 3)  # 26-connectivity (matches lesion CC convention)


class BlobLoss(nn.Module):
    """One-vs-rest per-connected-component soft-Dice over the real region channels.

    logits: (B, C>=4, D, H, W) raw net output; target: (B, C>=4, ...) region GT (ignore ch, if any,
    is the LAST channel and is excluded by slicing [:, :n_ch]). Returns a scalar loss (lower better).
    """

    def __init__(self, n_ch=4, smooth=1e-5, max_comp=48, pad=4):
        super().__init__()
        self.n_ch = n_ch
        self.smooth = smooth
        self.max_comp = max_comp   # cap components per (sample,channel); subsample if exceeded
        self.pad = pad             # bbox padding (voxels) so prediction context is included

    def forward(self, logits, target):
        p = torch.sigmoid(logits[:, : self.n_ch])
        y = target[:, : self.n_ch]
        if y.dtype != p.dtype:
            y = y.to(p.dtype)
        B, C = p.shape[0], self.n_ch
        losses = []
        y_np = (y > 0.5).detach().cpu().numpy().astype(np.uint8)  # CC on GT only (no grad needed)
        for b in range(B):
            for c in range(C):
                g = y_np[b, c]
                if g.sum() == 0:
                    continue
                lab, n = ndimage.label(g, STRUCT)
                if n == 0:
                    continue
                objs = ndimage.find_objects(lab)
                idxs = list(range(1, n + 1))
                if n > self.max_comp:  # keep cost bounded; random subset preserves small mets in expectation
                    idxs = list(np.random.choice(idxs, self.max_comp, replace=False))
                comp_losses = []
                for i in idxs:
                    sl = objs[i - 1]
                    if sl is None:
                        continue
                    # pad the bbox so surrounding predicted probability is scored too
                    psl = tuple(
                        slice(max(0, s.start - self.pad), min(dim, s.stop + self.pad))
                        for s, dim in zip(sl, g.shape)
                    )
                    pc = p[b, c][psl]
                    mc = torch.as_tensor((lab[psl] == i), dtype=pc.dtype, device=pc.device)
                    inter = (pc * mc).sum()
                    dice = (2 * inter + self.smooth) / (pc.sum() + mc.sum() + self.smooth)
                    comp_losses.append(1.0 - dice)
                if comp_losses:
                    losses.append(torch.stack(comp_losses).mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class _CombinedDSBlob(nn.Module):
    """base deep-supervision (Dice+TopK-BCE) + lambda * blob(full-res head only)."""

    def __init__(self, ds_loss, blob, lam):
        super().__init__()
        self.ds_loss = ds_loss
        self.blob = blob
        self.lam = lam

    def forward(self, output, target):
        base = self.ds_loss(output, target)
        o0 = output[0] if isinstance(output, (list, tuple)) else output
        t0 = target[0] if isinstance(target, (list, tuple)) else target
        return base + self.lam * self.blob(o0, t0)


class nnUNetTrainerCCLoss_BackSplit_FT(nnUNetTrainerDiceTopK10BCE):
    LAMBDA_BLOB = 0.2   # small aux weight -> protect existing DSC
    N_REAL_CH = 4       # WT,TC,ET,RC (exclude NETC/SNFH aux channels 4,5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 100          # FINE-TUNE budget
        self.initial_lr = 1e-3         # low LR (default 1e-2 would blow away the converged weights)

    def _build_loss(self):
        # rebuild the exact base (DS-wrapped Dice+TopK-BCE) via the parent, then bolt on blob aux.
        base = super()._build_loss()
        blob = BlobLoss(n_ch=self.N_REAL_CH)
        if not self.enable_deep_supervision:
            # base already returns a per-tensor loss; wrap so blob sees the single output/target
            class _CB(nn.Module):
                def __init__(s, b, bl, lam):
                    super().__init__(); s.b = b; s.bl = bl; s.lam = lam
                def forward(s, o, t):
                    return s.b(o, t) + s.lam * s.bl(o, t)
            return _CB(base, blob, self.LAMBDA_BLOB)
        return _CombinedDSBlob(base, blob, self.LAMBDA_BLOB)
