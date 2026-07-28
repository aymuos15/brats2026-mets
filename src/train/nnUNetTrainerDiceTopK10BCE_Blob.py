"""DiceTopK10BCE + blob loss (instance-aware) — region-based BraTS-METS BackSplit.

Motivation (2026-06-30): BackSplit lifts lesion-wise DSC/NSD but erodes SMALL-lesion detection
(F1 drop ~-0.13; confirmed intrinsic by the `9769617` control). Blob loss (Kofler et al., 2022)
is instance-aware: it splits each foreground region into connected components and computes a
per-component Dice, so a few LARGE lesions can no longer dominate the gradient and small/multiple
mets are forced to be segmented individually. We add it as an AUXILIARY term on top of the proven
DiceTopK10BCE backbone (NOT as a replacement — standalone blob collapses precision; here the
backbone keeps precision while blob restores small-lesion recall).

  total = DiceTopK10BCE(all scales)  +  blob_weight * blob(full-res, WT/TC/ET)

Per Kofler: for each GT connected component c, the per-component Dice masks OUT the voxels of all
OTHER components (so they neither reward nor penalise), keeping only component-c-foreground vs
true-background. Mean over components, channels, batch. Applied at full resolution only (CC is
costly); component count capped per (sample, channel) to bound per-iteration overhead.
"""
import numpy as np
import torch
from torch import nn
import cupy as cp
from cucim.skimage.measure import label as _cc_label

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import (
    nnUNetTrainerDiceTopK10BCE, DC_and_topkBCE_loss,
)
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def _gpu_label(binary_gpu):
    """26-connected components on-GPU via cuCIM. `binary_gpu`: torch bool/float CUDA tensor (D,H,W).
    Returns (label torch.int32 CUDA tensor, n_components). No host round-trip."""
    lab_cp = _cc_label(cp.asarray(binary_gpu), connectivity=3)   # cupy int array on GPU
    lab_t = torch.as_tensor(lab_cp, device=binary_gpu.device).to(torch.int32)
    return lab_t, int(lab_t.max().item())


def blob_loss(logits, target, region_idx, smooth=1e-5, max_blobs=32):
    """logits/target: (B,C,D,H,W). Per-instance Dice over GT connected components of channels
    `region_idx`, masking out other components. CC labelling is done on-GPU with cuCIM (no CPU
    sync). Returns a scalar loss (lower is better)."""
    p = torch.sigmoid(logits[:, region_idx])           # (B,R,...)
    t = (target[:, region_idx] > 0.5)
    total = logits.sum() * 0.0
    count = 0
    B, R = p.shape[0], p.shape[1]
    for b in range(B):
        for r in range(R):
            if not torch.any(t[b, r]):
                continue
            lab_t, n = _gpu_label(t[b, r])
            if n == 0:
                continue
            pr_full = p[b, r]
            comps = range(1, n + 1)
            if n > max_blobs:  # cap cost: keep the largest `max_blobs` components
                binc = torch.bincount(lab_t.reshape(-1), minlength=n + 1)[1:]
                comps = (torch.argsort(binc, descending=True)[:max_blobs] + 1).tolist()
            for ci in comps:
                comp = (lab_t == ci).float()
                keep = 1.0 - ((lab_t > 0) & (lab_t != ci)).float()  # mask OUT other blobs
                pr = pr_full * keep
                inter = (pr * comp).sum()
                denom = pr.sum() + comp.sum()
                total = total + (1.0 - (2 * inter + smooth) / (denom + smooth).clamp_min(1e-8))
                count += 1
    if count == 0:
        return logits.sum() * 0.0
    return total / count


class DC_topk_plus_blob(nn.Module):
    """Wraps the (deep-supervised) base loss and adds a full-res-only blob term."""

    def __init__(self, ds_base, region_idx, blob_weight=0.0, max_blobs=32):
        super().__init__()
        self.ds_base = ds_base
        self.region_idx = list(region_idx)
        self.blob_weight = blob_weight
        self.max_blobs = max_blobs

    def forward(self, output, target):
        base = self.ds_base(output, target)
        if self.blob_weight <= 0:
            return base
        o0 = output[0] if isinstance(output, (list, tuple)) else output
        t0 = target[0] if isinstance(target, (list, tuple)) else target
        return base + self.blob_weight * blob_loss(o0, t0, self.region_idx, max_blobs=self.max_blobs)


class nnUNetTrainerDiceTopK10BCE_Blob(nnUNetTrainerDiceTopK10BCE):
    """DiceTopK10BCE + instance-aware blob loss on WT/TC/ET (region order [WT,TC,ET,RC,aux...])."""
    K = 10
    BLOB_REGION_IDX = [0, 1, 2]   # WT, TC, ET (skip RC + aux back-split channels)
    BLOB_W = 1.0                  # final blob weight
    BLOB_RAMP_EPOCHS = 100        # ramp 0 -> BLOB_W after the backbone settles
    MAX_BLOBS = 32

    def _build_loss(self):
        assert self.label_manager.has_regions, "blob trainer is for region-based training"
        base = DC_and_topkBCE_loss(
            {}, {"batch_dice": self.configuration_manager.batch_dice,
                 "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp},
            use_ignore_label=self.label_manager.ignore_label is not None,
            k=self.K, dice_class=MemoryEfficientSoftDiceLoss)
        if self._do_i_compile():
            base.dc = torch.compile(base.dc)
        if self.enable_deep_supervision:
            dss = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(dss))])
            weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            weights = weights / weights.sum()
            base = DeepSupervisionWrapper(base, weights)
        return DC_topk_plus_blob(base, self.BLOB_REGION_IDX, blob_weight=0.0, max_blobs=self.MAX_BLOBS)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        w = min(self.BLOB_W, self.BLOB_W * self.current_epoch / max(1, self.BLOB_RAMP_EPOCHS))
        self.loss.blob_weight = float(w)
        self.print_to_log_file(f"blob_weight = {w:.4f}")
