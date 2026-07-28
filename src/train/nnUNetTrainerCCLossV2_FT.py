"""CC-loss v2: blob (recall) + METRIC-FAITHFUL per-component FP penalty (precision).

WHY (2026-07-13, from reading the official metrics_parser.py + sub 9771380):

  9771380 = CC-loss base + hysteresis scored small-instance F1 .429/.530/.543 =
  **rank 3 / 1422** of every scored submission in the challenge. Its ONLY problem is
  false-positive components, and the official metric charges them TWICE:

      lesionwise_dsc : large_lesion_dsc.extend([0] * num_fp)   -> each FP appends a ZERO
      small_f1       : 2*TP / (2*TP + num_fp + FN)             -> each FP in the denominator

  So DSC and F1 are NOT in tension. Both want max TP, min FP. Kill CC-loss's FPs and
  BOTH rise -> top-3 overall.

WHAT WAS WRONG WITH v1 (nnUNetTrainerBlobDist_NoRC):
  BlobLoss normalises PER GT COMPONENT (each lesion counts equally -> strong recall push).
  BoundaryLoss normalised PER VOXEL: (p * sdf).mean() over B*C*96^3 = 15.9M voxels. A
  spurious 50-voxel blob moves the loss by 9.4e-7, while finding one small lesion is worth
  2.2e-3. Recall pressure outweighed precision pressure ~2400:1 -> the net was effectively
  TOLD to spray FPs. The precision term was there in intent but three orders of magnitude
  too weak to bite.

THE FIX: make precision PER-COMPONENT too, symmetric with blob loss, and define "FP"
exactly as the scorer does (MaxBipartiteMatching, DSC, threshold ~0):
  - label the PREDICTED components (p > 0.5)
  - label the GT components
  - each GT component may claim AT MOST ONE predicted component (its best-DSC one)
  - every unclaimed predicted component is a FALSE POSITIVE  <-- incl. a 2nd fragment
    sitting on an already-matched lesion, which the 1:1 bipartite rule also calls an FP
  - penalise the mean probability of each FP component  -> O(1) per FP, same scale as blob

  This teaches "do not fire where there is no lesion, and do not fragment a lesion you
  already found" -- the two things that manufacture num_fp -- at the SAME gradient scale
  as "do find every lesion".

No signed-distance term: it cost ~2.2x epoch time (18 CPU EDTs/step) and contributed
essentially nothing. FT recipe mirrors the v1 CC-loss FT that produced 9771380's base:
warm-start from the completion checkpoint, short run, low LR.
"""
import numpy as np
import torch
from torch import nn
from scipy import ndimage

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import (
    nnUNetTrainerDiceTopK10BCE,
)

STRUCT = ndimage.generate_binary_structure(3, 3)   # 26-conn, matches panoptica CCA


class BlobRecallLoss(nn.Module):
    """One-vs-rest per-GT-component soft-Dice: every lesion counts equally (recall)."""

    def __init__(self, n_ch=4, smooth=1e-5, max_comp=48, pad=4, min_vox=20):
        super().__init__()
        self.n_ch, self.smooth, self.max_comp, self.pad = n_ch, smooth, max_comp, pad
        self.min_vox = min_vox   # official vol_threshold is 20 mm^3 (1mm iso -> 20 vox)

    def forward(self, p, y_np):
        losses = []
        B = p.shape[0]
        for b in range(B):
            for c in range(self.n_ch):
                g = y_np[b, c]
                if g.sum() == 0:
                    continue
                lab, n = ndimage.label(g, STRUCT)
                if n == 0:
                    continue
                objs = ndimage.find_objects(lab)
                idxs = list(range(1, n + 1))
                if len(idxs) > self.max_comp:
                    idxs = list(np.random.choice(idxs, self.max_comp, replace=False))
                comp = []
                for i in idxs:
                    sl = objs[i - 1]
                    if sl is None:
                        continue
                    psl = tuple(slice(max(0, s.start - self.pad), min(dim, s.stop + self.pad))
                                for s, dim in zip(sl, g.shape))
                    pc = p[b, c][psl]
                    mc = torch.as_tensor(lab[psl] == i, dtype=pc.dtype, device=pc.device)
                    inter = (pc * mc).sum()
                    dice = (2 * inter + self.smooth) / (pc.sum() + mc.sum() + self.smooth)
                    comp.append(1.0 - dice)
                if comp:
                    losses.append(torch.stack(comp).mean())
        if not losses:
            return None
        return torch.stack(losses).mean()


class FPComponentLoss(nn.Module):
    """Penalise every PREDICTED component the scorer would call a false positive.

    FP := a predicted component that no GT component claims, under 1:1 best-DSC matching.
    Penalty := mean predicted probability inside that component (gradient flows through p).
    Normalised per FP component -> same O(1) scale as BlobRecallLoss's per-GT term.
    """

    def __init__(self, n_ch=4, thresh=0.5, max_fp=64):
        super().__init__()
        self.n_ch, self.thresh, self.max_fp = n_ch, thresh, max_fp

    def forward(self, p, y_np):
        losses = []
        B = p.shape[0]
        p_det = p.detach()
        for b in range(B):
            for c in range(self.n_ch):
                pb = (p_det[b, c] > self.thresh).cpu().numpy()
                if not pb.any():
                    continue
                plab, pn = ndimage.label(pb, STRUCT)
                if pn == 0:
                    continue
                g = y_np[b, c]
                glab, gn = ndimage.label(g, STRUCT) if g.sum() else (None, 0)

                # 1:1 matching -- each GT component claims its single best-DSC pred component
                claimed = set()
                if gn:
                    psz = np.bincount(plab.ravel(), minlength=pn + 1)
                    for gi in range(1, gn + 1):
                        gmask = glab == gi
                        gsz = int(gmask.sum())
                        overl = np.bincount(plab[gmask], minlength=pn + 1)
                        overl[0] = 0
                        if overl.max() == 0:
                            continue                       # this GT lesion is a FN, not our business here
                        dsc = 2.0 * overl[1:] / np.maximum(psz[1:] + gsz, 1)
                        best = int(np.argmax(dsc)) + 1
                        claimed.add(best)                  # only ONE pred comp per GT comp

                fps = [i for i in range(1, pn + 1) if i not in claimed]
                if not fps:
                    continue
                if len(fps) > self.max_fp:
                    fps = list(np.random.choice(fps, self.max_fp, replace=False))
                for i in fps:
                    m = torch.as_tensor(plab == i, dtype=p.dtype, device=p.device)
                    denom = m.sum()
                    if denom < 1:
                        continue
                    losses.append((p[b, c] * m).sum() / denom)   # mean prob inside the FP blob
        if not losses:
            return None
        return torch.stack(losses).mean()


class _Combined(nn.Module):
    def __init__(self, ds_loss, blob, fp, lam_blob, lam_fp, n_ch):
        super().__init__()
        self.ds_loss, self.blob, self.fp = ds_loss, blob, fp
        self.lam_blob, self.lam_fp, self.n_ch = lam_blob, lam_fp, n_ch

    def forward(self, output, target):
        base = self.ds_loss(output, target)
        o0 = output[0] if isinstance(output, (list, tuple)) else output
        t0 = target[0] if isinstance(target, (list, tuple)) else target
        p = torch.sigmoid(o0[:, : self.n_ch])
        y_np = (t0[:, : self.n_ch] > 0.5).detach().cpu().numpy().astype(np.uint8)
        total = base
        lb = self.blob(p, y_np)
        if lb is not None:
            total = total + self.lam_blob * lb
        lf = self.fp(p, y_np)
        if lf is not None:
            total = total + self.lam_fp * lf
        return total


class nnUNetTrainerCCLossV2_FT(nnUNetTrainerDiceTopK10BCE):
    """FT recipe: warm-start from completion ckpt, 150 ep, lr 1e-3."""

    LAMBDA_BLOB = 0.2      # recall pressure  (as in v1, which produced the #3-F1 base)
    LAMBDA_FP = 0.2        # precision pressure -- SAME scale, the whole point of v2
    N_REAL_CH = 4          # WT,TC,ET,RC on D011 (6-region output, real regions are 0..3)

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        # explicit signature: nnUNet introspects locals(), *args breaks it
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.initial_lr = 1e-3

    def _build_loss(self):
        base = super()._build_loss()
        return _Combined(base,
                         BlobRecallLoss(n_ch=self.N_REAL_CH),
                         FPComponentLoss(n_ch=self.N_REAL_CH),
                         self.LAMBDA_BLOB, self.LAMBDA_FP, self.N_REAL_CH)

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[CCLossV2] lam_blob={self.LAMBDA_BLOB} lam_fp={self.LAMBDA_FP} "
            f"n_real_ch={self.N_REAL_CH} epochs={self.num_epochs} lr={self.initial_lr}\n"
            f"[CCLossV2] FP := predicted component unclaimed under 1:1 best-DSC matching "
            f"(matches the official MaxBipartiteMatching scorer, incl. fragments)")
