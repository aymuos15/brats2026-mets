#!/usr/bin/env python3
"""3-way prob-average: w_topk*TopK + w_ft*FT + w_bs*pureBackSplit, region ch [0:4].
Keep TopK at 0.5 (F1 anchor, as in 9770198); split the rest across the two
decorrelated DSC-strong parents (FT + pure-BackSplit) to push tumour DSC/NSD via
super-additivity without conceding F1. Decode (RCO=[2,1,3,4], thr 0.5) -> RC-fix.
Identical affine/transpose handling to ensemble_ft.py (which produced 9770198).
Usage: ensemble_3way.py <w_topk> <w_ft> <w_bs> <out_subdir>
"""
import os, sys, glob, numpy as np, nibabel as nib
from scipy.ndimage import label as cc_label

WT_, WF_, WB_ = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
SUB = sys.argv[4]
base = os.path.expanduser("~/brats2025/val_infer")
TOPK = base + "/pred_topk_best"
FT   = base + "/pred_ft_best"
BS   = base + "/pred_backsplit_best"
OUT  = base + "/" + SUB
RCO = [2, 1, 3, 4]; THR = [0.5, 0.5, 0.5, 0.5]
os.makedirs(OUT, exist_ok=True)
assert abs((WT_ + WF_ + WB_) - 1.0) < 1e-6, "weights must sum to 1"

def rc_largest(seg):
    m = (seg == 4)
    if not m.any(): return seg
    lab, n = cc_label(m)
    if n <= 1: return seg
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    seg[m & (lab != sizes.argmax())] = 0
    return seg

cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))
print(f"[3way {WT_}/{WF_}/{WB_}] {len(cases)} cases -> {SUB}"); miss = 0
for cid in cases:
    ff, bf = FT + "/" + cid + ".npz", BS + "/" + cid + ".npz"
    if not (os.path.exists(ff) and os.path.exists(bf)):
        print("MISSING", cid); miss += 1; continue
    pt = np.load(TOPK + "/" + cid + ".npz")["probabilities"][:4]
    pf = np.load(ff)["probabilities"][:4]
    pb = np.load(bf)["probabilities"][:4]
    avg = WT_ * pt + WF_ * pf + WB_ * pb
    seg = np.zeros(avg.shape[1:], dtype=np.uint8)
    for ch, cls in enumerate(RCO):
        seg[avg[ch] >= THR[ch]] = cls
    seg = rc_largest(seg)
    ref = nib.load(TOPK + "/" + cid + ".nii.gz")
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), OUT + "/" + cid + ".nii.gz")
n = len(glob.glob(OUT + "/*.nii.gz"))
print(f"[3way] DONE missing={miss} out={n}")
print("ENSEMBLE_COMPLETE")
