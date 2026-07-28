"""Slot 2 = D015 DilateISO STANDALONE, eroded, RC grafted from CC-loss.

D015 was trained on targets where isolated small mets were dilated k=2 (build_d015).
It CANNOT be probability-averaged into the 3-way (its extent is inflated), so it is a
standalone submission. Decode:
  tumour = D015 channels [wt,tc,et] -> hysteresis lo/hi -> binary
           -> SELECTIVE erosion: erode components smaller than TVOX by k=EK (only the
              small mets were dilated in training; large structures were NOT, so eroding
              them would wrongly shrink the metrics we are strong on).
  RC     = CC-loss channel 3 @ 0.5, keep-largest (as every submission).

EK and TVOX come from d015_extent.py (measure before trusting). EK=0 => no erosion.

Usage: assemble_d015.py <out_subdir> [lo] [hi] [EK] [TVOX] [nproc]
"""
import os, sys, glob
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label, binary_erosion, generate_binary_structure

SUB = sys.argv[1]
LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
HI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
EK = int(sys.argv[4]) if len(sys.argv) > 4 else 2        # erosion iterations; 0 = off
TVOX = int(sys.argv[5]) if len(sys.argv) > 5 else 500    # erode comps smaller than this
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 10

V = os.path.expanduser("~/brats2025/val_infer")
D015 = f"{V}/pred_d015_s025"
CCL = f"{V}/pred_ccloss_s025"          # RC donor (undilated), DSC .630
OUT = f"{V}/{SUB}"
STRUCT = np.ones((3, 3, 3), int)
BALL = generate_binary_structure(3, 1)
os.makedirs(OUT, exist_ok=True)


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo, STRUCT)
    if n == 0:
        return np.zeros(p.shape, bool)
    cm = np.zeros(n + 1, np.float32)
    np.maximum.at(cm, lab.ravel(), p.ravel())
    keep = cm > hi
    keep[0] = False
    return keep[lab]


def selective_erode(mask):
    """Erode only components < TVOX by EK. Large structures untouched."""
    if EK <= 0 or not mask.any():
        return mask
    lab, n = cc_label(mask, STRUCT)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    out = mask.copy()
    for i in range(1, n + 1):
        if sizes[i] < TVOX:
            comp = lab == i
            er = binary_erosion(comp, BALL, iterations=EK)
            if er.any():                    # never erode a lesion out of existence
                out[comp] = False
                out[er] = True
            # else: too small to erode safely -> leave as predicted
    return out


def rc_largest(seg):
    m = seg == 4
    if not m.any():
        return seg
    lab, n = cc_label(m, STRUCT)
    if n <= 1:
        return seg
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    seg[m & (lab != sizes.argmax())] = 0
    return seg


def one(cid):
    dp = np.load(f"{D015}/{cid}.npz")["probabilities"]
    cc = np.load(f"{CCL}/{cid}.npz")["probabilities"]
    seg = np.zeros(dp.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):      # wt->2, tc->1, et->3
        m = hyst(dp[ch], LO, HI)
        m = selective_erode(m)
        seg[m] = cls
    seg[cc[3] >= 0.5] = 4
    seg = rc_largest(seg)
    ref = nib.load(f"{D015}/{cid}.nii.gz")
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), f"{OUT}/{cid}.nii.gz")
    labs = set(np.unique(seg).tolist())
    return (int((seg == 4).any()), int(not seg.any()),
            int(not labs.issubset({0, 1, 2, 3, 4})),
            int(seg.size > 0 and (seg == 4).sum() / seg.size > 0.08))


if __name__ == "__main__":
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(D015 + "/*.npz"))
    print(f"[d015] {len(cases)} cases | lo={LO} hi={HI} | erode k={EK} comps<{TVOX}vox | RC=ccloss", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(one, cases, chunksize=1)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    rc = sum(r[0] for r in res); empty = sum(r[1] for r in res)
    badlab = sum(r[2] for r in res); baddom = sum(r[3] for r in res)
    print(f"[d015] DONE out={n} rc_cases={rc} empty={empty} bad_labels={badlab} rc_dominating={baddom}")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("ASSEMBLE_D015_COMPLETE")
