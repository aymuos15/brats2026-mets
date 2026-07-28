"""Assemble the 3-way fusion submission (slot 1).

Chosen by the local panoptica harness (80 GT cases, official config_mets.yaml):
it beats the team-best 9771379 on ALL NINE tumour metrics, and has the best
predicted leaderboard mean rank of 28 decodes tested.

  tumour (WT/TC/ET) = mean(TopK-4reg, completion-BackSplit, CC-loss)   <- 3-way average
                      decoded by HYSTERESIS lo=0.30 / hi=0.70
                      (keep components of p>lo that contain a p>hi core)
  RC (label 4)      = CC-loss channel 3 @ 0.5, then single-largest-CC RC-fix (unchanged)

Why it works: the official metric charges every FP component TWICE -- it appends a zero to
lesionwise DSC *and* sits in the F1 denominator. CC-loss finds more true lesions AND more
junk. Averaging it with two clean models keeps the corroborated detections and dilutes the
junk (FP/case 1.66 -> 0.85); the strict hi=0.70 core removes what survives. DSC and F1 rise
together instead of trading.

Usage: assemble_3way.py <out_subdir> [lo] [hi]
"""
import os, sys, glob
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

SUB = sys.argv[1]
LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
HI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else 10

V = os.path.expanduser("~/brats2025/val_infer")
TOPK = f"{V}/pred_topk_s025"
COMP = f"{V}/pred_topk_completion_s025"
CCL = f"{V}/pred_ccloss_s025"
OUT = f"{V}/{SUB}"
os.makedirs(OUT, exist_ok=True)


from scipy.ndimage import maximum as _ndmax
def hyst(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    peaks = np.asarray(_ndmax(p, lab, index=np.arange(1, n + 1)))
    keep = np.zeros(n + 1, bool)
    keep[1:] = peaks > hi
    return keep[lab]


def rc_largest(seg):
    m = seg == 4
    if not m.any():
        return seg
    lab, n = cc_label(m)
    if n <= 1:
        return seg
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    seg[m & (lab != sizes.argmax())] = 0
    return seg


def one(cid):
    pt = np.load(f"{TOPK}/{cid}.npz")["probabilities"]
    cp = np.load(f"{COMP}/{cid}.npz")["probabilities"]
    cc = np.load(f"{CCL}/{cid}.npz")["probabilities"]
    avg = (pt[:3] + cp[:3] + cc[:3]) / 3.0          # 3-way tumour average
    rcp = cc[3]                                      # RC from CC-loss (best donor, DSC .630)

    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):       # WT->2, TC->1, ET->3
        seg[hyst(avg[ch], LO, HI)] = cls
    seg[rcp >= 0.5] = 4
    seg = rc_largest(seg)

    ref = nib.load(f"{TOPK}/{cid}.nii.gz")
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), f"{OUT}/{cid}.nii.gz")
    labs = set(np.unique(seg).tolist())
    return (int((seg == 4).any()), int(not seg.any()),
            int(not labs.issubset({0, 1, 2, 3, 4})),
            int(seg.size > 0 and (seg == 4).sum() / seg.size > 0.08))


if __name__ == "__main__":
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))
    print(f"[3way] {len(cases)} cases | lo={LO} hi={HI} | RC from CC-loss -> {SUB}", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(one, cases, chunksize=1)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    rc = sum(r[0] for r in res)
    empty = sum(r[1] for r in res)
    badlab = sum(r[2] for r in res)
    baddom = sum(r[3] for r in res)
    print(f"[3way] DONE out={n} rc_cases={rc} empty={empty} bad_labels={badlab} rc_dominating={baddom}")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("ASSEMBLE_3WAY_COMPLETE")
