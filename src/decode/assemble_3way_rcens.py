"""Slot 1 = 9771508 with ONLY the RC donor changed (single CC-loss -> CC-loss + FgOversample mean).

Tumour (WT/TC/ET) is BYTE-IDENTICAL to 9771508: mean(TopK-4reg, completion, CC-loss) decoded by
hysteresis lo=0.30/hi=0.70. The ONLY change vs assemble_3way.py is the RC probability source:

    9771508:  rcp = cc[3]                     # single CC-loss donor, RC-DSC .630
    this:     rcp = (cc[3] + fgo[3]) / 2.0     # CC-loss (+) FgOversample mean donor

Keep-largest-CC RC-fix is RETAINED (same as 9771508). We deliberately do NOT touch component
count -- that is the harness-blind change that lost as 9771576 (predicted +0.028, delivered -0.016).
This isolates a single variable -- the RC donor -- so the leaderboard delta is attributable.

Usage: assemble_3way_rcens.py <out_subdir> [lo] [hi] [nproc]
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
FGO = f"{V}/pred_fgover_s025"
OUT = f"{V}/{SUB}"
os.makedirs(OUT, exist_ok=True)


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
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
    fgo = np.load(f"{FGO}/{cid}.npz")["probabilities"]
    avg = (pt[:3] + cp[:3] + cc[:3]) / 3.0          # 3-way tumour average (identical to 9771508)
    rcp = (cc[3] + fgo[3]) / 2.0                     # RC = CC-loss (+) FgOversample mean donor

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
    print(f"[rcens] {len(cases)} cases | lo={LO} hi={HI} | RC = mean(ccloss, fgover) -> {SUB}", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(one, cases, chunksize=1)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    rc = sum(r[0] for r in res)
    empty = sum(r[1] for r in res)
    badlab = sum(r[2] for r in res)
    baddom = sum(r[3] for r in res)
    print(f"[rcens] DONE out={n} rc_cases={rc} empty={empty} bad_labels={badlab} rc_dominating={baddom}")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("ASSEMBLE_RCENS_COMPLETE")
