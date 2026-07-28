"""RC step-probe: 9771992 with the RC channel taken from a FINER-STEP (0.125)
CC-loss re-prediction, tumour byte-identical.

Tumour (WT/TC/ET) is recomputed from the SAME cached step-0.25 probs and the SAME
hysteresis lo0.25/hi0.70 as 9771992 -> byte-identical tumour. ONLY the RC source
changes: cc[3] from pred_ccloss_s0125 (step 0.125) instead of pred_ccloss_s025
(step 0.25). RC threshold 0.5 + single-largest-CC unchanged (the STEP is the probe,
not the threshold). Because keep-largest runs after the threshold, RC stays one
component -> pure extent/boundary change -> only the 4 RC metrics can move.

Usage: assemble_rc_lowstep.py <out_subdir> [nproc]
"""
import os, sys, glob
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

LO, HI = 0.25, 0.70            # 9771992's operating point (extent optimum)
RC_T = 0.5
SUB = sys.argv[1]
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 8

V = os.path.expanduser("~/brats2025/val_infer")
TOPK = f"{V}/pred_topk_s025"
COMP = f"{V}/pred_topk_completion_s025"
CCL = f"{V}/pred_ccloss_s025"          # tumour members (step 0.25, unchanged)
RCDIR = f"{V}/pred_ccloss_s0125"       # RC source (step 0.125, finer)
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
    avg = (pt[:3] + cp[:3] + cc[:3]) / 3.0          # byte-identical 9771992 tumour
    rcp = np.load(f"{RCDIR}/{cid}.npz")["probabilities"][3]   # finer-step RC

    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):
        seg[hyst(avg[ch], LO, HI)] = cls
    seg[rcp >= RC_T] = 4
    seg = rc_largest(seg)

    ref = nib.load(f"{TOPK}/{cid}.nii.gz")
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), f"{OUT}/{cid}.nii.gz")
    labs = set(np.unique(seg).tolist())
    return (int((seg == 4).any()), int(not seg.any()),
            int(not labs.issubset({0, 1, 2, 3, 4})),
            int(seg.size > 0 and (seg == 4).sum() / seg.size > 0.08))


if __name__ == "__main__":
    miss = [os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz")
            if not os.path.exists(f"{RCDIR}/{os.path.basename(f)[:-4]}.npz")]
    if miss:
        print(f"[rc-lowstep] ABORT: {len(miss)} cases missing in {RCDIR} (e.g. {miss[:2]})")
        sys.exit(1)
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))
    print(f"[rc-lowstep] {len(cases)} cases | tumour lo{LO}/hi{HI} (=9771992) | RC=s0125 ch3 @ {RC_T} -> {SUB}", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(one, cases, chunksize=1)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    rc = sum(r[0] for r in res); empty = sum(r[1] for r in res)
    badlab = sum(r[2] for r in res); baddom = sum(r[3] for r in res)
    print(f"[rc-lowstep] DONE out={n} rc_cases={rc} empty={empty} bad_labels={badlab} rc_dominating={baddom}")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("ASSEMBLE_RC_LOWSTEP_COMPLETE")
