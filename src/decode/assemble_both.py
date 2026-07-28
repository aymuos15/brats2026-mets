"""Single-pass 3-way hysteresis assembler for TWO seed variants at once.
Loads each case's 3 prediction npz ONCE (dominant cost on the giant 640^3 UCSD cases,
~80s load each) and writes both lo=0.30/hi=0.65 and lo=0.30/hi=0.75 masks.
Identical decode to assemble_3way.py (verified). Usage: assemble_both.py [nproc]
"""
import os, sys, glob
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

V = os.path.expanduser("~/brats2025/val_infer")
TOPK, COMP, CCL = f"{V}/pred_topk_s025", f"{V}/pred_topk_completion_s025", f"{V}/pred_ccloss_s025"
VARIANTS = [("sub_3way_lo030hi065", 0.30, 0.65), ("sub_3way_lo030hi075", 0.30, 0.75)]
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for name, _, _ in VARIANTS:
    os.makedirs(f"{V}/{name}", exist_ok=True)


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
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    seg[m & (lab != sizes.argmax())] = 0
    return seg


def one(cid):
    # skip cases already done for BOTH variants (resume-safe)
    if all(os.path.exists(f"{V}/{name}/{cid}.nii.gz") for name, _, _ in VARIANTS):
        return cid, "skip"
    pt = np.load(f"{TOPK}/{cid}.npz")["probabilities"]
    cp = np.load(f"{COMP}/{cid}.npz")["probabilities"]
    cc = np.load(f"{CCL}/{cid}.npz")["probabilities"]
    avg = (pt[:3] + cp[:3] + cc[:3]) / 3.0
    rcp = cc[3].copy()
    del pt, cp, cc                                   # free ~11GB before the heavy decode
    ref = nib.load(f"{TOPK}/{cid}.nii.gz")
    for name, lo, hi in VARIANTS:
        seg = np.zeros(avg.shape[1:], np.uint8)
        for ch, cls in zip([0, 1, 2], [2, 1, 3]):
            seg[hyst(avg[ch], lo, hi)] = cls
        seg[rcp >= 0.5] = 4
        seg = rc_largest(seg)
        nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header),
                 f"{V}/{name}/{cid}.nii.gz")
    return cid, "done"


if __name__ == "__main__":
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))
    print(f"[both] {len(cases)} cases -> {[v[0] for v in VARIANTS]} | nproc={NPROC}", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, (cid, st) in enumerate(pool.imap_unordered(one, cases)):
            done += 1
            if done % 10 == 0 or done == len(cases):
                print(f"  ..{done}/{len(cases)}", flush=True)
    for name, _, _ in VARIANTS:
        n = len(glob.glob(f"{V}/{name}/*.nii.gz"))
        print(f"  {name}: {n} masks")
    print("ASSEMBLE_BOTH_COMPLETE")
