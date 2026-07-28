"""Assemble a BraTS submission: hysteresis tumour decode over a selectable base + RC graft.

Usage: assemble_hyst2.py <out_subdir> <base> <lo> <hi> [rc_src] [nproc]
   base   : topk_comp    = 0.5*TopK4reg + 0.5*completion  (base of best sub 9771150)
            ccloss_topk  = 0.5*CCloss   + 0.5*TopK4reg    (base of 9770752, top-6 F1 band)
   lo,hi  : hysteresis - keep CCs of (p>lo) that contain a (p>hi) core
   rc_src : ccloss (default; RC-DSC 0.630 = best donor) | onlyrc

Decode WT(2) -> TC(1) -> ET(3), then RC(4), then single-largest-CC RC-fix.
Same ops as assemble_hyst.py (which produced 9771150) with base selectable.
"""
import os, sys, glob, numpy as np, nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

SUB, BASE, LO, HI = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
RC_SRC = sys.argv[5] if len(sys.argv) > 5 else "ccloss"
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 12

V = os.path.expanduser("~/brats2025/val_infer")
TOPK, COMP, CCL = f"{V}/pred_topk_s025", f"{V}/pred_topk_completion_s025", f"{V}/pred_ccloss_s025"
OTHER = {"topk_comp": COMP, "ccloss_topk": CCL}[BASE]
RCDIR, RCCH = {"ccloss": (CCL, 3), "onlyrc": (f"{V}/pred_onlyrc_s025", 0)}[RC_SRC]
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
    ot = np.load(f"{OTHER}/{cid}.npz")["probabilities"]
    avg = 0.5 * pt[:3] + 0.5 * ot[:3]
    rcp = np.load(f"{RCDIR}/{cid}.npz")["probabilities"][RCCH]

    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):        # WT->2, TC->1, ET->3
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
    print(f"[hyst2] {len(cases)} cases | base={BASE} lo={LO} hi={HI} rc={RC_SRC} -> {SUB}", flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(one, cases, chunksize=1)
    rc_cases = sum(r[0] for r in res)
    empty = sum(r[1] for r in res)
    badlab = sum(r[2] for r in res)
    baddom = sum(r[3] for r in res)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    print(f"[hyst2] DONE out={n} rc_cases={rc_cases} empty={empty} "
          f"bad_labels={badlab} rc_dominating={baddom}")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("HYST2_ASSEMBLE_COMPLETE")
