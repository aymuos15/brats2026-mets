"""Does my in-loss FP definition agree with the OFFICIAL panoptica scorer?

If the loss penalises a different set of components than the metric charges us for,
the trainer optimises the wrong thing. Build synthetic cases with known answers,
score them with the real evaluator, and compare FP counts.

Cases:
  1. perfect match                     -> fp 0
  2. one spurious blob far from GT     -> fp 1
  3. GT lesion split into 2 pred blobs -> fp 1  (bipartite 1:1: only one can be claimed)
  4. two GT lesions, one missed        -> fp 0, fn 1
"""
import os, tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage

CFG = os.path.expanduser("~/brats2025/brats_eval/brats_evaluation/configs/config_mets.yaml")
STRUCT = ndimage.generate_binary_structure(3, 3)


def my_fp_count(pred_bin, gt_bin):
    """The FPComponentLoss rule, extracted."""
    plab, pn = ndimage.label(pred_bin, STRUCT)
    if pn == 0:
        return 0
    glab, gn = ndimage.label(gt_bin, STRUCT) if gt_bin.sum() else (None, 0)
    claimed = set()
    if gn:
        psz = np.bincount(plab.ravel(), minlength=pn + 1)
        for gi in range(1, gn + 1):
            gmask = glab == gi
            gsz = int(gmask.sum())
            overl = np.bincount(plab[gmask], minlength=pn + 1)
            overl[0] = 0
            if overl.max() == 0:
                continue
            dsc = 2.0 * overl[1:] / np.maximum(psz[1:] + gsz, 1)
            claimed.add(int(np.argmax(dsc)) + 1)
    return len([i for i in range(1, pn + 1) if i not in claimed])


def official_fp(pred, gt):
    from panoptica import Panoptica_Evaluator
    ev = Panoptica_Evaluator.load_from_config(CFG)
    aff = np.eye(4)
    with tempfile.TemporaryDirectory() as td:
        pp, gp = f"{td}/p.nii.gz", f"{td}/g.nii.gz"
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), aff), pp)
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), aff), gp)
        r = ev.evaluate(pp, gp)
        d = r["et"].to_dict(True)          # use the ET group == label 3 exactly
        return d.get("fp", 0), d.get("tp", 0), d.get("fn", 0)


def box(a, z, y, x, s, v=3):
    a[z:z + s, y:y + s, x:x + s] = v


S = 64
tests = {}

# 1. perfect
g = np.zeros((S, S, S), np.uint8); p = np.zeros_like(g)
box(g, 10, 10, 10, 6); box(p, 10, 10, 10, 6)
tests["perfect match"] = (p, g, 0)

# 2. spurious far blob
g = np.zeros((S, S, S), np.uint8); p = np.zeros_like(g)
box(g, 10, 10, 10, 6); box(p, 10, 10, 10, 6); box(p, 40, 40, 40, 4)
tests["one spurious blob"] = (p, g, 1)

# 3. one GT lesion, prediction FRAGMENTED into two disjoint blobs on it
g = np.zeros((S, S, S), np.uint8); p = np.zeros_like(g)
g[10:20, 10:16, 10:16] = 3
p[10:13, 10:16, 10:16] = 3      # fragment A
p[17:20, 10:16, 10:16] = 3      # fragment B (disjoint from A)
tests["fragmented lesion"] = (p, g, 1)

# 4. two GT lesions, one missed entirely
g = np.zeros((S, S, S), np.uint8); p = np.zeros_like(g)
box(g, 10, 10, 10, 6); box(g, 40, 40, 40, 6); box(p, 10, 10, 10, 6)
tests["one lesion missed"] = (p, g, 0)

print(f"{'case':<22} {'my FP':>6} {'official FP':>12} {'off TP':>7} {'off FN':>7}  {'match':>6}")
print("-" * 70)
allok = True
for name, (p, g, expect) in tests.items():
    mine = my_fp_count(p == 3, g == 3)
    ofp, otp, ofn = official_fp(p, g)
    ok = (mine == ofp)
    allok &= ok
    print(f"{name:<22} {mine:>6} {ofp:>12} {otp:>7} {ofn:>7}  {'OK' if ok else 'MISMATCH':>6}")
print("\nFP_LOGIC_AGREES" if allok else "\nFP_LOGIC_MISMATCH -- loss would optimise the wrong target")
