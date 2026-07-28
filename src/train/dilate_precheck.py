"""Dilate-train / erode-infer FEASIBILITY pre-check -- run BEFORE spending a day training.

The idea: dilate small lesions in the training target so the net (which is BLIND to 43% of
them, peak<0.10) gets a bigger, easier object to fire on; erode the prediction back at
inference. dilate-then-erode is morphological CLOSING, so an ISOLATED lesion is recovered
exactly -- the danger is MERGING: dilation bridges two distinct mets closer than ~2k voxels
into one blob, and small-lesion F1 COUNTS INSTANCES, so every merge is a lost instance on the
metric we are trying to move. This measures that cost at k=1,2,3 on the 80-case GT.

Also reports, for the LARGE structures, DSC(close_k(gt), gt) -- the boundary distortion a
uniform dilate/erode would inflict on the DSC metrics where we sit at 94-99% of field-best.
"""
import os, json
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label, binary_dilation, binary_erosion, generate_binary_structure

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
VOL_THRESH = 20.0
STRUCT = np.ones((3, 3, 3), int)
BALL = generate_binary_structure(3, 1)   # 6-connectivity ball; iterate k times = radius-k
REG = {"wt": [1, 2, 3], "tc": [1, 3], "et": [3]}
KS = [1, 2, 3]


def one(cid):
    img = nib.load(f"{LE}/gt/{cid}.nii.gz")
    a = np.asarray(img.dataobj)
    vox = float(np.prod(img.header.get_zooms()[:3]))
    r = {}
    for reg, labs in REG.items():
        gt = np.isin(a, labs)
        glab, gn = cc_label(gt, STRUCT)
        # classify each original component small vs large
        small_ids, large_mask = [], np.zeros_like(gt)
        for i in range(1, gn + 1):
            m = glab == i
            if m.sum() * vox < VOL_THRESH:
                small_ids.append(i)
            else:
                large_mask |= m
        n_small = len(small_ids)
        per_k = {}
        for k in KS:
            dil = binary_dilation(gt, BALL, iterations=k)
            dlab, dn = cc_label(dil, STRUCT)
            # how many distinct SMALL originals landed in a dilated component that also
            # contains >=1 other original component -> merged (lost as a separate instance)
            merged = 0
            for i in small_ids:
                did = dlab[glab == i][0]
                # originals sharing this dilated component:
                origs_here = np.unique(glab[(dlab == did) & (glab > 0)])
                if len(origs_here) > 1:
                    merged += 1
            # large-structure closing distortion (recover original after dilate+erode)
            clo = binary_erosion(dil, BALL, iterations=k)
            inter = (clo & large_mask).sum()
            dsc = 2 * inter / (clo.sum() + large_mask.sum() + 1e-9) if large_mask.sum() else -1
            per_k[k] = (n_small, merged, dsc, large_mask.sum() > 0)
        r[reg] = per_k
    return r


if __name__ == "__main__":
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/gt/{c}.nii.gz")]
    print(f"[precheck] {len(cases)} cases | dilate small-lesion MERGE cost + large-DSC closing distortion", flush=True)
    agg = {reg: {k: [0, 0, [], 0] for k in KS} for reg in REG}  # n_small, merged, dscs, n_cases_with_large
    with Pool(12) as pool:
        for o in pool.imap_unordered(one, cases):
            for reg in REG:
                for k in KS:
                    ns, mg, dsc, haslarge = o[reg][k]
                    agg[reg][k][0] += ns
                    agg[reg][k][1] += mg
                    if haslarge:
                        agg[reg][k][2].append(dsc)
                        agg[reg][k][3] += 1
    print(f"\n{'region':6} {'k':>2} {'#small':>7} {'merged':>7} {'merge%':>7}   {'large-close-DSC':>15}")
    for reg in REG:
        for k in KS:
            ns, mg, dscs, _ = agg[reg][k]
            mp = 100 * mg / max(ns, 1)
            md = np.mean(dscs) if dscs else float('nan')
            print(f"{reg:6} {k:>2} {ns:>7} {mg:>7} {mp:>6.1f}%   {md:>15.4f}")
    print("\nRead: merge% = fraction of small GT lesions that would be FUSED into a neighbour by "
          "dilation-k (a lost F1 instance, best-case even with perfect training). "
          "large-close-DSC = ceiling of what uniform erode-k recovers on large structures.")
    print("PRECHECK_COMPLETE")
