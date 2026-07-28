"""Dataset015 — ISOLATED-small-lesion label DILATION (k=2), no inference erosion.

THE PROBLEM (measured): the net is BLIND (peak<0.10) to ~half the small mets it misses.
SmallCC (D014) attacked this by SAMPLING small lesions 25x more; it did not un-blind them.
This attacks the same blindness from the other side: make the TARGET bigger. A 30-voxel met
grown by k=2 becomes ~4x the volume -> more gradient per lesion AND a spatially larger object
a 3D receptive field can latch onto.

WHY IT IS SAFE (pre-checked on 80 GT cases):
  - Only ISOLATED small mets are dilated (no other met within 2k voxels) -> ZERO instance
    merging. ~100% of blind small mets are isolated at k=1, ~88% at k=2.
  - LARGE structures are untouched -> no large-structure DSC collateral.
  - Shell is painted ONLY into background/SNFH (labels 0,2), never over another met, RC(4),
    or the ignore label (-1). No component-count change at inference, so NO erosion needed:
    small predictions come out ~2vox oversized, which is negligible for instance-F1 and for
    the tiny volumetric-DSC weight of small lesions.

MECHANISM ONLY CHANGES THE TARGET the loss sees. Sampling (.pkl class_locations) and image
data (.b2nd) are symlinked from D007 unchanged -- original lesion voxels are still valid
foreground centres inside the enlarged lesion. Trainer = stock DiceTopK10BCE, FT from
checkpoint_topk_snap. Loss untouched.
"""
import os, sys, glob, pickle, shutil, json
import numpy as np
import blosc2
from multiprocessing import Pool
from scipy.ndimage import (label as cc_label, binary_dilation,
                           generate_binary_structure, find_objects)

HOME = os.path.expanduser("~")
PRE = f"{HOME}/projects/CAI4CAI/SegData/preprocessed"
SRC = f"{PRE}/Dataset007_BraTSMets2025"
DST = f"{PRE}/Dataset015_BraTSMetsDilateISO"
SRC_D = f"{SRC}/nnUNetPlans_3d_fullres"
DST_D = f"{DST}/nnUNetPlans_3d_fullres"

SMALL_MAX = 200                 # match D014's small-lesion definition (voxels)
K = 2                           # dilation radius
STRUCT = np.ones((3, 3, 3), int)
BALL = generate_binary_structure(3, 1)
TC = [1, 3]                     # metastasis = tumour core (NETC=1, ET=3)
PAINT_INTO = [0, 2]            # grow only into background / SNFH -- never over met/RC(-1 ignore)


def build_one(pkl):
    cid = os.path.basename(pkl)[:-4]
    src_seg = blosc2.open(f"{SRC_D}/{cid}_seg.b2nd")
    seg = np.asarray(src_seg[:])            # (1, X, Y, Z) int16
    s = seg[0]
    shape = s.shape
    tc = np.isin(s, TC)
    n_small = n_dil = n_skip_clustered = 0
    if tc.any():
        lab, n = cc_label(tc, STRUCT)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        objs = find_objects(lab)
        for i in range(1, n + 1):
            if sizes[i] == 0 or sizes[i] >= SMALL_MAX:
                continue
            n_small += 1
            sl = objs[i - 1]
            pad = 2 * K + 1
            sub = tuple(slice(max(0, a.start - pad), min(d, a.stop + pad))
                        for a, d in zip(sl, shape))
            lab_sub = lab[sub]; s_sub = s[sub]
            comp = lab_sub == i
            other_tc = np.isin(s_sub, TC) & (lab_sub != i)
            # isolated iff growing this met by 2k reaches no other met
            if (binary_dilation(comp, BALL, iterations=2 * K) & other_tc).any():
                n_skip_clustered += 1
                continue
            has_et = bool((s_sub[comp] == 3).any())
            dil = binary_dilation(comp, BALL, iterations=K)
            shell = dil & ~comp & np.isin(s_sub, PAINT_INTO)
            s_sub[shell] = 3 if has_et else 1
            s[sub] = s_sub
            n_dil += 1
    # write new dilated seg, preserving exact blosc2 layout + dtype
    dst = f"{DST_D}/{cid}_seg.b2nd"
    if os.path.exists(dst):
        os.remove(dst)
    blosc2.asarray(np.ascontiguousarray(seg, dtype=src_seg.dtype),
                   urlpath=dst, chunks=src_seg.chunks, blocks=src_seg.blocks)
    # symlink unchanged image .b2nd and properties .pkl
    for ext in (f"{cid}.b2nd", f"{cid}.pkl"):
        lnk = f"{DST_D}/{ext}"
        if not os.path.exists(lnk):
            os.symlink(f"{SRC_D}/{ext}", lnk)
    return cid, n_small, n_dil, n_skip_clustered


def setup_dataset_files():
    os.makedirs(DST_D, exist_ok=True)
    for f in ("dataset.json", "dataset_fingerprint.json"):
        if os.path.exists(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    # PLANS: copy AND patch dataset_name -> Dataset015 (the §4 trap: nnU-Net reads the
    # preprocessed+results paths from dataset_name INSIDE the plans, not from -d 15)
    pj = json.load(open(f"{SRC}/nnUNetResEncUNetLPlans.json"))
    pj["dataset_name"] = "Dataset015_BraTSMetsDilateISO"
    json.dump(pj, open(f"{DST}/nnUNetResEncUNetLPlans.json", "w"), indent=1)
    if not os.path.exists(f"{DST}/gt_segmentations"):
        os.symlink(f"{SRC}/gt_segmentations", f"{DST}/gt_segmentations")   # ORIGINAL gt for eval


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # >0 = test on first N cases
    setup_dataset_files()
    pkls = sorted(glob.glob(f"{SRC_D}/*.pkl"))
    if LIMIT:
        pkls = pkls[:LIMIT]
    print(f"[d015] {len(pkls)} cases | dilate ISOLATED small mets (<{SMALL_MAX}vox) by k={K} "
          f"into {PAINT_INTO}; large + clustered untouched", flush=True)
    tot_s = tot_d = tot_c = 0; ncase_dil = 0
    with Pool(NPROC) as pool:
        for i, (cid, ns, nd, nc) in enumerate(pool.imap_unordered(build_one, pkls, chunksize=8)):
            tot_s += ns; tot_d += nd; tot_c += nc; ncase_dil += (nd > 0)
            if (i + 1) % 300 == 0:
                print(f"  ..{i+1}/{len(pkls)}", flush=True)
    print(f"\n  small mets total       : {tot_s:,}")
    print(f"  DILATED (isolated)     : {tot_d:,}  across {ncase_dil} cases")
    print(f"  skipped (clustered)    : {tot_c:,}  ({100*tot_c/max(tot_s,1):.0f}% of small)")
    print("BUILD_D015_COMPLETE")
