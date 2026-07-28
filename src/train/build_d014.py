"""Dataset014 — small-lesion-targeted patch sampling.

THE PROBLEM (measured, 2026-07-14): the network is COMPLETELY BLIND (peak prob < 0.10)
to 43% of the small lesions it misses. Not under-threshold -- invisible. No decode
recovers those; only training can.

WHY IT IS BLIND: nnU-Net's sampler picks a random KEY from class_locations, then a random
VOXEL from it. Small metastases live inside the (1,2,3) key alongside every SNFH and
large-lesion voxel in the case, so a 30-voxel met is patch-centred ~0.3% of the time. It
contributes almost no gradient no matter what loss you use. FgOversample (0.33 -> 0.50)
failed for exactly this reason: it oversamples FOREGROUND, which is the wrong distribution.

THE FIX: add a NEW key holding only small-component voxels. The sampler picks keys
UNIFORMLY, so the new key wins 1 time in 4 -> a small met is patch-centred ~8% of the time.
A ~25x increase. And within the key we store voxels weighted EQUALLY PER LESION, not per
voxel -- the blob-loss principle, applied to sampling.

Stock nnU-Net: the sampler uses `selected_class` only to index the dict, so an arbitrary
key is safe. No trainer changes, no re-preprocessing -- the .b2nd data is symlinked.
"""
import os, sys, glob, pickle, shutil
import numpy as np
import blosc2
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
PRE = f"{HOME}/projects/CAI4CAI/SegData/preprocessed"
SRC = f"{PRE}/Dataset007_BraTSMets2025"
DST = f"{PRE}/Dataset014_BraTSMetsSmallCC"
SRC_D = f"{SRC}/nnUNetPlans_3d_fullres"
DST_D = f"{DST}/nnUNetPlans_3d_fullres"

SMALL_MAX = 200          # voxels. the metric scores "small" as <20mm^3 (~20 vox); we go a
                         # little wider so the net learns the whole small-lesion regime.
PER_COMP = 6             # voxels stored per lesion -> every lesion gets EQUAL weight,
                         # regardless of how many voxels it has
NEW_KEY = (1, 2, 3, 99)  # a tuple that cannot collide with the real region keys
STRUCT = np.ones((3, 3, 3), int)


def one(pkl):
    cid = os.path.basename(pkl)[:-4]
    seg = np.asarray(blosc2.open(f"{SRC_D}/{cid}_seg.b2nd")[:])   # (1, X, Y, Z)
    d = pickle.load(open(pkl, "rb"))
    wt = np.isin(seg[0], [1, 2, 3])
    rows = []
    n_small = n_large = 0
    if wt.any():
        lab, n = cc_label(wt, STRUCT)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        for i in range(1, n + 1):
            if sizes[i] == 0:
                continue
            if sizes[i] >= SMALL_MAX:
                n_large += 1
                continue
            n_small += 1
            xs, ys, zs = np.where(lab == i)
            k = min(PER_COMP, len(xs))
            pick = np.random.choice(len(xs), k, replace=False)
            for j in pick:
                rows.append([0, xs[j], ys[j], zs[j]])   # leading 0 = the channel dim
    cl = dict(d["class_locations"])
    cl[NEW_KEY] = (np.array(rows, dtype=np.int64) if rows
                   else np.zeros((0, 4), dtype=np.int64))
    d["class_locations"] = cl
    pickle.dump(d, open(f"{DST_D}/{cid}.pkl", "wb"))
    return cid, n_small, n_large, len(rows)


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.makedirs(DST_D, exist_ok=True)
    # dataset.json + plans + fingerprint: 4-REGION scheme (no BackSplit -- it costs F1)
    for f in ("dataset.json", "nnUNetResEncUNetLPlans.json", "dataset_fingerprint.json"):
        if os.path.exists(f"{SRC}/{f}") and not os.path.exists(f"{DST}/{f}"):
            shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    if not os.path.exists(f"{DST}/gt_segmentations"):
        os.symlink(f"{SRC}/gt_segmentations", f"{DST}/gt_segmentations")
    # symlink the heavy data -- no re-preprocessing
    for b in glob.glob(f"{SRC_D}/*.b2nd"):
        dst = f"{DST_D}/{os.path.basename(b)}"
        if not os.path.exists(dst):
            os.symlink(b, dst)

    pkls = sorted(glob.glob(f"{SRC_D}/*.pkl"))
    print(f"[d014] {len(pkls)} cases | small = WT component < {SMALL_MAX} vox | "
          f"{PER_COMP} voxels per lesion (EQUAL weight per lesion)", flush=True)
    tot_s = tot_l = tot_r = 0
    with_small = 0
    with Pool(NPROC) as pool:
        for i, (cid, ns, nl, nr) in enumerate(pool.imap_unordered(one, pkls, chunksize=8)):
            tot_s += ns; tot_l += nl; tot_r += nr
            with_small += (ns > 0)
            if (i + 1) % 300 == 0:
                print(f"  ..{i+1}/{len(pkls)}", flush=True)

    print(f"\n  small lesions found : {tot_s:,}  across {with_small} / {len(pkls)} cases")
    print(f"  large lesions       : {tot_l:,}")
    print(f"  sampling rows added : {tot_r:,}")
    print(f"\n  the sampler picks a key UNIFORMLY. With this key present a typical case has")
    print(f"  4 eligible keys -> a small lesion is patch-centred ~{100*0.33/4:.0f}% of the time,")
    print(f"  against ~0.3% today. A ~25x increase in the gradient the network sees.")
    # verify one
    d = pickle.load(open(f"{DST_D}/{os.path.basename(pkls[0])}", "rb"))
    print(f"\n  verify {os.path.basename(pkls[0])}: keys = {list(d['class_locations'].keys())}")
    for k, v in d["class_locations"].items():
        print(f"     {str(k):<16} {np.asarray(v).shape}")
    print("BUILD_D014_COMPLETE")
