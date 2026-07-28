"""Does Dataset014 ACTUALLY change what the network sees?

The banner proves the settings loaded. It does not prove the sampler behaves differently.
So: replicate nnU-Net's own get_bbox selection logic verbatim, run it thousands of times on
real cases, and measure what fraction of patches are centred ON A SMALL LESION -- in D007
(the old sampling) versus D014 (the new key).

Also validates the coordinates themselves: every voxel in the new key must actually lie
inside a small tumour component of the PREPROCESSED seg (same space as the existing keys).
"""
import os, pickle, glob
import numpy as np
import blosc2
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
PRE = f"{HOME}/projects/CAI4CAI/SegData/preprocessed"
D007 = f"{PRE}/Dataset007_BraTSMets2025/nnUNetPlans_3d_fullres"
D014 = f"{PRE}/Dataset014_BraTSMetsSmallCC/nnUNetPlans_3d_fullres"
SMALL_MAX = 200
NEW_KEY = (1, 2, 3, 99)
STRUCT = np.ones((3, 3, 3), int)
N_DRAWS = 4000
ANNOTATED = (1, 2, 3, 4)          # nnU-Net's annotated_classes_key -- gets popped if present


def pick(class_locations, rng):
    """nnU-Net's get_bbox selection logic, verbatim."""
    eligible = [k for k in class_locations.keys() if len(class_locations[k]) > 0]
    tmp = [k == ANNOTATED if isinstance(k, tuple) else False for k in eligible]
    if any(tmp) and len(eligible) > 1:
        eligible.pop(int(np.where(tmp)[0][0]))
    if not eligible:
        return None, None
    k = eligible[rng.integers(len(eligible))]
    v = class_locations[k]
    return k, v[rng.integers(len(v))]


def main():
    rng = np.random.default_rng(0)
    cases = [os.path.basename(f)[:-4] for f in sorted(glob.glob(D014 + "/*.pkl"))]
    # only cases that HAVE small lesions -- those are the ones the change is for
    usable = []
    for c in cases:
        d = pickle.load(open(f"{D014}/{c}.pkl", "rb"))
        if len(d["class_locations"].get(NEW_KEY, [])) > 0:
            usable.append(c)
        if len(usable) >= 12:
            break
    print(f"testing on {len(usable)} cases that contain small lesions\n")

    tot = {"d007": 0, "d014": 0}
    hits = {"d007": 0, "d014": 0}
    coord_ok = coord_bad = 0

    for c in usable:
        seg = np.asarray(blosc2.open(f"{D014}/{c}_seg.b2nd")[:])[0]
        wt = np.isin(seg, [1, 2, 3])
        lab, n = cc_label(wt, STRUCT)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        small_ids = {i for i in range(1, n + 1) if 0 < sizes[i] < SMALL_MAX}
        small_mask = np.isin(lab, list(small_ids)) if small_ids else np.zeros_like(wt)

        d7 = pickle.load(open(f"{D007}/{c}.pkl", "rb"))["class_locations"]
        d14 = pickle.load(open(f"{D014}/{c}.pkl", "rb"))["class_locations"]

        # --- are the NEW coords actually inside small lesions?
        for row in np.asarray(d14[NEW_KEY]):
            _, x, y, z = row
            if small_mask[x, y, z]:
                coord_ok += 1
            else:
                coord_bad += 1

        # --- how often does each sampling scheme centre a patch on a small lesion?
        for name, cl in (("d007", d7), ("d014", d14)):
            for _ in range(N_DRAWS // len(usable)):
                k, v = pick(cl, rng)
                if v is None:
                    continue
                tot[name] += 1
                _, x, y, z = v
                if small_mask[x, y, z]:
                    hits[name] += 1

    print("=== coordinate validity (new key) ===")
    print(f"  voxels inside a small lesion : {coord_ok}")
    print(f"  voxels NOT inside one        : {coord_bad}   {'<-- BUG' if coord_bad else '(good)'}")

    print("\n=== how often is a patch CENTRED on a small lesion? ===")
    for name in ("d007", "d014"):
        p = 100 * hits[name] / max(tot[name], 1)
        print(f"  {name.upper():<5} {hits[name]:5d} / {tot[name]:5d} draws = {p:5.2f}%")
    r7 = hits["d007"] / max(tot["d007"], 1)
    r14 = hits["d014"] / max(tot["d014"], 1)
    print(f"\n  increase: {r14/max(r7,1e-9):.1f}x")
    print("\n  (these are FOREGROUND draws. With oversample_foreground_percent=0.5, half of all")
    print("   patches are foreground-forced, so multiply by 0.5 for the share of ALL patches.)")
    if r14 > 5 * max(r7, 1e-9) and coord_bad == 0:
        print("\n  => D014 IS WORKING. The network now sees small lesions far more often.")
    else:
        print("\n  => SOMETHING IS WRONG.")


if __name__ == "__main__":
    main()
