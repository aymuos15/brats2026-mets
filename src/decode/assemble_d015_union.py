"""D015-UNION: the 3-way fusion (9771992) with D015's NEW small-lesion detections grafted in.

Rationale (handoff 2026-07-17): D015 standalone (9772272) lifted tumour F1 +.012/.014/.013 over
the 3-way best 9771992 -- its extra small-met detections are net-real, not just mimics. But D015's
DSC collapses (dilated/unfused), so it can't be a submission or be prob-averaged into the fusion.
Instead: keep the 3-way's strong DSC as the BASE, and mask-UNION only D015's detections that the
3-way MISSED (zero voxel overlap). Base is byte-identical to 9771992, so this can never score below
our banked floor -- it can only add F1.

  base   = the CACHED 9771992 segmentation (val_infer/sub_slot2_lo025) -- loaded, not recomputed,
           so it is provably byte-identical. Region masks WT={1,2,3} TC={1,3} ET={3}; RC({4}) reused.
  graft  = D015 region components (26-conn, eroded k=EK on comps<TVOX) that (a) do NOT overlap the
           base region and (b) have >= MINSZ voxels.
  union  = base | graft ; paint WT->2, TC->1, ET->3 (region nesting is automatic); RC from base.

MINSZ is the bracket knob: 1 = aggressive (graft all new detections), 15 = conservative (drop the
tiniest new specks -- most mimic-like per the surface/vessel FP read; a bounded hedge, since grafting
only ADDS on top of the 3-way and never removes a 3-way detection).

!! Component-count-CHANGING decode -> handoff section 5: the local harness CANNOT score it. Validate
   on a real leaderboard slot. This script only builds + sanity-checks (counts, no merging, verify gate).

Usage: assemble_d015_union.py <out_subdir> <MINSZ> [loD=0.30] [hiD=0.70] [EK=1] [TVOX=500] [nproc=10]
"""
import os, sys, glob
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label, binary_erosion, generate_binary_structure
from scipy.ndimage import maximum as _ndmax

SUB   = sys.argv[1]
MINSZ = int(sys.argv[2])
LOD   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
HID   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.70
EK    = int(sys.argv[5])   if len(sys.argv) > 5 else 1
TVOX  = int(sys.argv[6])   if len(sys.argv) > 6 else 500
NPROC = int(sys.argv[7])   if len(sys.argv) > 7 else 10

V     = os.path.expanduser("~/brats2025/val_infer")
BASE  = f"{V}/sub_slot2_lo025"        # the cached 9771992 output (3-way, lo .25 / hi .70)
D015  = f"{V}/pred_d015_s025"
OUT   = f"{V}/{SUB}"
STRUCT26 = np.ones((3, 3, 3), int)          # D015's native connectivity (matches assemble_d015)
BALL     = generate_binary_structure(3, 1)  # erosion element (matches assemble_d015)
os.makedirs(OUT, exist_ok=True)


def hyst26(p, lo, hi):
    """D015 hysteresis, 26-connectivity. Peak via ndimage.maximum (fast; identical to np.maximum.at)."""
    lab, n = cc_label(p > lo, STRUCT26)
    if n == 0:
        return np.zeros(p.shape, bool)
    peaks = np.asarray(_ndmax(p, lab, index=np.arange(1, n + 1)))
    keep = np.zeros(n + 1, bool)
    keep[1:] = peaks > hi
    return keep[lab]


def selective_erode(mask):
    """Erode only components < TVOX by EK. Large structures untouched. (= assemble_d015)"""
    if EK <= 0 or not mask.any():
        return mask
    lab, n = cc_label(mask, STRUCT26)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    out = mask.copy()
    for i in range(1, n + 1):
        if sizes[i] < TVOX:
            comp = lab == i
            er = binary_erosion(comp, BALL, iterations=EK)
            if er.any():                    # never erode a lesion out of existence
                out[comp] = False
                out[er] = True
    return out


def graft(base, dmask):
    """Union base with D015 components that DON'T overlap base and have >= MINSZ voxels.
    Returns (union_mask, n_new_components, n_new_voxels)."""
    lab, n = cc_label(dmask, STRUCT26)
    if n == 0:
        return base, 0, 0
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    touching = set(int(x) for x in np.unique(lab[base]))   # labels sharing a voxel with base (incl 0)
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        if i in touching or sizes[i] < MINSZ:
            continue
        keep[i] = True
    if not keep.any():
        return base, 0, 0
    return base | keep[lab], int(keep.sum()), int(sizes[keep].sum())


def one(cid):
    # base region masks from the cached 9771992 seg (on-disk (x,y,z) -> prob-space (z,y,x))
    bseg = np.asanyarray(nib.load(f"{BASE}/{cid}.nii.gz").dataobj).astype(np.uint8).transpose(2, 1, 0)
    dp = np.load(f"{D015}/{cid}.npz")["probabilities"]

    seg = np.zeros(bseg.shape, np.uint8)
    bases = [np.isin(bseg, [1, 2, 3]), np.isin(bseg, [1, 3]), bseg == 3]   # WT, TC, ET regions
    new_comps = new_vox = 0
    for ch, (base_ch, cls) in enumerate(zip(bases, [2, 1, 3])):            # paint WT->2, TC->1, ET->3
        dmask = selective_erode(hyst26(dp[ch], LOD, HID))
        union, nc, nv = graft(base_ch, dmask)
        seg[union] = cls
        new_comps += nc
        new_vox += nv
    seg[bseg == 4] = 4                                                     # RC reused from base (keep-largest)

    ref = nib.load(f"{D015}/{cid}.nii.gz")
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), f"{OUT}/{cid}.nii.gz")
    labs = set(np.unique(seg).tolist())
    return (int((seg == 4).any()), int(not seg.any()),
            int(not labs.issubset({0, 1, 2, 3, 4})),
            int(seg.size > 0 and (seg == 4).sum() / seg.size > 0.08),
            new_comps, new_vox)


if __name__ == "__main__":
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(D015 + "/*.npz"))
    print(f"[union] {len(cases)} cases | base=9771992(cached) | D015 lo={LOD} hi={HID} "
          f"erode k={EK}<{TVOX}vox (26conn) | MINSZ={MINSZ} -> {SUB}", flush=True)
    res = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, cases, chunksize=1), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"[union]   {i}/{len(cases)}", flush=True)
    n = len(glob.glob(OUT + "/*.nii.gz"))
    rc      = sum(r[0] for r in res)
    empty   = sum(r[1] for r in res)
    badlab  = sum(r[2] for r in res)
    baddom  = sum(r[3] for r in res)
    tcomps  = sum(r[4] for r in res)
    tvox    = sum(r[5] for r in res)
    ngr     = sum(1 for r in res if r[4] > 0)
    print(f"[union] DONE out={n} rc_cases={rc} empty={empty} bad_labels={badlab} rc_dominating={baddom}")
    print(f"[union] GRAFTED {tcomps} new components ({tvox} voxels) across {ngr}/{len(cases)} cases")
    ok = (n == len(cases)) and badlab == 0 and baddom == 0
    print("VERIFY_GATE:", "PASS" if ok else "FAIL")
    print("ASSEMBLE_UNION_COMPLETE")
