"""Union recall recovery for the TopK⊕FT ensemble (BraTS-METS Task 1).

WHY: the 50/50 probability-average ensemble (sub 9770198) throws away small lesions
that only ONE parent found — a lesion at prob 0.8 in TopK but 0.0 in FT averages to 0.4,
falls under the 0.5 threshold, and is deleted. That lost detection is the F1 leak.

FIX: take the ensemble mask as the base, then OR-back whole connected-components that the
TopK standalone mask contains but the base is MISSING ENTIRELY. Restricted to the tumour-core
labels (NETC=1, ET=3) so we lift TC/ET detection F1 without touching edema(2)/RC(4). We paste
only into background voxels of the base, so every existing structure — and thus DSC/NSD — is
untouched. This is NOT a re-threshold (no global cut); it surgically re-injects missed lesions.

Label scheme: NETC=1, SNFH/edema=2, ET=3, RC=4. Tumour core (TC) = {1,3}; ET = {3}.

USAGE (sie272, env ~/envs/brats):
  # dry-run: report how many lesions would be recovered, write nothing
  python union_recover.py --base val_infer/ens_ft_5050 --topk val_infer/pred_topk_best --dry-run
  # apply: write recovered masks to --out (then RC-fix + verify + zip as usual)
  python union_recover.py --base val_infer/ens_ft_5050 --topk val_infer/pred_topk_best \
      --out val_infer/ens_ft_5050_union
Post: RC is untouched by the ET/TC union, so the locked RC-fix + verify gate + zip -j still apply.
"""
import argparse
import glob
import os

import cc3d
import nibabel as nib
import numpy as np

TC_LABELS = (1, 3)  # tumour core = NETC + ET; ET (=3) is the small-lesion axis we care about


def tc_mask(a):
    return (a == 1) | (a == 3)


def recover_one(base, topk, max_overlap_frac, connectivity):
    """Return (new_base, n_recovered_cc, n_recovered_vox).

    A TopK tumour-core connected-component is 'missed' if its overlap with the base tumour-core
    is <= max_overlap_frac (default 0.0 = base contains NONE of it). Missed CCs are pasted with
    their TopK labels (1/3) into background voxels of the base only.
    """
    base_tc = tc_mask(base)
    topk_tc = tc_mask(topk)
    out = base.copy()
    cc = cc3d.connected_components(topk_tc.astype(np.uint8), connectivity=connectivity)
    n_cc = n_vox = 0
    sizes = recover_one.sizes  # module-level accumulator for the size histogram
    for lbl in range(1, cc.max() + 1):
        comp = cc == lbl
        csize = int(comp.sum())
        if csize == 0:
            continue
        overlap = int((comp & base_tc).sum())
        if overlap / csize <= max_overlap_frac:
            paste = comp & (out == 0)  # background of base only — never overwrite existing labels
            npaste = int(paste.sum())
            if npaste == 0:
                continue  # CC lands entirely on existing edema/RC — nothing added, don't count it
            out[paste] = topk[paste]   # writes 1 or 3 exactly as TopK predicted them
            n_cc += 1
            n_vox += npaste
            sizes.append(npaste)
    return out, n_cc, n_vox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ensemble mask dir (the submitted base, e.g. ens_ft_5050)")
    ap.add_argument("--topk", required=True, help="TopK standalone mask dir (recall-strong source)")
    ap.add_argument("--out", help="output dir for recovered masks (omit with --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="report stats only, write nothing")
    ap.add_argument("--max-overlap-frac", type=float, default=0.0,
                    help="CC counts as MISSED if its overlap with base TC <= this (0.0 = fully absent)")
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    a = ap.parse_args()

    base_files = sorted(glob.glob(os.path.join(a.base, "*.nii.gz")))
    assert base_files, f"no .nii.gz in {a.base}"
    if not a.dry_run:
        assert a.out, "--out required unless --dry-run"
        os.makedirs(a.out, exist_ok=True)

    recover_one.sizes = []
    tot_cc = tot_vox = cases_changed = 0
    for bf in base_files:
        name = os.path.basename(bf)
        tf = os.path.join(a.topk, name)
        if not os.path.exists(tf):
            print(f"  WARN no TopK match for {name} — copying base unchanged")
            if not a.dry_run:
                nib.save(nib.load(bf), os.path.join(a.out, name))
            continue
        bimg = nib.load(bf)
        base = np.asanyarray(bimg.dataobj).astype(np.uint8)
        topk = np.asanyarray(nib.load(tf).dataobj).astype(np.uint8)
        assert base.shape == topk.shape, f"shape mismatch {name}: {base.shape} vs {topk.shape}"
        out, ncc, nvox = recover_one(base, topk, a.max_overlap_frac, a.connectivity)
        if ncc:
            cases_changed += 1
            tot_cc += ncc
            tot_vox += nvox
            print(f"  {name}: +{ncc} lesion(s), +{nvox} vox "
                  f"(ET vox +{int(((out == 3) & (base == 0)).sum())})")
        if not a.dry_run:
            nib.save(nib.Nifti1Image(out, bimg.affine, bimg.header), os.path.join(a.out, name))

    sizes = np.array(recover_one.sizes)
    print(f"\n{'DRY-RUN ' if a.dry_run else ''}SUMMARY: {cases_changed}/{len(base_files)} cases changed, "
          f"{tot_cc} lesions recovered, {tot_vox} voxels added.")
    if sizes.size:
        buckets = [(1, 3), (4, 10), (11, 50), (51, 200), (201, 10**9)]
        print("  recovered-lesion size histogram (voxels):")
        for lo, hi in buckets:
            n = int(((sizes >= lo) & (sizes <= hi)).sum())
            print(f"    {lo:>4}-{hi if hi < 10**9 else '+':<4}: {n}")
        print(f"  median {int(np.median(sizes))}, max {int(sizes.max())} vox")
    if not a.dry_run:
        print(f"Wrote {a.out}. Next: postprocess.py --rc-largest 1 -> verify -> zip -j -> submit.")


if __name__ == "__main__":
    main()
