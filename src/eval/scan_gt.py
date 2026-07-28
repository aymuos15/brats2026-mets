"""Scan local GT (650 training cases with labels) to build a stratified eval set.

Records per case: has RC(4), #WT instances, #SMALL WT instances (<20 mm^3 = the
official vol_threshold from metrics_parser.py), total tumour volume.
We need small lesions well represented -- small_instance_f1 is the metric we're
chasing, and it is NaN (excluded) for any case with no small lesions at all.
"""
import os, glob, json, numpy as np, nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

TRAIN = os.path.expanduser("~/brats2025/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training")
VOL_THRESH = 20.0  # mm^3, official default in metrics_parser.py
STRUCT = np.ones((3, 3, 3), int)  # panoptica CCA uses full connectivity


def one(d):
    cid = os.path.basename(d)
    f = os.path.join(d, f"{cid}-seg.nii.gz")
    if not os.path.exists(f):
        return None
    img = nib.load(f)
    a = np.asarray(img.dataobj).astype(np.uint8)
    vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    wt = np.isin(a, [1, 2, 3])
    lab, n = cc_label(wt, STRUCT)
    small = large = 0
    if n:
        sizes = np.bincount(lab.ravel())[1:] * vox_mm3
        small = int((sizes < VOL_THRESH).sum())
        large = int((sizes >= VOL_THRESH).sum())
    return {"case": cid, "has_rc": bool((a == 4).any()),
            "wt_inst": int(n), "wt_small": small, "wt_large": large,
            "tumour_vox": int(wt.sum())}


if __name__ == "__main__":
    dirs = sorted(glob.glob(TRAIN + "/BraTS-MET-*"))
    print(f"scanning {len(dirs)} cases...", flush=True)
    with Pool(16) as p:
        res = [r for r in p.map(one, dirs, chunksize=4) if r]
    json.dump(res, open(os.path.expanduser("~/brats2025/gt_scan.json"), "w"))
    n_rc = sum(r["has_rc"] for r in res)
    n_small = sum(r["wt_small"] > 0 for r in res)
    print(f"cases with GT      : {len(res)}")
    print(f"cases with RC      : {n_rc}  ({100*n_rc/len(res):.1f}%)   [expect ~13%]")
    print(f"cases with >=1 SMALL WT lesion (<{VOL_THRESH} mm3): {n_small} ({100*n_small/len(res):.1f}%)")
    print(f"  -> only these contribute to small_instance_f1 (else NaN/excluded)")
    print(f"total WT instances : {sum(r['wt_inst'] for r in res)}")
    print(f"total SMALL WT inst: {sum(r['wt_small'] for r in res)}")
    print(f"total LARGE WT inst: {sum(r['wt_large'] for r in res)}")
    print("SCAN_COMPLETE")
