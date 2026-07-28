"""Build the RC (cavity) eval set — the blind spot in the harness.

167 training cases carry a resection cavity. Our tumour eval cohort has ZERO, so
RC DSC + RC NSD (2 of the 11 ranked metrics) could never be tuned locally. We sit
at .630; teams #3 and #4 post .671.

Take 100 of the 167 (deterministic, stratified by cavity volume so the tiny ones --
which are where a threshold or a largest-component rule actually bites -- are
represented, not just the big obvious voids).
"""
import os, glob, json, shutil
import numpy as np
import nibabel as nib

HOME = os.path.expanduser("~")
SRC = f"{HOME}/brats2025/extracted_rc"
OUT_IMG = f"{HOME}/brats2025/rceval/images"
OUT_GT = f"{HOME}/brats2025/rceval/gt"
N = 100
MODS = ["t1c", "t1n", "t2f", "t2w"]

cases = sorted(os.path.basename(d) for d in glob.glob(SRC + "/BraTS-MET-*"))
print(f"cavity cases available: {len(cases)}")

# measure cavity volume + component count per case
info = []
for c in cases:
    a = np.asarray(nib.load(f"{SRC}/{c}/{c}-seg.nii.gz").dataobj)
    vox = int((a == 4).sum())
    info.append((c, vox))
info.sort(key=lambda x: x[1])

# stratify by volume: take every k-th across the sorted range so small AND large are in
idx = np.linspace(0, len(info) - 1, min(N, len(info))).astype(int)
sel = [info[i] for i in sorted(set(idx.tolist()))]

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_GT, exist_ok=True)
for c, _ in sel:
    for i, m in enumerate(MODS):
        dst = f"{OUT_IMG}/{c}_{i:04d}.nii.gz"
        if not os.path.exists(dst):
            os.symlink(f"{SRC}/{c}/{c}-{m}.nii.gz", dst)
    g = f"{OUT_GT}/{c}.nii.gz"
    if not os.path.exists(g):
        shutil.copy(f"{SRC}/{c}/{c}-seg.nii.gz", g)

vols = [v for _, v in sel]
print(f"\nRC eval set: {len(sel)} cases")
print(f"  cavity volume  min {min(vols)}  median {int(np.median(vols))}  max {max(vols)} vox")
print(f"  tiny (<100 vox): {sum(1 for v in vols if v < 100)} cases  <- where thresholds bite")
print(f"  images -> {OUT_IMG}  ({len(glob.glob(OUT_IMG + '/*.nii.gz'))} files)")
print(f"  gt     -> {OUT_GT}   ({len(glob.glob(OUT_GT + '/*.nii.gz'))} files)")
json.dump([c for c, _ in sel], open(f"{HOME}/brats2025/rceval/cases.json", "w"))
print("BUILD_RC_COMPLETE")
