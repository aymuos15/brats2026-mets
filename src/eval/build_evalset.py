"""Build the local eval set: cases that actually determine small_instance_f1.

Per the official metrics_parser.py, a case with NO small GT lesion returns NaN for
small_instance_f1 and is EXCLUDED from the mean. So the metric is decided entirely
by the subset of cases that contain >=1 small (<20 mm^3) lesion. We draw the eval
set from those, which maximises signal per GPU-hour.

Stages the 4 modalities into nnUNet predict layout (_0000.._0003) and copies GT.
"""
import os, glob, json, shutil, numpy as np, nibabel as nib

HOME = os.path.expanduser("~")
TRAIN = f"{HOME}/brats2025/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
OUT_IMG = f"{HOME}/brats2025/localeval/images"
OUT_GT = f"{HOME}/brats2025/localeval/gt"
N = 80
MODS = ["t1c", "t1n", "t2f", "t2w"]     # channel order 0..3 per dataset.json

# sanity: what labels exist
u = set()
for d in sorted(glob.glob(TRAIN + "/BraTS-MET-*"))[:40]:
    c = os.path.basename(d)
    u |= set(np.unique(np.asarray(nib.load(f"{d}/{c}-seg.nii.gz").dataobj)).tolist())
print("unique GT labels (40-case sample):", sorted(int(x) for x in u))

scan = json.load(open(f"{HOME}/brats2025/gt_scan.json"))
pool = [r for r in scan if r["wt_small"] > 0]
pool.sort(key=lambda r: r["case"])           # deterministic
rng = np.random.RandomState(1234)
idx = rng.choice(len(pool), size=min(N, len(pool)), replace=False)
sel = [pool[i] for i in sorted(idx)]

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_GT, exist_ok=True)
for r in sel:
    c = r["case"]
    for i, m in enumerate(MODS):
        src = f"{TRAIN}/{c}/{c}-{m}.nii.gz"
        dst = f"{OUT_IMG}/{c}_{i:04d}.nii.gz"
        if not os.path.exists(dst):
            os.symlink(src, dst)
    g = f"{OUT_GT}/{c}.nii.gz"
    if not os.path.exists(g):
        shutil.copy(f"{TRAIN}/{c}/{c}-seg.nii.gz", g)

print(f"\neval set: {len(sel)} cases (all contain >=1 small lesion)")
print(f"  small WT instances: {sum(r['wt_small'] for r in sel)}")
print(f"  large WT instances: {sum(r['wt_large'] for r in sel)}")
print(f"  images -> {OUT_IMG}  ({len(glob.glob(OUT_IMG+'/*.nii.gz'))} files = {len(sel)}x4)")
print(f"  gt     -> {OUT_GT}   ({len(glob.glob(OUT_GT+'/*.nii.gz'))} files)")
json.dump([r["case"] for r in sel], open(f"{HOME}/brats2025/localeval/cases.json", "w"))
print("BUILD_COMPLETE")
