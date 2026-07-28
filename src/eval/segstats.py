"""Aggregate label stats for an assembled submission dir (light: masks only)."""
import sys, os, glob
import numpy as np
import nibabel as nib

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.nii.gz")))
tot = {1: 0, 2: 0, 3: 0, 4: 0}
ncomp_tc = 0
from scipy.ndimage import label as cc_label
for f in files:
    a = np.asarray(nib.load(f).dataobj).astype(np.uint8)
    for l in tot:
        tot[l] += int((a == l).sum())
    ncomp_tc += cc_label(np.isin(a, [1, 3]))[1]
print(f"{os.path.basename(d):>22}  n={len(files)}")
print(f"   voxels  NETC(1)={tot[1]:>9}  ED(2)={tot[2]:>9}  ET(3)={tot[3]:>9}  RC(4)={tot[4]:>9}")
print(f"   total tumour (1+2+3) = {tot[1]+tot[2]+tot[3]:>10}")
print(f"   TC components across all cases = {ncomp_tc}")
