#!/bin/bash
# Build the 50/25/25 TopK/FT/pureBackSplit 3-way submission (slot 2). Verify + zip.
set -e
cd ~/brats2025
source ~/envs/brats/bin/activate
VI=val_infer
rm -rf $VI/ens_3way_502525
echo "=== [1] 3-way ensemble 0.5/0.25/0.25 ==="
python ensemble_3way.py 0.5 0.25 0.25 ens_3way_502525 | tail -4

echo "=== [2] verify gate ==="
python - <<'PY'
import glob, numpy as np, nibabel as nib
fs = sorted(glob.glob("val_infer/ens_3way_502525/*.nii.gz"))
assert len(fs) == 179, f"expected 179, got {len(fs)}"
bad = empty = 0
for f in fs:
    a = np.asanyarray(nib.load(f).dataobj)
    u = set(np.unique(a).tolist())
    if not u.issubset({0,1,2,3,4}): bad += 1; print("BAD LABELS", f, u)
    nz = int((a>0).sum())
    if nz == 0: empty += 1
    elif any((a==lab).sum() > 0.08*a.size for lab in (1,2,3,4)):
        print("DOMINANT", f)
print(f"files=179 bad_labels={bad} empty={empty}")
assert bad == 0
print("VERIFY OK")
PY

echo "=== [3] diff vs submitted 9770198 base (sanity: should differ on many, not all) ==="
mkdir -p $VI/_subzip && (cd $VI/_subzip && unzip -oq ../sub_ens_ft_5050.zip)
D=0
for f in $VI/ens_3way_502525/*.nii.gz; do
  n=$(basename "$f"); cmp -s "$f" "$VI/_subzip/$n" || D=$((D+1))
done
echo "3way-vs-9770198 differing files: $D / 179"

echo "=== [4] zip ==="
rm -f sub_3way.zip
cd $VI/ens_3way_502525 && zip -jq ~/brats2025/sub_3way.zip *.nii.gz && cd ~/brats2025
echo "zip: $(unzip -l sub_3way.zip | tail -1)"
echo "BUILD_DONE"
