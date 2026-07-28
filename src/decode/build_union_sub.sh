#!/bin/bash
# Build the union recall-recovery submission (slot 1) on sie272.
# Steps: (0) SAFETY: reconstruct the base = RC-fix(ens_ft_5050), diff vs the submitted
#        9770198 zip so we KNOW the union sits on the real submission. (1) union-recover
#        ET/TC whole CCs from TopK. (2) RC-fix. (3) verify gate. (4) zip -j.
set -e
cd ~/brats2025
source ~/envs/brats/bin/activate
VI=val_infer

echo "=== [0] SAFETY: reconstruct base (RC-fix of ens_ft_5050, no union) and diff vs submitted zip ==="
rm -rf $VI/_basecheck $VI/_subzip $VI/ens_ft_5050_union $VI/ens_ft_5050_union_rcfix
python postprocess.py -i $VI/ens_ft_5050 -o $VI/_basecheck --rc-largest 1
mkdir -p $VI/_subzip && (cd $VI/_subzip && unzip -oq ../sub_ens_ft_5050.zip)
DIFFS=0
for f in $VI/_basecheck/*.nii.gz; do
  n=$(basename "$f")
  if ! cmp -s "$f" "$VI/_subzip/$n"; then DIFFS=$((DIFFS+1)); fi
done
echo "base-vs-submitted differing files: $DIFFS / $(ls $VI/_basecheck/*.nii.gz | wc -l)"
if [ "$DIFFS" -ne 0 ]; then
  echo "WARN: base reconstruction != submitted zip. ens_ft_5050 may already be RC-fixed or differ; inspect before trusting."
fi

echo "=== [1] union-recover (all 132, ET/TC only) ==="
python union_recover.py --base $VI/ens_ft_5050 --topk $VI/pred_topk_best --out $VI/ens_ft_5050_union | tail -8

echo "=== [2] RC-fix ==="
python postprocess.py -i $VI/ens_ft_5050_union -o $VI/ens_ft_5050_union_rcfix --rc-largest 1

echo "=== [2b] how many final files differ from the submitted base (should ~= cases changed) ==="
CHANGED=0
for f in $VI/ens_ft_5050_union_rcfix/*.nii.gz; do
  n=$(basename "$f")
  if ! cmp -s "$f" "$VI/_subzip/$n"; then CHANGED=$((CHANGED+1)); fi
done
echo "final-vs-submitted differing files: $CHANGED"

echo "=== [3] verify gate ==="
python - <<'PY'
import glob, numpy as np, nibabel as nib
fs = sorted(glob.glob("val_infer/ens_ft_5050_union_rcfix/*.nii.gz"))
assert len(fs) == 179, f"expected 179, got {len(fs)}"
bad = empty = 0
for f in fs:
    a = np.asanyarray(nib.load(f).dataobj)
    u = set(np.unique(a).tolist())
    if not u.issubset({0,1,2,3,4}): bad += 1; print("BAD LABELS", f, u)
    nz = int((a>0).sum())
    if nz == 0: empty += 1
    if nz:
        for lab in (1,2,3,4):
            if (a==lab).sum() > 0.08*a.size: print("DOMINANT", f, lab)
print(f"files=179 bad_labels={bad} empty={empty}")
assert bad == 0, "bad labels present"
print("VERIFY OK")
PY

echo "=== [4] zip ==="
rm -f sub_union.zip
cd $VI/ens_ft_5050_union_rcfix && zip -jq ~/brats2025/sub_union.zip *.nii.gz && cd ~/brats2025
echo "zip files: $(unzip -l sub_union.zip | tail -1)"
echo "BUILD_DONE"
