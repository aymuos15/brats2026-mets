#!/bin/bash
# Build both D015-union variants and stage their submission zips. Does NOT submit.
set -u
source ~/brats2025/nnunet_env.sh >/dev/null 2>&1
cd ~/brats2025
for cfg in "conservative sub_d015union_cons 15" "aggressive sub_d015union_aggr 1"; do
  set -- $cfg; tag=$1; sub=$2; minsz=$3
  echo "=========== $tag  (MINSZ=$minsz) ==========="
  python assemble_d015_union.py "$sub" "$minsz" 2>&1 | tail -6
  ( cd "val_infer/$sub" && zip -q -j "$HOME/brats2025/$sub.zip" *.nii.gz )
  echo "ZIPPED $sub.zip : $(unzip -l "$HOME/brats2025/$sub.zip" | tail -1)"
  echo
done
echo BUILD_UNION_COMPLETE
