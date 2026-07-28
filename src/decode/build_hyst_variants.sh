#!/bin/bash
set -u
source ~/brats2025/nnunet_env.sh >/dev/null 2>&1
cd ~/brats2025
for cfg in "lo030hi065 0.30 0.65" "lo030hi075 0.30 0.75"; do
  set -- $cfg; name=$1; lo=$2; hi=$3
  echo "=== assemble $name (lo=$lo hi=$hi) ==="
  python assemble_3way.py sub_3way_$name $lo $hi 3 2>&1 | tail -3
  ( cd val_infer/sub_3way_$name && zip -q -j ~/brats2025/sub_3way_$name.zip *.nii.gz )
  echo "ZIPPED sub_3way_$name.zip : $(unzip -l ~/brats2025/sub_3way_$name.zip | tail -1)"
done
echo BUILD_ZIP_COMPLETE
