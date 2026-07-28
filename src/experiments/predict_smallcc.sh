#!/bin/bash
# Predict the 80-case tumour eval set with the SmallCC fine-tune (Dataset014),
# SAME step size (0.25) + save_probabilities as predict_localeval.sh, so its probs
# drop straight into miss_diag_smallcc.py alongside the existing 3 members.
#
#   smallcc = D014 SmallCC_FT  ckpt <arg1, default checkpoint_best.pth> -> pred_smallcc
#
# Usage: bash predict_smallcc.sh [checkpoint_best.pth|checkpoint_final.pth]
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
CHK=${1:-checkpoint_best.pth}
IN=~/brats2025/localeval/images
O=~/brats2025/localeval/pred_smallcc

if [ -f "$O/.done" ]; then echo "[skip] pred_smallcc already done"; exit 0; fi
echo "=== PREDICT smallcc (d=14 tr=nnUNetTrainerSmallCC_FT chk=$CHK) $(date) ==="
mkdir -p "$O"
nnUNetv2_predict -i "$IN" -o "$O" \
  -d 14 -c 3d_fullres -f all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerSmallCC_FT -chk "$CHK" \
  -step_size 0.25 --save_probabilities || { echo "PREDICT_FAILED smallcc"; exit 1; }
touch "$O/.done"
echo "=== DONE smallcc: $(ls $O/*.npz 2>/dev/null | wc -l) npz  $(date) ==="
