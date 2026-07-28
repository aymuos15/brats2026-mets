#!/bin/bash
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
cd ~/brats2025
IN=val_infer/input_snap946
O=val_infer/pred_ccloss_s0125
mkdir -p "$O"
S=$(date +%s)
echo "=== CCLOSS step0.125 val predict START $(date) ==="
nnUNetv2_predict -i "$IN" -o "$O" \
  -d 11 -c 3d_fullres -f all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerCCLoss_BackSplit_FT -chk checkpoint_best.pth \
  -step_size 0.125 --save_probabilities && touch "$O/.done"
E=$(date +%s)
echo "=== CCLOSS step0.125 DONE npz=$(ls $O/*.npz 2>/dev/null | wc -l) elapsed=$((E-S))s $(date) ==="
echo "CCLOSS_S0125_COMPLETE"
