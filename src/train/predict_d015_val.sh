#!/bin/bash
# D015 DilateISO fine-tune -> predict the 179 VALIDATION cases (not training self-val).
# Same step 0.25 + save_probabilities as every other member, so its probs drop into
# the same decode path. Output: val_infer/pred_d015_s025
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
CHK=${1:-checkpoint_best.pth}
IN=~/brats2025/val_infer/input_snap946
O=~/brats2025/val_infer/pred_d015_s025
echo "=== PREDICT d015 val (d=15 tr=nnUNetTrainerDiceTopK10BCE_FT150 chk=$CHK) $(date) ==="
mkdir -p "$O"
nnUNetv2_predict -i "$IN" -o "$O" \
  -d 15 -c 3d_fullres -f all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerDiceTopK10BCE_FT150 -chk "$CHK" \
  -step_size 0.25 --save_probabilities && touch "$O/.done"
echo "=== DONE d015 val: $(ls $O/*.npz 2>/dev/null | wc -l) npz  $(date) ==="
