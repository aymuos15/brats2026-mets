#!/bin/bash
# Predict 3 more members on the 80-case local eval set, so the harness can score them
# against real GT with the official metric BEFORE we spend a submission slot.
#
#   ccloss_v2 : the FP-penalty fine-tune. Does fixing the loss beat fixing the decode?
#               (9771380 = blob loss + hysteresis = F1 rank 3/1422, sunk by its FPs)
#   ft        : Focal-Tversky. Recall lever, trained weeks ago, NEVER used in a fusion.
#   fgover    : FgOversample. Same -- trained, never fused.
#
# The 3-way (topk+completion+ccloss) works because averaging dilutes idiosyncratic FPs while
# corroborated TPs survive. That argument gets STRONGER with more members. These give us a
# 5- or 6-way to test, and the harness ranks them for free.
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
IN=~/brats2025/localeval/images
O=~/brats2025/localeval

run () {  # name trainer ckpt
  local name=$1 tr=$2 chk=$3
  if [ -f "$O/pred_$name/.done" ]; then echo "[skip] $name"; return; fi
  echo "=== PREDICT $name (tr=$tr chk=$chk)  $(date +%H:%M) ==="
  mkdir -p "$O/pred_$name"
  nnUNetv2_predict -i "$IN" -o "$O/pred_$name" \
    -d 11 -c 3d_fullres -f all -p nnUNetResEncUNetLPlans -tr "$tr" -chk "$chk" \
    -step_size 0.25 --save_probabilities || { echo "PREDICT_FAILED $name"; exit 1; }
  touch "$O/pred_$name/.done"
  echo "=== DONE $name: $(ls $O/pred_$name/*.npz | wc -l) npz  $(date +%H:%M) ==="
}

run ccloss_v2 nnUNetTrainerCCLossV2_FT             checkpoint_final.pth
run ft        nnUNetTrainerFocalTversky_BackSplit  checkpoint_best.pth
run fgover    nnUNetTrainerDiceTopK10BCE_FgOversample checkpoint_best.pth

echo "ALL_MEMBER_PREDICTS_COMPLETE $(date +%H:%M)"
