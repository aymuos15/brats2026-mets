#!/bin/bash
# Predict the 130-case local eval set with the SAME three models + SAME checkpoints
# + SAME step size (0.25) that produced our scored val submissions, so that decode
# rankings measured here transfer to the leaderboard.
#
#   topk       = D007 4-region  ckpt topk_snap      -> pred_topk_s025      (parent of 9770519/9770654)
#   completion = D011 BackSplit ckpt snap_ep946     -> pred_topk_completion_s025 (parent of 9770646/9770654)
#   ccloss     = D011 CCLoss FT ckpt best           -> pred_ccloss_s025    (parent of 9770749/9771002 RC)
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
IN=~/brats2025/localeval/images
O=~/brats2025/localeval

run () {  # name dataset trainer ckpt
  local name=$1 d=$2 tr=$3 chk=$4
  if [ -f "$O/pred_$name/.done" ]; then echo "[skip] $name already done"; return; fi
  echo "=== PREDICT $name (d=$d tr=$tr chk=$chk) $(date) ==="
  mkdir -p "$O/pred_$name"
  nnUNetv2_predict -i "$IN" -o "$O/pred_$name" \
    -d "$d" -c 3d_fullres -f all -p nnUNetResEncUNetLPlans -tr "$tr" -chk "$chk" \
    -step_size 0.25 --save_probabilities || { echo "PREDICT_FAILED $name"; exit 1; }
  touch "$O/pred_$name/.done"
  echo "=== DONE $name: $(ls $O/pred_$name/*.npz 2>/dev/null | wc -l) npz  $(date) ==="
}

run topk       7  nnUNetTrainerDiceTopK10BCE      checkpoint_topk_snap.pth
run completion 11 nnUNetTrainerDiceTopK10BCE      checkpoint_snap_ep946.pth
run ccloss     11 nnUNetTrainerCCLoss_BackSplit_FT checkpoint_best.pth

echo "ALL_PREDICTS_COMPLETE $(date)"
