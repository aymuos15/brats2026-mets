#!/usr/bin/env bash
set -e
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
echo "[fgoversample] start $(date) | Dataset011 BackSplit + oversample_fg=0.5"
nnUNetv2_train 11 3d_fullres all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerDiceTopK10BCE_FgOversample
echo "[fgoversample] exited $(date) rc=$?"
