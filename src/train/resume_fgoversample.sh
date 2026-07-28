#!/usr/bin/env bash
set -e
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
echo "[fgoversample RESUME] start $(date)"
nnUNetv2_train 11 3d_fullres all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerDiceTopK10BCE_FgOversample --c
echo "[fgoversample RESUME] exited $(date) rc=$?"
