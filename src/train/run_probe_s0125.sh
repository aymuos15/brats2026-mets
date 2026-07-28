#!/bin/bash
set -u
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
cd ~/brats2025
S=$(date +%s)
nnUNetv2_predict -i val_infer/probe_in -o val_infer/probe_out_s0125 \
  -d 11 -c 3d_fullres -f all -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerCCLoss_BackSplit_FT -chk checkpoint_best.pth \
  -step_size 0.125 --save_probabilities
E=$(date +%s)
echo "PROBE_ELAPSED $((E-S)) sec for 3 cases at step 0.125"
echo "PROBE_DONE"
