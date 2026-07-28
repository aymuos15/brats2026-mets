#!/bin/bash
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
nnUNetv2_train 11 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainerMedNeXtV2_L_k3_AdamW
