#!/bin/bash
source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
exec nnUNetv2_train 7 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer
