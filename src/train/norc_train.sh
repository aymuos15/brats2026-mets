source ~/brats2025/nnunet_env.sh
export nnUNet_compile=0
nnUNetv2_train 12 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceTopK10BCE
