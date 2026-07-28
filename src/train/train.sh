source ~/brats2025/nnunet_env.sh
export nnUNet_compile=false
echo "[train] start $(date) | Dataset007 3d_fullres all | ResEnc-L patch96 batch4 | region-based Dice+BCE (default trainer)"
nnUNetv2_train 7 3d_fullres all -p nnUNetResEncUNetLPlans
echo "[train] exited $(date) rc=$?"
