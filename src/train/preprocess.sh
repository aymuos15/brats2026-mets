set -e
source ~/brats2025/nnunet_env.sh
echo "[preprocess] start $(date +%T)  raw=$nnUNet_raw"
nnUNetv2_plan_and_preprocess -d 7 -pl nnUNetPlannerResEncL --verify_dataset_integrity -np 12
echo "[preprocess] done $(date +%T)"
echo "PREPROCESS_COMPLETE"
