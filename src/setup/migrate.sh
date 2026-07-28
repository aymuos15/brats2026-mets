set -e
echo "[mig] start $(date +%T)"
ssh sie272-pc "mkdir -p ~/projects/CAI4CAI/SegData/preprocessed ~/projects/CAI4CAI/models ~/envs ~/brats2025"
echo "[mig] rsync env (~5G)..."
rsync -a --info=progress2 ~/envs/brats/ sie272-pc:~/envs/brats/ 2>&1 | tail -1
echo "[mig] rsync scripts..."
rsync -a ~/brats2025/nnunet_env.sh ~/brats2025/train.sh sie272-pc:~/brats2025/
echo "[mig] rsync preprocessed 3d_fullres (~75G, excl 2d)..."
rsync -a --info=progress2 --exclude nnUNetPlans_2d \
  ~/projects/CAI4CAI/SegData/preprocessed/Dataset007_BraTSMets2025/ \
  sie272-pc:~/projects/CAI4CAI/SegData/preprocessed/Dataset007_BraTSMets2025/ 2>&1 | tail -1
echo "[mig] DONE $(date +%T)"
echo "MIGRATE_COMPLETE"
