set -e
cd ~/brats2025
echo "[extract] start $(date +%T)"
mkdir -p extracted
for z in raw_zips/*.zip; do echo "[extract] $z"; unzip -o -q "$z" -d extracted/; done
echo "[extract] done $(date +%T)"
echo "[build] start $(date +%T)"
~/envs/brats/bin/python prepare_007.py
echo "[build] done $(date +%T)"
echo "BUILD_ALL_COMPLETE"
