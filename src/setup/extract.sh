set -e
cd ~/brats2025
mkdir -p extracted
for z in raw_zips/*.zip; do echo "unzip $z"; unzip -o -q "$z" -d extracted/; done
echo "EXTRACT_COMPLETE"
ls extracted/
