set -e
. .venv/bin/activate
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"-\")"
pip install --quiet nnunetv2
python -c "import nnunetv2,nibabel,SimpleITK;print(\"nnunetv2 OK\")"
echo ENV_SETUP_COMPLETE
