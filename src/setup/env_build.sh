set -e
export PATH="$HOME/.local/bin:$PATH"
ENV=$HOME/envs/brats
echo "[1/4] create venv"
uv venv "$ENV" --python 3.12
echo "[2/4] torch cu130 (no-deps)"
uv pip install --python "$ENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu130 --no-deps
echo "[3/4] nvidia cu13 runtime deps (regular PyPI; pypi.nvidia.com blocked)"
uv pip install --python "$ENV/bin/python" nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 numpy
echo "[4/4] editable install of CAI4CAI/nnUNet fork (nnunetv2 2.6.4)"
uv pip install --python "$ENV/bin/python" -e "$HOME/projects/CAI4CAI/nnUNet"
echo "--- verify ---"
"$ENV/bin/python" - <<PY
import torch, nnunetv2, nibabel, SimpleITK
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("nnunetv2", nnunetv2.__version__)
PY
"$ENV/bin/nnUNetv2_plan_and_preprocess" -h >/dev/null 2>&1 && echo "nnUNetv2 CLI OK"
echo ENV_BUILD_COMPLETE
