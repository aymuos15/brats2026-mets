set -e
export PATH="$HOME/.local/bin:$PATH"
ENV=$HOME/envs/brats
uv pip install --python "$ENV/bin/python" "nnunetv2==2.6.4" nvitop
"$ENV/bin/python" -c "import torch,nnunetv2;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"| nnunetv2\",nnunetv2.__version__)"
"$ENV/bin/nnUNetv2_plan_and_preprocess" -h >/dev/null 2>&1 && echo "CLI OK"
echo ENV_FIX_COMPLETE
