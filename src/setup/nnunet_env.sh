# source this to use the brats nnU-Net env on sie271-pc
export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="$HOME/projects/CAI4CAI/SegData/raw"
export nnUNet_preprocessed="$HOME/projects/CAI4CAI/SegData/preprocessed"
export nnUNet_results="$HOME/projects/CAI4CAI/models"
source "$HOME/envs/brats/bin/activate"
