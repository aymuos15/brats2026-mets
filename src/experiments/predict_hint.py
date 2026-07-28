"""Inference wrapper for the Hint-BackSplit model.

WHY THIS EXISTS: the trained net has a 7-channel first conv (4 modalities + 3 LoG hint
scales). nnUNetv2_predict feeds it 4-channel preprocessed data, so we must re-apply the
SAME multi-scale LoG hint at predict time. The network side is already handled — the
predictor calls nnUNetTrainerDiceTopK10BCE_Hint.build_network_architecture, which builds
a 7-channel net. This wrapper only injects the hint channels into each patch before the
forward pass, via the exact same add_hint() used in training.

USAGE (on a Spark, env ~/envs/brats, after deploying hint_ops.py + the trainer into
the env's variants/loss/ so recursive_find_python_class can import the trainer):

    source ~/brats2025/nnunet_env.sh
    export nnUNet_compile=0
    python predict_hint.py \
      -i <staged_4ch_input_dir> -o <out_dir> \
      -d 11 -c 3d_fullres -p nnUNetResEncUNetLPlans \
      -tr nnUNetTrainerDiceTopK10BCE_Hint -f all -chk checkpoint_best.pth \
      --step 0.375

Then postprocess + zip + submit exactly as always:
    ~/brats2025/postprocess.py --rc-largest 1  ->  zip -j  ->  syn.store/submit
(ov 0.375 + RC-fix are locked; the hint changes only the model input, not the recipe.)
"""
import argparse

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

try:
    from nnUNetTrainerDiceTopK10BCE_Hint import add_hint
except ImportError:
    from .nnUNetTrainerDiceTopK10BCE_Hint import add_hint


class HintPredictor(nnUNetPredictor):
    """nnUNetPredictor that appends the multi-scale LoG hint to every patch."""

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, *patch) preprocessed modalities. Append hint -> (B, 7, *patch).
        # Done BEFORE mirroring so the (spatial) flips apply consistently to the image
        # and its hint channels; super() handles the network call + TTA averaging.
        return super()._internal_maybe_mirror_and_predict(add_hint(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("-d", required=True)
    ap.add_argument("-c", default="3d_fullres")
    ap.add_argument("-p", default="nnUNetResEncUNetLPlans")
    ap.add_argument("-tr", default="nnUNetTrainerDiceTopK10BCE_Hint")
    ap.add_argument("-f", default="all")
    ap.add_argument("-chk", default="checkpoint_best.pth")
    ap.add_argument("--step", type=float, default=0.375)  # locked overlap
    ap.add_argument("--disable_tta", action="store_true")
    a = ap.parse_args()

    predictor = HintPredictor(
        tile_step_size=a.step,
        use_mirroring=not a.disable_tta,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        _model_folder(a), use_folds=(a.f,), checkpoint_name=a.chk)
    predictor.predict_from_files(a.i, a.o, save_probabilities=False,
                                 overwrite=True, num_processes_preprocessing=2,
                                 num_processes_segmentation_export=2)


def _model_folder(a):
    import os
    base = os.environ["nnUNet_results"]
    ds = f"Dataset{int(a.d):03d}_BraTSMetsBackSplit"
    return os.path.join(base, ds, f"{a.tr}__{a.p}__{a.c}")


if __name__ == "__main__":
    main()
