"""
Hint-BackSplit trainer — HintU's load-bearing mechanism ported into nnU-Net v2.

WHAT: prepends hand-crafted salience channels (multi-scale 3D Laplacian-of-Gaussian
of T1c, sigmas ~1,2,4mm) to the input, so small enhancing lesions survive early
downsampling. LoG (not a raw Laplacian) is the operator brain-mets detection frameworks
actually use for small-lesion candidate selection (arXiv:1908.04701, arXiv:2105.13406):
the Gaussian pre-smooth kills voxel noise, and the scales catch ~2-15mm lesions. This is
the part of HintU (arXiv:2406.13445) whose own ablation (HintO) shows it carries the gain;
we DROP the full-resolution cross-attention (O(voxels^2) -> infeasible in 3D, un-ablated,
would OOM the GB10).

WHY PAIRED WITH BACKSPLIT (Dataset011, 6-region, TopK backbone = the 9769617 config):
BackSplit is our DSC/NSD-strong but recall-weak ensemble parent (F1: WT -0.134,
TC -0.170, ET -0.176). The 50/50 ensemble (9769908, rank 14) only concedes recall,
so lifting BackSplit's recall lifts the ensemble. The Laplacian's bright-blob prior
fits ET (enhances on T1c, channel 0) — exactly BackSplit's worst F1 drop.

Backbone: subclasses nnUNetTrainerDiceTopK10BCE (region-compatible soft-Dice + TopK-BCE),
run on Dataset011_BraTSMetsBackSplit so the 6-region supervision is active.

Input channels: T1c(0) T1n(1) T2f(2) T2w(3)  ->  + LoG(T1c) at each sigma (channels 4..6).

INFERENCE TODO (before submitting): nnUNetv2_predict feeds the network 4-channel
preprocessed data, so the hint must be re-applied at predict time. Two options:
  (a) a thin predictor wrapper that appends the same Laplacian channel, or
  (b) bake the Laplacian into a preprocessing step / extra raw channel.
Training-side injection is implemented here; the predict wrapper is a launch-time task.
"""
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import (
    nnUNetTrainerDiceTopK10BCE,
)

# hint operators live in a standalone module so they import without nnU-Net (tests) and
# so the tested code is exactly the shipped code.
try:
    from hint_ops import multiscale_log as _multiscale_log, DEFAULT_SIGMAS
except ImportError:  # when deployed into site-packages next to the trainer
    from .hint_ops import multiscale_log as _multiscale_log, DEFAULT_SIGMAS


SIGMAS = DEFAULT_SIGMAS          # LoG scales in voxels (~1,2,4 mm)
EXTRA_IN = len(DEFAULT_SIGMAS)   # one hint channel per scale
HINT_SRC_CHANNEL = 0             # T1c


def add_hint(data: torch.Tensor) -> torch.Tensor:
    """Append the multi-scale LoG hint channels to a (B, C, ...) input.

    Shared by train/val (this trainer) and inference (predict_hint.HintPredictor) so
    training and prediction see byte-identical hint construction.
    """
    with torch.no_grad():
        src = data[:, HINT_SRC_CHANNEL:HINT_SRC_CHANNEL + 1]
        hint = _multiscale_log(src.float(), SIGMAS).to(data.dtype)
    return torch.cat([data, hint], dim=1)


class nnUNetTrainerDiceTopK10BCE_Hint(nnUNetTrainerDiceTopK10BCE):
    """BackSplit + TopK backbone, with multi-scale LoG(T1c) hint channels prepended.

    Design: `num_input_channels` stays at 4 (matches the 4-channel preprocessed data on
    disk / the dataloader), but the *network* is built for 4 + EXTRA_IN via the
    build_network_architecture override below. train_step/validation_step append the
    hint so the tensor entering the net is 7-channel. Because the predictor also calls
    this class's build_network_architecture, inference rebuilds the identical 7-channel
    net (see predict_hint.py for the matching input-side injection).
    """

    SIGMAS = SIGMAS
    EXTRA_IN = EXTRA_IN
    HINT_SRC_CHANNEL = HINT_SRC_CHANNEL

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        # +EXTRA_IN so the first conv expects the appended hint channels. Called by both
        # this trainer's initialize() AND the predictor -> train/predict nets match.
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels + EXTRA_IN, num_output_channels, enable_deep_supervision)

    def train_step(self, batch: dict) -> dict:
        batch = {**batch, "data": add_hint(batch["data"])}
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        batch = {**batch, "data": add_hint(batch["data"])}
        return super().validation_step(batch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        # nnU-Net's end-of-training validation spins up a plain nnUNetPredictor that would
        # feed 4-channel data to our 7-channel net -> crash. We predict externally for
        # submission (predict_hint.py) and pick checkpoints by EMA pseudo-dice, so this
        # internal pass is not needed. Skip it cleanly.
        self.print_to_log_file(
            "[Hint] skipping built-in perform_actual_validation "
            "(use predict_hint.py for the 7-channel-aware val prediction).")
