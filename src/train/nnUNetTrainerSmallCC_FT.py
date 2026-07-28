"""Small-lesion-targeted sampling — fine-tune. The only lever aimed at the 43% blindness.

MEASURED (2026-07-14, 80 GT cases, official metric):
  - we detect 45% of small lesions
  - of the ones we miss, 43% have a PEAK PROBABILITY BELOW 0.10 -- the network never fired.
    No decode recovers those. Measured on cases it was TRAINED on, so it is a LOWER BOUND.
  - our false positives have a median peak of 0.96 -- MORE confident than the lesions we
    find. Every confidence-based FP suppression method is therefore dead (7 refuted).

  => The gap is detection, and detection is a TRAINING problem.

WHY THE NETWORK IS BLIND: nnU-Net's sampler picks a random KEY from class_locations, then a
random VOXEL from it. Small mets live inside the (1,2,3) key among every SNFH and
large-lesion voxel in the case, so a 30-voxel met is patch-centred ~0.3% of the time --
almost no gradient, whatever the loss. FgOversample (0.33 -> 0.50) failed for exactly this
reason: it oversamples FOREGROUND, the wrong distribution. In the training set small lesions
(5,853) OUTNUMBER large ones (3,217) nearly 2:1, and they were being starved.

THE FIX (Dataset014): a new class_locations key holding ONLY small-component voxels, with
every lesion weighted EQUALLY (6 voxels each, regardless of size -- the blob-loss principle
applied to sampling). The sampler picks keys uniformly, so it wins 1 in 4. With
oversample_foreground_percent raised to 0.50, a small met is patch-centred ~12% of the time
instead of ~0.3% -- a ~40x increase.

BASE: the D007 4-REGION TopK model, deliberately NOT BackSplit. BackSplit demonstrably costs
F1 (9769617 control: .27/.33/.32 vs the 4-region net's .40/.50/.50). Training a DETECTION
model on the label scheme that suppresses detection is the mistake BlobDist made.

Stock loss (Dice + TopK-10% BCE). This experiment changes WHAT THE NETWORK SEES, nothing else.
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import (
    nnUNetTrainerDiceTopK10BCE,
)


class nnUNetTrainerSmallCC_FT(nnUNetTrainerDiceTopK10BCE):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        # explicit signature: nnU-Net introspects locals(), *args breaks it
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.initial_lr = 1e-3
        self.oversample_foreground_percent = 0.50   # was 0.33

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[SmallCC] oversample_foreground_percent = {self.oversample_foreground_percent}\n"
            f"[SmallCC] epochs={self.num_epochs} lr={self.initial_lr}\n"
            f"[SmallCC] Dataset014 carries an extra class_locations key (1,2,3,99) holding ONLY\n"
            f"[SmallCC] small-component voxels, every lesion weighted EQUALLY. The sampler picks\n"
            f"[SmallCC] keys uniformly -> a small met is patch-centred ~12% of the time, vs ~0.3%.\n"
            f"[SmallCC] Loss is untouched. This changes WHAT THE NETWORK SEES, nothing else.")
