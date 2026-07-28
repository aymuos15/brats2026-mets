"""Fine-tune variant of the region DiceTopK10BCE trainer: 150 ep, lr 1e-3 (warm-start)."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import nnUNetTrainerDiceTopK10BCE

class nnUNetTrainerDiceTopK10BCE_FT150(nnUNetTrainerDiceTopK10BCE):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.initial_lr = 1e-3
