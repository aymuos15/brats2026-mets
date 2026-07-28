"""ONLY-RC specialist trainer with foreground-prior bias init (RetinaNet/focal trick).
RC is ~0.0265% of voxels -> a from-scratch sigmoid head starting at 0.5 gets swamped by the
'push to background' gradient and can collapse to all-background. Seed the output-head bias to
logit(p_RC) so the net starts at the true base rate and learns LOCATION, not rarity. Free; training
adjusts the bias afterward."""
import math, torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceTopK10BCE import nnUNetTrainerDiceTopK10BCE

P_RC = 0.000265                       # measured RC voxel fraction
_BIAS = math.log(P_RC / (1 - P_RC))   # ~ -8.24

class nnUNetTrainerDiceTopK10BCE_RCprior(nnUNetTrainerDiceTopK10BCE):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                    num_input_channels, num_output_channels, enable_deep_supervision=True):
        net = nnUNetTrainerDiceTopK10BCE.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision)
        n = 0
        with torch.no_grad():
            seg_layers = getattr(getattr(net, 'decoder', None), 'seg_layers', None)
            if seg_layers is not None:
                for seg in seg_layers:
                    if getattr(seg, 'bias', None) is not None:
                        seg.bias.fill_(_BIAS); n += 1
        print(f'[RCprior] foreground-prior bias = {_BIAS:.3f} (p={P_RC}) set on {n} seg head(s)', flush=True)
        return net
