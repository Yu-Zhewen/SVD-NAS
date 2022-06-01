import torch
from torch import Tensor
import torch.nn as nn
import math
import copy
import numpy as np
from hybrid_svd.svd.svd_utils import *

def generate_low_rank_residual_wrapper(scheme, original_module, original_ofm_size, groups=-1):
    
    if scheme == 0:
        return LowRankScheme0Residual(original_module, original_ofm_size)
    elif scheme == 1:
        return LowRankScheme1Residual(original_module, original_ofm_size)
    elif scheme == 2:
        return LowRankScheme2Residual(original_module, original_ofm_size)
    elif scheme == 3:
        return LowRankScheme3Residual(original_module, original_ofm_size)
    elif scheme == 4:
        if groups == 1:
            return LowRankScheme1Residual(original_module, original_ofm_size)
        elif groups == original_module.in_channels:
            return LowRankScheme3Residual(original_module, original_ofm_size)
        else:
            return LowRankScheme4Residual(original_module, original_ofm_size, groups)
    elif scheme == 5:
        if groups == original_module.out_channels:
            return LowRankScheme0Residual(original_module, original_ofm_size)
        else:
            return LowRankScheme5Residual(original_module, original_ofm_size, groups)
    else:
        assert False

class LowRankResidual(StaticLowRankDecomposition):
    def __init__(self, low_rank_conv1, low_rank_conv2, residual_conv, original_ofm_size):
        super(LowRankResidual, self).__init__(low_rank_conv1, low_rank_conv2)

        self.residual_conv = residual_conv
        self.original_ofm_size = original_ofm_size

    def forward(self, x: Tensor) -> Tensor:
        intermediate = self.low_rank_conv1(x)
        out = self.low_rank_conv2(intermediate)

        out += self.residual_conv(x)
        return out

    def set_requires_grad(self):
        for name, param in self.named_parameters():
            if "weight" in name and "weight_mask" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

class LowRankResidualWrapper():
    def export_decomposition(self):
        #print((self.rank*self.get_per_rank_params())/self.get_original_module_params())

        low_rank_weight_array1 = self.low_rank_conv1.weight.detach().clone().numpy()
        low_rank_weight_array2 = self.low_rank_conv2.weight.detach().clone().numpy()

        u_s_sqrt, vh_s_sqrt = self.unfold_low_rank_weight(low_rank_weight_array1, low_rank_weight_array2)
        approximated_weight_array = self.fold_original_weight(u_s_sqrt @ vh_s_sqrt)

        original_weight_array = self.original_module.weight.detach().clone().numpy()
        residual_weight_array = original_weight_array - approximated_weight_array

        residual_conv = copy.deepcopy(self.original_module)
        residual_conv.bias = None
        residual_conv.weight.data.copy_(torch.from_numpy(residual_weight_array))

        decomposed_module = LowRankResidual(self.low_rank_conv1, self.low_rank_conv2, residual_conv, self.original_ofm_size)
        
        return decomposed_module

class LowRankScheme0Residual(StaticLowRankScheme0):
    def __init__(self, original_module, original_ofm_size):
        super(LowRankScheme0Residual, self).__init__(original_module, original_ofm_size)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)

class LowRankScheme1Residual(StaticLowRankScheme1):
    def __init__(self, original_module, original_ofm_size):
        super(LowRankScheme1Residual, self).__init__(original_module, original_ofm_size)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)

class LowRankScheme2Residual(StaticLowRankScheme2):
    def __init__(self, original_module, original_ofm_size):
        super(LowRankScheme2Residual, self).__init__(original_module, original_ofm_size)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)

class LowRankScheme3Residual(StaticLowRankScheme3):
    def __init__(self, original_module, original_ofm_size):
        super(LowRankScheme3Residual, self).__init__(original_module, original_ofm_size)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)

class LowRankScheme4Residual(StaticLowRankScheme4):
    def __init__(self, original_module, original_ofm_size, groups):
        super(LowRankScheme4Residual, self).__init__(original_module, original_ofm_size, groups)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)

class LowRankScheme5Residual(StaticLowRankScheme5):
    def __init__(self, original_module, original_ofm_size, groups):
        super(LowRankScheme5Residual, self).__init__(original_module, original_ofm_size, groups)

    def export_decomposition(self):
        return LowRankResidualWrapper.export_decomposition(self)