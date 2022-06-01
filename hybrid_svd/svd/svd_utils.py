import numpy as np
import copy
import torch
import torch.nn as nn
from torch import Tensor
import sys
import math

from hybrid_svd.common.utils import *


def generate_low_rank_wrapper(scheme, original_module, original_ofm_size, groups=-1):
    
    if scheme == 0:
        return StaticLowRankScheme0(original_module, original_ofm_size)
    elif scheme == 1:
        return StaticLowRankScheme1(original_module, original_ofm_size)
    elif scheme == 2:
        return StaticLowRankScheme2(original_module, original_ofm_size)
    elif scheme == 3:
        return StaticLowRankScheme3(original_module, original_ofm_size)
    elif scheme == 4:
        if groups == 1:
            return StaticLowRankScheme1(original_module, original_ofm_size)
        elif groups == original_module.in_channels:
            return StaticLowRankScheme3(original_module, original_ofm_size)
        else:
            return StaticLowRankScheme4(original_module, original_ofm_size, groups)
    elif scheme == 5:
        if groups == original_module.out_channels:
            return StaticLowRankScheme0(original_module, original_ofm_size)
        else:
            return StaticLowRankScheme5(original_module, original_ofm_size, groups)
    else:
        assert False

def enumerate_scheme_candidates(original_module, original_ofm_size):
    in_channels = original_module.in_channels
    out_channels = original_module.out_channels

    candidates = []

    candidates.append(generate_low_rank_wrapper(2, original_module, original_ofm_size))

    for group in get_factors(in_channels):
        candidates.append(generate_low_rank_wrapper(4, original_module, original_ofm_size, group))

    for group in get_factors(out_channels):
        candidates.append(generate_low_rank_wrapper(5, original_module, original_ofm_size, group))

    return candidates

class StaticLowRankDecomposition(nn.Module):
    def __init__(self, low_rank_conv1, low_rank_conv2):
        super(StaticLowRankDecomposition, self).__init__()
        self.low_rank_conv1 = low_rank_conv1
        self.low_rank_conv2 = low_rank_conv2

    def forward(self, x: Tensor) -> Tensor:
        intermediate = self.low_rank_conv1(x)
        out = self.low_rank_conv2(intermediate)
        return out

    def set_requires_grad(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

class StaticLowRankDecompositionWrapper():
    def __init__(self, original_module, original_ofm_size):
        self.original_module = original_module
        self.original_ofm_size = original_ofm_size

        assert self.original_module.groups == 1

    def get_original_module_macs(self):
        macs = self.original_ofm_size[0] \
               * self.original_ofm_size[1] \
               * self.original_module.out_channels \
               * self.original_module.in_channels \
               * self.original_module.kernel_size[0] \
               * self.original_module.kernel_size[1]

        return macs

    def get_original_module_params(self):
        params = self.original_module.out_channels \
                 * self.original_module.in_channels \
                 * self.original_module.kernel_size[0] \
                 * self.original_module.kernel_size[1]

        return params

    def decompose_weight(self, unfolded_weight, rank_slice=None):
        if rank_slice == None:
            rank_slice = list(range(self.rank))

        assert len(rank_slice) > 0

        [u, s, vh] = np.linalg.svd(unfolded_weight, full_matrices=False)

        u_lowrank = u[ : , : , rank_slice]
        s_lowrank = s[ : , rank_slice]
        vh_lowrank = vh[ : , rank_slice, : ]

        u_s_sqrt = np.zeros_like(u_lowrank)
        vh_s_sqrt = np.zeros_like(vh_lowrank)

        for i in range(self.groups):
            s_sqrt_diag = np.diag(np.sqrt(s_lowrank[i]))
            u_s_sqrt[i] = u_lowrank[i] @ s_sqrt_diag
            vh_s_sqrt[i] = s_sqrt_diag @ vh_lowrank[i]

        return u_s_sqrt, vh_s_sqrt
    
    def generate_low_rank_weight(self):
        rank_slice = list(range(self.rank))

        original_weight = self.original_module.weight.detach().clone().numpy()
        unfolded_weight = self.unfold_original_weight(original_weight)
        u_s_sqrt, vh_s_sqrt = self.decompose_weight(unfolded_weight, rank_slice)
        low_rank_weight_array1, low_rank_weight_array2 = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt)
            
        self.low_rank_conv1.weight.data.copy_(torch.from_numpy(low_rank_weight_array1))
        self.low_rank_conv2.weight.data.copy_(torch.from_numpy(low_rank_weight_array2))
        if self.original_module.bias != None:
            self.low_rank_conv2.bias.data.copy_(self.original_module.bias.detach().clone())

    def export_decomposition(self):
        decomposed_module = StaticLowRankDecomposition(self.low_rank_conv1, self.low_rank_conv2)
        return decomposed_module
                                               
class StaticLowRankScheme2(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme2, self).__init__(original_module, original_ofm_size)
        self.groups = 1
        self.scheme = 2

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels * self.original_module.kernel_size[0], 
                        self.original_module.in_channels * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * (self.original_module.out_channels * self.original_module.kernel_size[0] 
                            + self.original_module.in_channels * self.original_module.kernel_size[1] * self.original_module.stride[0])

        return per_rank_macs

    def get_per_rank_params(self):
        per_rank_params = self.original_module.out_channels \
                            * self.original_module.kernel_size[0] \
                            + self.original_module.in_channels * self.original_module.kernel_size[1]
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.transpose(original_weight_array, (0, 2, 1, 3))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (1,
                                                                   self.original_module.out_channels * self.original_module.kernel_size[0],
                                                                   self.original_module.in_channels * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.original_module.out_channels,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.in_channels,
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (0, 2, 1, 3))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank
        assert rank > 0

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.rank, 
                                        kernel_size=(1, self.original_module.kernel_size[1]), 
                                        stride=(1, self.original_module.stride[1]), 
                                        padding=(0, self.original_module.padding[1]), 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=(self.original_module.kernel_size[0], 1), 
                                        stride=(self.original_module.stride[0], 1), 
                                        padding=(self.original_module.padding[0], 0), 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        low_rank_weight_array1 = np.reshape(vh_s_sqrt, (rank, 
                                                        self.original_module.in_channels, 
                                                        1, 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = np.reshape(u_s_sqrt, (self.original_module.out_channels,
                                                       self.original_module.kernel_size[0], 
                                                       rank, 
                                                       1))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (0, 2 ,1, 3))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        vh_s_sqrt = np.reshape(low_rank_weight_array1, (1,
                                                        rank,
                                                        self.original_module.in_channels * self.original_module.kernel_size[1]))
        u_s_sqrt = np.transpose(low_rank_weight_array2, (0, 2 ,1, 3))
        u_s_sqrt = np.reshape(u_s_sqrt, (1,
                                         self.original_module.out_channels * self.original_module.kernel_size[0],
                                         rank))
        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme4(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size, groups):
        super(StaticLowRankScheme4, self).__init__(original_module, original_ofm_size)
        self.groups = groups
        self.scheme = 4

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels, 
                        int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * self.groups \
                        * (self.original_module.out_channels 
                            + int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.groups \
                            * (self.original_module.out_channels 
                                + int(self.original_module.in_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.reshape(original_weight_array, (self.original_module.out_channels,
                                                                   self.groups,
                                                                   int(self.original_module.in_channels / self.groups),
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = np.transpose(unfolded_weight_array, (1, 0, 2, 3, 4))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (self.groups,
                                                                   self.original_module.out_channels,
                                                                   int(self.original_module.in_channels / self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.groups,
                                                                       self.original_module.out_channels,
                                                                       int(self.original_module.in_channels / self.groups),
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (1, 0, 2, 3, 4))
        approximated_weight_array = np.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        assert rank > 0
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.groups * self.rank, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.groups, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.groups * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        low_rank_weight_array1 = np.reshape(vh_s_sqrt, (self.groups * rank, 
                                                        int(self.original_module.in_channels/self.groups), 
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = np.reshape(u_s_sqrt, (self.groups,
                                                       self.original_module.out_channels,
                                                       rank,
                                                       1, 
                                                       1))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (1, 0, 2, 3, 4))
        low_rank_weight_array2 = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                                     self.groups * rank,
                                                                     1,
                                                                     1))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0

        vh_s_sqrt = np.reshape(low_rank_weight_array1, (self.groups,
                                                        rank, 
                                                        int(self.original_module.in_channels/self.groups)*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))
        u_s_sqrt = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                       self.groups,
                                                       rank,
                                                       1,
                                                       1))
        u_s_sqrt = np.transpose(u_s_sqrt, (1, 0, 2, 3, 4))
        u_s_sqrt = np.reshape(u_s_sqrt, (self.groups,
                                         self.original_module.out_channels,
                                         rank))       

        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme5(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size, groups):
        super(StaticLowRankScheme5, self).__init__(original_module, original_ofm_size)
        self.groups = groups
        self.scheme = 5

    def get_max_rank(self):
        max_rank = min(self.original_module.in_channels, 
                        int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])                          
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * self.groups \
                        * (self.original_module.in_channels * self.original_module.stride[0] * self.original_module.stride[1]
                            + int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.groups \
                            * (self.original_module.in_channels 
                                + int(self.original_module.out_channels/self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.reshape(original_weight_array, (self.groups,
                                                                   int(self.original_module.out_channels / self.groups), 
                                                                   self.original_module.in_channels,
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = np.transpose(unfolded_weight_array, (0, 2, 1, 3, 4))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (self.groups,
                                                                   self.original_module.in_channels,
                                                                   int(self.original_module.out_channels / self.groups) * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.groups,
                                                                       self.original_module.in_channels,
                                                                       int(self.original_module.out_channels / self.groups),
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (0, 2, 1, 3, 4))
        approximated_weight_array = np.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))
                    
        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        assert rank > 0
        self.rank = rank
        
        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.groups * self.rank, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.groups * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.groups, 
                                        bias=(self.original_module.bias!=None), dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        low_rank_weight_array1 = np.reshape(u_s_sqrt, (self.groups, 
                                                       self.original_module.in_channels, 
                                                       rank, 
                                                       1, 
                                                       1))
        low_rank_weight_array1 = np.transpose(low_rank_weight_array1, (0, 2, 1, 3, 4))
        low_rank_weight_array1 = np.reshape(low_rank_weight_array1, (self.groups*rank, 
                                                                     self.original_module.in_channels, 
                                                                     1, 
                                                                     1))

        low_rank_weight_array2 = np.reshape(vh_s_sqrt, (self.groups, 
                                                        rank,
                                                        int(self.original_module.out_channels/self.groups),
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (0, 2, 1, 3, 4))
        low_rank_weight_array2 = np.reshape(low_rank_weight_array2, (self.original_module.out_channels, 
                                                                     rank,
                                                                     self.original_module.kernel_size[0], 
                                                                     self.original_module.kernel_size[1]))

        return low_rank_weight_array1, low_rank_weight_array2  

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        assert rank > 0
        u_s_sqrt = np.reshape(low_rank_weight_array1,(self.groups,
                                                     rank,
                                                     self.original_module.in_channels,
                                                     1,
                                                     1))
        u_s_sqrt = np.transpose(u_s_sqrt, (0, 2, 1, 3, 4))
        u_s_sqrt = np.reshape(u_s_sqrt, (self.groups, 
                                         self.original_module.in_channels, 
                                         rank))

        vh_s_sqrt = np.reshape(low_rank_weight_array2, (self.groups,
                                                        int(self.original_module.out_channels/self.groups), 
                                                        rank, 
                                                        self.original_module.kernel_size[0],
                                                        self.original_module.kernel_size[1]))  
        vh_s_sqrt = np.transpose(vh_s_sqrt, (0, 2, 1, 3, 4))  
        vh_s_sqrt = np.reshape(vh_s_sqrt, (self.groups, 
                                           rank,
                                           int(self.original_module.out_channels/self.groups)*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))    
        return u_s_sqrt, vh_s_sqrt   

class StaticLowRankScheme0(StaticLowRankScheme5):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme0, self).__init__(original_module, original_ofm_size, original_module.out_channels)
        self.scheme = 0

class StaticLowRankScheme1(StaticLowRankScheme4):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme1, self).__init__(original_module, original_ofm_size, 1)
        self.scheme = 1

class StaticLowRankScheme3(StaticLowRankScheme4):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme3, self).__init__(original_module, original_ofm_size, original_module.in_channels)
        self.scheme = 3

def svd_target_module_filter(model_name, module_name, module):
    if model_name == "efficientnetb0":
        assert module_name != None
        if isinstance(module, nn.Conv2d) and "fc" not in module_name:
            if module.kernel_size == (1, 1):
                assert module.groups == 1
                return True
        return False
    elif model_name == "mobilenetv2":
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1):
                assert module.groups == 1
                return True
        return False
    elif model_name == "resnet18":
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3) and module.in_channels != 3:
                assert module.groups == 1
                return True
        return False
    else:
        assert False