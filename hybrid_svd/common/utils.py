import sys
import torch
import math
import logging

import numpy as np
import copy
import torch.nn as nn
from torch import Tensor
import functools

from .pytorch_example_imagenet_main import *

import thop
import torchvision

INPUT_IMAGE_CHANNEL = 3
INPUT_IMAGE_HEIGHT = 224
INPUT_IMAGE_WIDTH = 224

def find_conv_bn(model, conv_module):
    pending = False
    for name, module in model.named_modules():
        if pending:
            assert isinstance(module, nn.BatchNorm2d)
            bn = module
            return conv_module, bn

        if module == conv_module:
            pending = True
    assert False

def find_conv_bn_relu(model, conv_module):
    pending = False
    for name, module in model.named_modules():
        if pending:
            if isinstance(module, nn.BatchNorm2d):
                bn = module
            if isinstance(module, nn.ReLU):
                relu = module
                return conv_module, bn, relu

        if module == conv_module:
            pending = True

    assert False

def fuse_batch_normalisation(model, conv_module):
    conv, bn = find_conv_bn(model, conv_module)

    conv.weight, conv.bias = nn.utils.fuse_conv_bn_weights(conv.weight, conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    bn.reset_parameters()

def calculate_macs_params(model, input, turn_on_warnings, verbose=True, custom_compensation={}):
    # MACs and Parameters data
    macs, params = thop.profile(model, inputs=(input, ), verbose=turn_on_warnings)

    # pruning by mask
    for macs_compensation, params_compensation in custom_compensation.values():
        macs -= macs_compensation
        params -= params_compensation

    format_macs, format_params = thop.clever_format([macs, params], "%.3f")
    if verbose:
        print("MACs:", format_macs, "Params:", format_params)
    return macs, params

def update_feature_map_size(name, module, current_feature_map_size):
    if isinstance(module, nn.MaxPool2d):
        return (current_feature_map_size[0] / module.stride, current_feature_map_size[1] / module.stride)
    elif isinstance(module, nn.Conv2d) and "downsample" not in name:    
        return (current_feature_map_size[0] / module.stride[0], current_feature_map_size[1] / module.stride[1])
    else:
        return current_feature_map_size

def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_factors(n):
    return np.sort(list(set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))

def image_visualiser(input_tensor):
    # export imagenet-style tensor into image

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    im = transforms.ToPILImage()(inv_normalize(input_tensor)).convert("RGB")
    
    return im

def load_torch_vision_model(model_name):
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "efficientnetb0":
        model = torchvision.models.efficientnet_b0(pretrained=True)
    else:
        assert False
    
    model.cpu()
    return model