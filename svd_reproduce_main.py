import torch
import torchprune as tp
import copy

from hybrid_svd.common.utils import *
from hybrid_svd.svd.svd_utils import *

import argparse
import os


def alds_main(model, random_input, args, val_loader):
    low_rank_model = copy.deepcopy(model)

    low_rank_model = tp.util.net.NetHandle(low_rank_model)
    low_rank_model = tp.ALDSNet(low_rank_model, None, None)

    if torch.cuda.is_available():
        random_input = random_input.cuda()
        low_rank_model.cuda()

    low_rank_model.eval()

    with torch.no_grad():
        low_rank_model(random_input)
    low_rank_model.compress(keep_ratio=(1-args.compress_hyper))

    print(
        f"The network has {low_rank_model.size()} parameters and "
        f"{low_rank_model.flops()} FLOPs left."
    )

    low_rank_model = low_rank_model.compressed_net.torchnet

    calculate_macs_params(low_rank_model, random_input, False)
    validate(val_loader, low_rank_model, nn.CrossEntropyLoss())

def fgroup_main(model, random_input, args, val_loader):
    assert args.model_name == "resnet18"

    approximate_scheme = [4,4,4,4,4,4,4,4,4,4,4,4,-1,-1,-1,-1]
    approximate_rank = [8,8,8,8,32,32,32,32,128,128,128,128,-1,-1,-1,-1]

    low_rank_model = copy.deepcopy(model)

    replace_dict = {}
    conv_layer_index = 0
    current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    for name, module in low_rank_model.named_modules():
        current_input_feature_map_size = current_output_feature_map_size
        current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
        if svd_target_module_filter(args.model_name, name, module): 
            scheme = approximate_scheme[conv_layer_index]
            rank = approximate_rank[conv_layer_index]

            if scheme != -1:

                low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size, int(module.in_channels/rank))

                low_rank_conv_wrapper.initialise_low_rank_module(rank)
                low_rank_conv_wrapper.generate_low_rank_weight()
                low_rank_conv = low_rank_conv_wrapper.export_decomposition()
                                
                replace_dict[module] = low_rank_conv

            conv_layer_index += 1


    for name, module in low_rank_model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                decomposed_conv = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,decomposed_conv)


    model.eval()
    low_rank_model.eval()

    if torch.cuda.is_available():
        model.cuda()
        low_rank_model.cuda()
        random_input = random_input.cuda()

    calculate_macs_params(low_rank_model, random_input, False)
    validate(val_loader, low_rank_model, nn.CrossEntropyLoss())

def trp_main(model, random_input, args, val_loader):
    scheme = 1

    low_rank_model = copy.deepcopy(model)

    conv_layer_index = 0
    replace_dict = {}
    current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    for name, module in low_rank_model.named_modules():
        current_input_feature_map_size = current_output_feature_map_size
        current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
        if svd_target_module_filter(args.model_name, name, module): 
            low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size)

            original_weight = low_rank_conv_wrapper.original_module.weight.detach().clone().numpy()
            unfolded_weight = low_rank_conv_wrapper.unfold_original_weight(original_weight)
            [u,s,vh] = np.linalg.svd(unfolded_weight, full_matrices=False) 

            assert s.shape[0]==1
            s = s[0]
            s = s**2

            for r in range(len(s)):
                if np.sum(s[r:]) < (1-args.compress_hyper)*np.sum(s):
                    break
            
            low_rank_conv_wrapper.initialise_low_rank_module(r)
            low_rank_conv_wrapper.generate_low_rank_weight()
            low_rank_conv = low_rank_conv_wrapper.export_decomposition()        
            replace_dict[module] = low_rank_conv

            conv_layer_index += 1

    for name, module in low_rank_model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                decomposed_conv = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,decomposed_conv)


    model.eval()
    low_rank_model.eval()

    if torch.cuda.is_available():
        model.cuda()
        low_rank_model.cuda()
        random_input = random_input.cuda()

    calculate_macs_params(low_rank_model, random_input, False)
    validate(val_loader, low_rank_model, nn.CrossEntropyLoss())

def learning_rank_main(model, random_input, args, val_loader):
    if args.reproduce_method == "learning_s1":
        scheme = 1
    elif args.reproduce_method == "learning_s2":
        scheme = 2

    low_rank_model = copy.deepcopy(model)

    candidates = []
    rank_counters = []
    conv_layer_index = 0
    current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    for name, module in low_rank_model.named_modules():
        current_input_feature_map_size = current_output_feature_map_size
        current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
        if svd_target_module_filter(args.model_name, name, module): 
            low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size)

            original_weight = low_rank_conv_wrapper.original_module.weight.detach().clone().numpy()
            unfolded_weight = low_rank_conv_wrapper.unfold_original_weight(original_weight)
            [u,s,vh] = np.linalg.svd(unfolded_weight, full_matrices=False) 

            assert s.shape[0]==1
            s = s[0]
            s = s**2

            layer_score = -s+args.compress_hyper*low_rank_conv_wrapper.get_original_module_macs()
            candidates += [(conv_layer_index, x) for i,x in enumerate(layer_score)]
            rank_counters.append(low_rank_conv_wrapper.get_max_rank())
            
            conv_layer_index += 1

    sorted_candidates = sorted(candidates, key=lambda i: i[1], reverse=True) 

    for conv_layer_index, score in sorted_candidates:
        if score <=0 or rank_counters[conv_layer_index] == 1:
            break
        rank_counters[conv_layer_index] -= 1


    replace_dict = {}
    conv_layer_index = 0
    current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    for name, module in low_rank_model.named_modules():
        current_input_feature_map_size = current_output_feature_map_size
        current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
        if svd_target_module_filter(args.model_name, name, module): 
            rank= rank_counters[conv_layer_index]
            low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size)
            
            if rank * low_rank_conv_wrapper.get_per_rank_macs() < low_rank_conv_wrapper.get_original_module_macs():
                low_rank_conv_wrapper.initialise_low_rank_module(rank)
                low_rank_conv_wrapper.generate_low_rank_weight()
                low_rank_conv = low_rank_conv_wrapper.export_decomposition()        
                replace_dict[module] = low_rank_conv

            conv_layer_index += 1


    for name, module in low_rank_model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                decomposed_conv = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,decomposed_conv)


    model.eval()
    low_rank_model.eval()

    if torch.cuda.is_available():
        model.cuda()
        low_rank_model.cuda()
        random_input = random_input.cuda()

    calculate_macs_params(low_rank_model, random_input, False)
    validate(val_loader, low_rank_model, nn.CrossEntropyLoss())

parser = argparse.ArgumentParser(description='SVD optimization')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id (main) to use.')
parser.add_argument('--data_parallel', default=None, type=int, metavar='N', nargs='+',
                    help='GPU ids to use.')


parser.add_argument('--data', default="ILSVRC2012_img", type=str,
                    help='directory of the dataset')
parser.add_argument('--batch_size', default='64', type=int, 
                    help='')
parser.add_argument('--workers', default='4', type=int, 
                    help='')

parser.add_argument('--model_name', default="efficientnetb0", type=str,
                    help='')       

parser.add_argument('--output_path', default=None, type=str,
                    help='output path')


parser.add_argument('--reproduce_method', default="alds", type=str,
                    help='')
parser.add_argument('--compress_hyper', default=0.05, type=float,
                    help='')

args = parser.parse_args()

if args.output_path == None:
    args.output_path = os.getcwd() + "/output"

print(args)

torch.manual_seed(0)
np.random.seed(0)

random_input = torch.randn(1, INPUT_IMAGE_CHANNEL, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
if args.gpu is not None:
    print("Using gpu " + str(args.gpu))
    torch.cuda.set_device(args.gpu)

valdir = os.path.join(args.data, 'val')
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

model = load_torch_vision_model(args.model_name)

method_dict = {
                "alds": alds_main,
                "fgroup":fgroup_main,
                "learning_s1":learning_rank_main,
                "learning_s2":learning_rank_main,
                "trp":trp_main
                }

method_dict[args.reproduce_method](model, random_input, args, val_loader)

