import torch
import torch.nn as nn
import os

from distilled_data import GaussianDataset, NumpyFolderDataset
from hybrid_svd.common.utils import update_learning_rate
from hybrid_svd.svd.svd_utils import *
from hybrid_svd.svd.svd_residual import generate_low_rank_residual_wrapper

import argparse
import copy

import csv
import numpy as np

def weighted_mse_loss(weight, input, target):
    loss =  ((weight*(input-target))**2).mean()
    noramlised_loss = loss / ((weight.detach()** 2).mean())
    return noramlised_loss

def kd_kl_loss(student_outputs, labels, teacher_outputs):
    alpha = 0.95
    T = 6
    KD_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(student_outputs/T, dim=1),
                             nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              nn.functional.cross_entropy(student_outputs, labels) * (1. - alpha)

    return KD_loss

def kd_load_synthesised_dataset(model_name, data_path, label_path, image_num, kd_method):
    assert model_name in data_path, "loading incorrect dataset"

    distilled_data = torch.load(data_path,map_location=torch.device('cpu'))
    distilled_data = torch.cat(distilled_data, dim = 0)
    if label_path:
        distilled_label = torch.load(label_path, map_location=torch.device('cpu'))
        distilled_label = torch.cat(distilled_label, dim = 0)
    else:
        distilled_label = torch.zeros(distilled_data.size(0))
        assert "entropy" not in kd_method, "unlabelled dataset"
    kd_dataset = torch.utils.data.TensorDataset(distilled_data, distilled_label)

    assert image_num <= distilled_data.size(0)

    return kd_dataset

def kd_main(args):

    torch.manual_seed(0)
    np.random.seed(0)
    model = load_torch_vision_model(args.model_name)

    random_input = torch.randn(1, INPUT_IMAGE_CHANNEL, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')

    if args.gpu is not None:
        print("Using gpu " + str(args.gpu))
        torch.cuda.set_device(args.gpu)

    # dataset to fine-tune the weights
    if args.kd_data_src == "random":
        kd_data_loader = torch.utils.data.DataLoader(GaussianDataset(args.image_num, (INPUT_IMAGE_CHANNEL, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)
    else:
        if args.kd_data_src:
            if os.path.isdir(args.kd_data_src):
                kd_dataset = NumpyFolderDataset(args.kd_data_src)
            else:
                kd_dataset = kd_load_synthesised_dataset(args.model_name, args.kd_data_src, None, args.image_num, args.low_rank_loss)

        else:
            kd_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                            ]))

            print(len(kd_dataset))

        kd_subdataset, _ = torch.utils.data.random_split(kd_dataset, (args.image_num,len(kd_dataset)-args.image_num))

        kd_data_loader = torch.utils.data.DataLoader(
            kd_subdataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

    # dataset to report accuracy
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #for batch_idx, (images, target) in enumerate(kd_data_loader):
    #    im = image_visualiser(images[0])
    #    im.save("{}.png".format(batch_idx))

    # generate low rank
    conv_mapping = {}
    input_output_mapping = {}
    original_dict = {}
    replace_dict = {}

    low_rank_model = copy.deepcopy(model)    
    if args.freeze == 1:
        for name, param in low_rank_model.named_parameters():
            param.requires_grad = False

    if args.svd_option == "B2":
        conv_layer_index = 0
        current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
        for name, module in low_rank_model.named_modules():
            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
            if svd_target_module_filter(args.model_name, name, module): 
                scheme = args.approximate_scheme[conv_layer_index]
                if scheme not in [-1, -2]:
                    group = args.approximate_groups[conv_layer_index] if args.approximate_groups != None else -1

                    last_iteration_low_rank_conv_wrapper = generate_low_rank_residual_wrapper(args.last_iteration_schemes[conv_layer_index], module, current_output_feature_map_size, args.last_iteration_groups[conv_layer_index])
                    last_iteration_low_rank_conv_wrapper.initialise_low_rank_module(args.last_iteration_ranks[conv_layer_index])
                
                    last_iteration_low_rank_conv_wrapper.generate_low_rank_weight()
                    
                    last_iteration_low_rank_conv = last_iteration_low_rank_conv_wrapper.export_decomposition()
                    replace_dict[module] = last_iteration_low_rank_conv

                    low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, last_iteration_low_rank_conv.residual_conv, current_output_feature_map_size, group)
                    low_rank_conv_wrapper.initialise_low_rank_module(args.rank[conv_layer_index])
                    low_rank_conv_wrapper.generate_low_rank_weight()
                    low_rank_conv = low_rank_conv_wrapper.export_decomposition()
                    last_iteration_low_rank_conv.residual_conv = low_rank_conv
                    if args.freeze == 1:
                        low_rank_conv.set_requires_grad()
                        last_iteration_low_rank_conv.set_requires_grad()

                conv_layer_index += 1
    elif args.svd_option == "B1":
        conv_layer_index = 0
        current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
        for name, module in low_rank_model.named_modules():
            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
            if svd_target_module_filter(args.model_name, name, module): 
                scheme = args.approximate_scheme[conv_layer_index]
                if scheme not in [-1, -2]:
                    group = args.approximate_groups[conv_layer_index] if args.approximate_groups != None else -1

                    low_rank_conv_wrapper = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size, group)

                    low_rank_conv_wrapper.initialise_low_rank_module(args.rank[conv_layer_index])
                    low_rank_conv_wrapper.generate_low_rank_weight()
                    low_rank_conv = low_rank_conv_wrapper.export_decomposition()     
                    if args.freeze == 1:
                        low_rank_conv.set_requires_grad()
                    
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

    print("distillation")
    # register hook
    conv_input = {}
    conv_output = {}
    conv_output_grad = {}
    low_rank_conv_output = {}

    def register_kd_hook(handler_collection, origin_start, origin_end, low_rank_start, low_rank_end, low_rank_loss):
        def log_conv_input(m, input, output):
            conv_input[m] = input[0]

        def log_conv_output(m, input, output):
            conv_output[m] = output

        def log_conv_output_grad(m, grad_input, grad_output):
            conv_output_grad[m] = grad_output[0]

        def log_low_rank_conv_output(m, input, output):
            low_rank_conv_output[m] = output

        if low_rank_loss in ["entropy", "kd_entropy"]:
            pass
        elif low_rank_loss in ["kd_entropy_l1", "kd_entropy_l2", "kd_l1", "kd_l2", "random_bn_l2"]:
            handler_collection.append(origin_end.register_forward_hook(log_conv_output))
            handler_collection.append(low_rank_end.register_forward_hook(log_low_rank_conv_output))
        elif low_rank_loss in ["greedy_l1", "greedy_l2"]:
            handler_collection.append(origin_start.register_forward_hook(log_conv_input))
            handler_collection.append(origin_end.register_forward_hook(log_conv_output))
        elif low_rank_loss in ["greedy_weighted_l2"]:
            handler_collection.append(origin_start.register_forward_hook(log_conv_input))
            handler_collection.append(origin_end.register_forward_hook(log_conv_output))
            handler_collection.append(origin_end.register_backward_hook(log_conv_output_grad))
        else:
            assert False

    handler_collection = []
    conv_layer_index = 0
    low_rank_module_index = 0
    low_rank_module_list = list(replace_dict.values())
    bn_stats_dict = {}
    for name, module in model.named_modules():
        if svd_target_module_filter(args.model_name, name, module):
            scheme = args.approximate_scheme[conv_layer_index]
            if scheme != -1:
                low_rank_conv = low_rank_module_list[low_rank_module_index]

                input_output_mapping[module] = module
                conv_mapping[module] = nn.Sequential(low_rank_conv)
                register_kd_hook(handler_collection, module, module, low_rank_conv, low_rank_conv, args.low_rank_loss)
                
                low_rank_module_index +=1
            conv_layer_index += 1
    
    meter_dict = {}
    optimizer_dict = {}
    if args.low_rank_loss in ["entropy", "kd_entropy", "kd_entropy_l1", "kd_entropy_l2", "kd_l1", "kd_l2", "random_bn_l2"]:
        losses = AverageMeter('Loss', ':.4e')
        meter_dict[low_rank_model] = losses

        optimizer_dict[low_rank_model] = torch.optim.SGD(filter(lambda p: p.requires_grad, low_rank_model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.low_rank_loss in ["greedy_l1", "greedy_l2", "greedy_weighted_l2"]:
        for orginal_module, low_rank_module in conv_mapping.items():
            losses = AverageMeter('Loss', ':.4e')
            meter_dict[orginal_module] = losses  
            optimizer_dict[low_rank_module] = torch.optim.SGD(filter(lambda p: p.requires_grad, low_rank_module.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False
    
    lr_decay = 1.0
    for epoch in range(100*int(1281167/args.image_num)):
        print("epoch {}:".format(epoch))

        if args.freeze == 1:
            low_rank_model.eval()
        else:
            low_rank_model.train()

        lr = args.lr*lr_decay
        for optimizer in optimizer_dict.values():
            update_learning_rate(optimizer, lr)

        if args.kd_data_src == "random":
            torch.manual_seed(0)
            np.random.seed(0)

        for batch_idx, (images, target) in enumerate(kd_data_loader):
        
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if args.low_rank_loss in ["entropy"]:
                low_rank_output = low_rank_model(images)
                loss = nn.CrossEntropyLoss()(low_rank_output,target)

                meter_dict[low_rank_model].update(loss, images.size(0))
                optimizer_dict[low_rank_model].zero_grad()
                loss.backward()
                optimizer_dict[low_rank_model].step()
            elif args.low_rank_loss in ["kd_entropy"]:
                with torch.no_grad():
                    output = model(images)
                low_rank_output = low_rank_model(images)
                loss = kd_kl_loss(low_rank_output, target, output)

                meter_dict[low_rank_model].update(loss, images.size(0))
                optimizer_dict[low_rank_model].zero_grad()
                loss.backward()
                optimizer_dict[low_rank_model].step()
            elif args.low_rank_loss in ["kd_entropy_l1", "kd_entropy_l2"]:
                with torch.no_grad():
                    output = model(images)
                low_rank_output = low_rank_model(images)
                loss = kd_kl_loss(low_rank_output, target, output)
                for orginal_module, low_rank_module in conv_mapping.items():

                    if args.low_rank_loss == "kd_entropy_l1":
                        activation_loss = nn.L1Loss()(low_rank_conv_output[low_rank_module[-1]], conv_output[input_output_mapping[orginal_module]].detach())
                    else:
                        activation_loss = nn.MSELoss()(low_rank_conv_output[low_rank_module[-1]], conv_output[input_output_mapping[orginal_module]].detach())

                    loss += activation_loss

                meter_dict[low_rank_model].update(loss, images.size(0))
                optimizer_dict[low_rank_model].zero_grad()
                loss.backward()
                optimizer_dict[low_rank_model].step()

            elif args.low_rank_loss in ["kd_l1", "kd_l2"]:
                with torch.no_grad():
                    output = model(images)
                low_rank_output = low_rank_model(images)
                bFirst = True
                for orginal_module, low_rank_module in conv_mapping.items():

                    if args.low_rank_loss == "kd_l1":
                        activation_loss = nn.L1Loss()(low_rank_conv_output[low_rank_module[-1]], conv_output[input_output_mapping[orginal_module]].detach())
                    else:
                        activation_loss = nn.MSELoss()(low_rank_conv_output[low_rank_module[-1]], conv_output[input_output_mapping[orginal_module]].detach())

                    if bFirst:
                        loss = activation_loss
                        bFirst = False
                    else:
                        loss += activation_loss

                meter_dict[low_rank_model].update(loss, images.size(0))
                optimizer_dict[low_rank_model].zero_grad()
                loss.backward()
                optimizer_dict[low_rank_model].step()

            elif args.low_rank_loss in ["greedy_l1", "greedy_l2"]:
                with torch.no_grad():
                    model(images)

                for orginal_module, low_rank_module in conv_mapping.items():            
                    low_rank_output = low_rank_module(conv_input[orginal_module].detach())
                    
                    if args.low_rank_loss == "greedy_l1":
                        loss = nn.L1Loss()(low_rank_output, conv_output[input_output_mapping[orginal_module]].detach())
                    else:
                        loss = nn.MSELoss()(low_rank_output, conv_output[input_output_mapping[orginal_module]].detach())

                    meter_dict[orginal_module].update(loss, images.size(0))
                    optimizer_dict[low_rank_module].zero_grad()
                    loss.backward()
                    optimizer_dict[low_rank_module].step()

            elif args.low_rank_loss in ["greedy_weighted_l2"]:
                output = model(images)
                loss = nn.CrossEntropyLoss()(output,target)
                model.zero_grad()
                loss.backward()

                for orginal_module, low_rank_module in conv_mapping.items():            
                    low_rank_output = low_rank_module(conv_input[orginal_module].detach())
                    loss = weighted_mse_loss(conv_output_grad[input_output_mapping[orginal_module]].detach(), low_rank_output, conv_output[input_output_mapping[orginal_module]].detach())

                    meter_dict[orginal_module].update(loss, images.size(0))
                    optimizer_dict[low_rank_module].zero_grad()
                    loss.backward()
                    optimizer_dict[low_rank_module].step()

        meter_dict["val_top1"] , meter_dict["val_top5"] = validate(val_loader, low_rank_model, nn.CrossEntropyLoss())
            
        if epoch == 0 or meter_dict["val_top1"].avg > meter_dict["best_top1"].avg:
            meter_dict["best_top1"] = meter_dict["val_top1"]
            early_stop_patience = 0
            torch.save(low_rank_model.state_dict(), os.path.join(args.output_path, "low_rank_model_state_dict"))
        else:
            early_stop_patience += 1

        meter_dict = dump_meter(epoch, meter_dict, os.path.join(args.output_path, "meter_log.csv"))             
        
        if early_stop_patience > 10:
            lr_decay *= 0.1
            early_stop_patience = 0
            if args.lr * lr_decay < 0.0001:
                print("Early Stop")
                break
            else:
                print("Learning Rate Decay")


    for handler in handler_collection:
        handler.remove() 
    low_rank_model.load_state_dict(torch.load(os.path.join(args.output_path, "low_rank_model_state_dict")))
    torch.save(low_rank_model, os.path.join(args.output_path, "low_rank_model"))
    os.remove(os.path.join(args.output_path, "low_rank_model_state_dict"))

def dump_meter(start, meter_dict, output_path):
    csv_entry = [start]


    for meter_name, meter_obj in meter_dict.items():
        csv_entry.append(meter_obj.avg.item())        
        if meter_name == "val_top1":
            print(' * Val Acc@1 {top1.avg:.3f}'.format(top1=meter_obj))
        elif meter_name == "val_top5":
            print(' * Val Acc@5 {top5.avg:.3f}'.format(top5=meter_obj))
        elif meter_name == "best_top1":
            print(' * Best Val Acc@1 {top1.avg:.3f}'.format(top1=meter_obj))
        else:
            print("module loss: {}".format(meter_obj.avg))
            meter_obj.reset()            

    with open(output_path, mode='a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_entry) 

    return meter_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVD optimization')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id (main) to use.')
    parser.add_argument('--data_parallel', default=None, type=int, metavar='N', nargs='+',
                        help='GPU ids to use.')


    parser.add_argument('--data', default="ILSVRC2012_img", type=str,
                        help='directory of ImageNet')
    parser.add_argument('--batch_size', default='64', type=int, 
                        help='')
    parser.add_argument('--workers', default='4', type=int, 
                        help='')

    parser.add_argument('-s', '--approximate_scheme', default=None, type=int, metavar='N', nargs='+',
                        help='used for B1') 
    parser.add_argument('-g', '--approximate_groups', default=None, type=int, metavar='N', nargs='+',
                        help='used for B1')                     
    parser.add_argument('-r', '--rank', default=None, type=int, metavar='N', nargs='+',
                        help='used for B1')               

    parser.add_argument('--output_path', default=None, type=str,
                        help='output path')

    parser.add_argument('--kd_data_src', default="zeroq_img25000_distilled_data_resnet18_32_dir", type=str,
                        help='for few-sample and full-training, set it as None; for post-training, specify the path of the synthesised dataset')                    
    parser.add_argument('--image_num', default=25000, type=int,
                        help='')

    parser.add_argument('--svd_option', default="B1", choices=["B1", "B2"], type=str,
                        help='')  
    parser.add_argument('--low_rank_loss', default="kd_l2", type=str,
                        help='choose kd_l2 for synthesised dataset, choose kd_entropy_l2 for few-sample, choose entropy for full-training')
    parser.add_argument('--last_iteration_schemes', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only') 
    parser.add_argument('--last_iteration_groups', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only')                     
    parser.add_argument('--last_iteration_ranks', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only')  

    parser.add_argument('--model_name', default='resnet18', choices=['resnet18', "mobilenetv2", "efficientnetb0"], type=str,
                        help='output path')

    args = parser.parse_args()

    if args.output_path == None:
        args.output_path = os.getcwd() + "/output"

    args.freeze = 0
    args.lr = 0.001
    # freeze the rest of the model when using distilled dataset
    if args.kd_data_src != None:
        args.freeze = 1

    args.momentum = 0.9
    args.weight_decay = 1e-4

    print(args)
    kd_main(args)
