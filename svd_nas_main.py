import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import csv
import copy
import argparse

from hybrid_svd.common.pytorch_example_imagenet_main import *
from hybrid_svd.svd.svd_gradient_nas import *
from kd_low_rank import kd_load_synthesised_dataset

def nas_main(args):
    torch.manual_seed(0)
    np.random.seed(0)

    random_input = torch.randn(1, INPUT_IMAGE_CHANNEL, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # dataset used to report the accuracy
    valdir = os.path.join(args.data, 'val')

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # dataset used in the design space pruning by accuracy
    assert args.proposer_image_num <= len(val_dataset)
    proposer_dataset, _ = torch.utils.data.random_split(val_dataset, (args.proposer_image_num,len(val_dataset)-args.proposer_image_num))
    proposer_dataloader = torch.utils.data.DataLoader(
        proposer_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # dataset used to train the sampling parameters theta
    if args.search_with_valid_subset:
        search_dataloader = torch.utils.data.DataLoader(
            proposer_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        search_dataloader = val_loader


    if args.gpu is not None:
        print("Using gpu " + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        random_input = random_input.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    model = load_torch_vision_model(args.model_name)
    if torch.cuda.is_available():
        model.cuda()

    macs_before_low_rank, params_before_low_rank = calculate_macs_params(model, random_input, False)
    perf_baseline = macs_before_low_rank

    # replace the targeted layers with super blocks
    model.cpu()
    super_model = copy.deepcopy(model)
    replace_dict = {}
    macs_params_compensation = {}

    if args.svd_option == "B1":
        candidates = decomposition_proposer(args.model_name, super_model, proposer_dataloader, args.proposer_image_num)

        conv_layer_index = 0
        current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
        for name, module in super_model.named_modules(): 
            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
            if svd_target_module_filter(args.model_name, name, module):                                    
                super_block = LowRankDecompositionSuperBlock(args.model_name, name, conv_layer_index, module, current_input_feature_map_size, current_output_feature_map_size, candidates[conv_layer_index], args.candidates_dropout)
                
                if args.sample_mode != "search":
                    del super_block.simulated_candidates
                    if args.sample_mode == "random":
                        super_block.generate_random_sample()

                replace_dict[module] = super_block
                conv_layer_index += 1

    elif args.svd_option == "B2":
        original_candidates = decomposition_proposer(args.model_name, super_model, proposer_dataloader, args.proposer_image_num)

        last_iteration_performance_losses = 0
        conv_layer_index = 0
        current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
        for name, module in super_model.named_modules(): 
            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
            if svd_target_module_filter(args.model_name, name, module): 
                scheme = args.last_iteration_schemes[conv_layer_index]
                group = args.last_iteration_groups[conv_layer_index]

                if scheme != -1:
                    low_rank_conv_wrapper = generate_low_rank_residual_wrapper(scheme, module, current_output_feature_map_size, group)
                    
                    # from B1 to B2, WITHDRAW strategy 
                    delta_rank = int(0.2*low_rank_conv_wrapper.get_original_module_macs() / low_rank_conv_wrapper.get_per_rank_macs())
                    if delta_rank == 0:
                        delta_rank = 1
                    
                    args.last_iteration_ranks[conv_layer_index] -= delta_rank
                    if args.last_iteration_ranks[conv_layer_index] == 0:
                        args.last_iteration_ranks[conv_layer_index] = 1

                    rank = args.last_iteration_ranks[conv_layer_index]

                    low_rank_conv_wrapper.initialise_low_rank_module(rank)
                    
                    low_rank_conv_wrapper.generate_low_rank_weight()

                    low_rank_conv = low_rank_conv_wrapper.export_decomposition()
                    replace_dict[module] = low_rank_conv
                    
                    candidate_bound = {}
                    for scheme_wrapper, rank_list in original_candidates[conv_layer_index].items():
                        candidate_bound[(scheme_wrapper.scheme, scheme_wrapper.groups)] = rank*low_rank_conv_wrapper.get_per_rank_macs()/low_rank_conv_wrapper.get_original_module_macs()
                    
                    candidates = residual_iterative_decomposition_proposer(args.model_name, low_rank_conv.residual_conv, current_output_feature_map_size, candidate_bound)

                    super_block = LowRankDecompositionSuperBlock(args.model_name, name, conv_layer_index, low_rank_conv.residual_conv, current_input_feature_map_size, current_output_feature_map_size, candidates, args.candidates_dropout)
                    low_rank_conv.residual_conv = super_block

                    # log the performance loss of last iteration
                    super_block.performance_losses[-1] -= low_rank_conv_wrapper.get_per_rank_macs() * low_rank_conv_wrapper.rank
                    last_iteration_performance_losses += low_rank_conv_wrapper.get_per_rank_macs() * low_rank_conv_wrapper.rank
                else:
                    dummy_low_rank_conv_wrapper = generate_low_rank_residual_wrapper(0, module, current_output_feature_map_size, 0)
                    last_iteration_performance_losses += dummy_low_rank_conv_wrapper.get_original_module_macs()

                conv_layer_index += 1

        print(args.last_iteration_ranks)

    for name, module in super_model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                super_block = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,super_block)

    if torch.cuda.is_available():
        super_model.cuda()
        model.cuda()

    super_model.eval()
    model.eval()

    entropy_criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        entropy_criterion.cuda()

    theta_list = []
    for name, parameter in super_model.named_parameters():
        if "super_theta" in name:
            parameter.requires_grad = True 
            theta_list.append(parameter) 
        else:
            parameter.requires_grad = False

    theta_optimizer = torch.optim.Adam(theta_list, lr=args.theta_lr, betas=args.theta_betas, weight_decay=args.theta_decay)
        
    for epoch in range(args.total_epoch):
        if epoch == 0:
            current_tempature = args.gumbel_softmax_tempature_init
        else:
            current_tempature = current_tempature * args.gumbel_softmax_tempature_decay

        for m in super_model.modules():
            if isinstance(m, LowRankDecompositionSuperBlock):
                m.temperature = current_tempature 

        iterations_per_epoch = int(50000/len(search_dataloader.dataset)) if args.search_with_valid_subset else 1
        assert iterations_per_epoch in [1,100]

        entropy_losses = AverageMeter('Entropy Loss', ':.4e')
        performance_losses = AverageMeter('Performance Loss', ':.4e')
        total_losses = AverageMeter('Total Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            int(len(search_dataloader)*iterations_per_epoch),
            [entropy_losses, performance_losses, total_losses, top1, top5],
            prefix='Test: ')

        if args.sample_mode == "search":
            for _ in range(iterations_per_epoch):
                for batch_index, (images, target) in enumerate(search_dataloader):
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)

                    output = super_model(images)
                    entropy_loss = entropy_criterion(output, target)


                    bFirst = True
                    for m in super_model.modules():
                        if isinstance(m, LowRankDecompositionSuperBlock):
                            if bFirst:
                                performance_loss = m.weighted_performance_losses
                            else:
                                performance_loss = performance_loss + m.weighted_performance_losses
                            bFirst = False

                    if args.svd_option == "B2":
                        performance_loss += last_iteration_performance_losses

                    total_loss = ((torch.log(performance_loss) / torch.log(torch.tensor(perf_baseline))) ** args.multi_obj_beta) * entropy_loss

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                    total_losses.update(total_loss.item(), images.size(0))
                    entropy_losses.update(entropy_loss.item(), images.size(0))
                    performance_losses.update(performance_loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))


                    if args.print_freq != 0 and batch_index % args.print_freq == 0:
                        progress.display(batch_index)

                    theta_optimizer.zero_grad()
                    total_loss.backward()
                    theta_optimizer.step()
                    
                    for m in super_model.modules():
                        if isinstance(m, LowRankDecompositionSuperBlock):
                            old_thetas = [m.super_theta_last_iteration.data[idx] for idx in m.sampled_candidates]
                            new_thetas = [m.super_theta.data[idx] for idx in m.sampled_candidates]

                            offset = m.temperature * math.log(sum([math.exp(theta/m.temperature) for theta in new_thetas]) / sum([math.exp(theta/m.temperature) for theta in old_thetas]))

                            for idx in m.sampled_candidates:
                                m.super_theta.data[idx] -= offset

            print('Epoch {epoch} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(epoch=epoch, top1=top1, top5=top5))

            print('Total loss {total_losses.avg:.4e} Entropy loss {entropy_losses.avg:.4e} Performance loss {performance_losses.avg:.4e}'
                    .format(total_losses=total_losses, entropy_losses=entropy_losses, performance_losses=performance_losses))

        if (epoch+1) % 10 == 0 or args.sample_mode != "search":
            print('Evaluation')

            print('super theta')
            for name, module in super_model.named_modules(): 
                if isinstance(module, LowRankDecompositionSuperBlock):
                    print(module.super_theta.detach().cpu().numpy())

            for sample_index in range(args.model_samples):
                print('Sample', sample_index)
                macs_params_compensation_copy = copy.deepcopy(macs_params_compensation)
                csv_entry = [epoch]
                selected_model = load_torch_vision_model(args.model_name)

                selected_modules = []
                candidates_size = []
                for name, module in super_model.named_modules(): 
                    if isinstance(module, LowRankDecompositionSuperBlock):
                        
                        if args.sample_mode == "enumerate":
                            current_size = np.product([1] + candidates_size)
                            module.enumerate_counter = len(module.candidates)-1 - int(sample_index / current_size) 
                            candidates_size.append(len(module.candidates))

                        selected_modules.append(module.sample(hard=(sample_index == 0), mode=args.sample_mode))

                if args.svd_option == "B2":
                    for conv_layer_index, scheme in enumerate(args.last_iteration_schemes):
                        if scheme == -1:
                            selected_modules.insert(conv_layer_index, scheme)

                selected_schemes = []
                selected_groups = []
                selected_ranks = []
                for module_index, m in enumerate(selected_modules):
                    if m not in [-1,-2]:
                        selected_schemes.append(m.scheme)
                        selected_groups.append(m.groups)
                        selected_ranks.append(m.rank)
                    else:
                        # no compression
                        selected_schemes.append(m)
                        selected_groups.append(m)
                        selected_ranks.append(m)

                csv_entry.append(selected_schemes)
                csv_entry.append(selected_groups)
                csv_entry.append(selected_ranks)           
                    
                conv_layer_index = 0
                replace_dict = {}
                if args.svd_option == "B2":
                    for name, module in selected_model.named_modules(): 
                        if svd_target_module_filter(args.model_name, name, module):
                            scheme = args.last_iteration_schemes[conv_layer_index]
                            group = args.last_iteration_groups[conv_layer_index]
                            rank = args.last_iteration_ranks[conv_layer_index]
                            if selected_modules[conv_layer_index] != -1 and args.last_iteration_schemes[conv_layer_index] != -1:
                                low_rank_conv_wrapper = generate_low_rank_residual_wrapper(scheme, module, current_output_feature_map_size, group)
                                low_rank_conv_wrapper.initialise_low_rank_module(rank)
                            
                                low_rank_conv_wrapper.generate_low_rank_weight()

                                low_rank_conv = low_rank_conv_wrapper.export_decomposition()
                                replace_dict[module] = low_rank_conv
                                low_rank_conv.residual_conv = selected_modules[conv_layer_index].export_decomposition()
                            conv_layer_index += 1

                else:
                    for name, module in selected_model.named_modules(): 
                        if svd_target_module_filter(args.model_name, name, module):
                            if selected_modules[conv_layer_index] not in [-1,-2]:                                    
                                replace_dict[module] = selected_modules[conv_layer_index].export_decomposition()
                            conv_layer_index += 1
                    
                for name, module in selected_model.named_modules(): 
                    for subname, submodule in module.named_children():
                        if submodule in replace_dict.keys():
                            super_block = replace_dict[submodule]
                            assert(hasattr(module, subname))
                            setattr(module,subname,super_block)

                if torch.cuda.is_available():
                    selected_model.cuda()

                macs_after_low_rank, params_after_low_rank = calculate_macs_params(selected_model, random_input, False, custom_compensation=macs_params_compensation_copy)
                csv_entry.append(macs_after_low_rank)
                csv_entry.append(params_after_low_rank)
                selected_model.eval()

                acc1, acc5 = validate(val_loader, selected_model, entropy_criterion)
                csv_entry.append(acc1.avg.item())
                csv_entry.append(acc5.avg.item())

                if args.model_name == "resnet18":
                    perf_baseline = macs_after_low_rank

                with open(args.output_path + "/nas_log.csv", mode='a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(csv_entry) 

    #torch.save(super_model, os.path.join(args.output_path, "super_model"))


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
                    

    parser.add_argument('--output_path', default=None, type=str,
                        help='output path')

    parser.add_argument('--multi_obj_beta', default=4, type=int,
                        help='')
    parser.add_argument('--sample_mode', default="search", choices=["search", "random", "enumerate"], type=str,
                        help='')

    parser.add_argument('--svd_option', default="B1", choices=["B1", "B2"], type=str,
                        help='')  
    parser.add_argument('--last_iteration_schemes', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only') 
    parser.add_argument('--last_iteration_groups', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only')                     
    parser.add_argument('--last_iteration_ranks', default=None, type=int, metavar='N', nargs='+',
                        help='used for B2 only')  

    parser.add_argument('--model_name', default='resnet18', choices=['resnet18', "mobilenetv2", "efficientnetb0"], type=str,
                        help='output path')

    args = parser.parse_args()

    args.theta_lr = 0.01
    args.theta_decay = 5e-4 
    args.theta_betas = (0.9, 0.999)

    args.total_epoch = 50 if args.svd_option == "residual_iterative" else 100
    args.model_samples = 5 if args.model_name == "resnet18" else 1
    args.candidates_dropout = 2

    args.gumbel_softmax_tempature_init = 0.842 if args.svd_option == "residual_iterative" else 5
    args.gumbel_softmax_tempature_decay = 0.965

    args.proposer_image_num = 500
    
    args.print_freq = 0
    if args.output_path == None:
        args.output_path = os.getcwd() + "/output"

    args.search_with_valid_subset = False

    print(args)

    nas_main(args)