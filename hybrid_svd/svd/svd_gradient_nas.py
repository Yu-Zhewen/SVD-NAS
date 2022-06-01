import numpy as np
from sqlitedict import SqliteDict
import copy
import torch
import torch.nn as nn
from scipy.special import softmax
import scipy.stats as stats

from hybrid_svd.svd.svd_utils import *
from hybrid_svd.svd.svd_residual import generate_low_rank_residual_wrapper

class LowRankDecompositionSuperBlock(nn.Module):
    def __init__(self, model_name, module_name, conv_layer_index, original_module, original_ifm_size, original_ofm_size, candidates, dropout):
        super(LowRankDecompositionSuperBlock, self).__init__()
        simulated_candidates = []
        self.candidates = []
        performance_losses = []

        for scheme_wrapper, rank_list in candidates.items():
            original_weight = scheme_wrapper.original_module.weight.detach().clone().numpy()
            unfolded_weight = scheme_wrapper.unfold_original_weight(original_weight)
            [u, s, vh] = np.linalg.svd(unfolded_weight, full_matrices=False)

            for r_index, rank in enumerate(rank_list):
                performance_loss = rank*scheme_wrapper.get_per_rank_macs()
                
                s_masked = copy.deepcopy(s)
                s_masked[:, rank:] = 0

                u_s_sqrt = np.zeros_like(u)
                vh_s_sqrt = np.zeros_like(vh)

                for j in range(scheme_wrapper.groups):
                    s_sqrt_diag = np.diag(np.sqrt(s_masked[j]))
                    u_s_sqrt[j] = u[j] @ s_sqrt_diag
                    vh_s_sqrt[j] = s_sqrt_diag @ vh[j]

                approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                approximated_weight = scheme_wrapper.fold_original_weight(approximated_unfolded_weight)
            
                simulate_low_rank_module = copy.deepcopy(original_module)
                simulate_low_rank_module.weight.data.copy_(torch.from_numpy(approximated_weight))

                simulated_candidates.append(simulate_low_rank_module) 
                self.candidates.append((scheme_wrapper,rank)) # candidates are not ModuleList !
                performance_losses.append(performance_loss)

        # add original conv as one candidate
        simulated_candidates.append(copy.deepcopy(original_module))
        self.candidates.append((-1, -1))
        performance_losses.append(scheme_wrapper.get_original_module_macs())

        self.simulated_candidates = nn.ModuleList(simulated_candidates)
        performance_losses = torch.tensor(performance_losses, requires_grad=False)
        self.register_buffer("performance_losses", performance_losses)

        self.super_theta = nn.Parameter(torch.ones(len(self.candidates)),requires_grad=True)

        if dropout > 1:
            self.dropout = dropout
        else:
            self.dropout = int(len(self.candidates) * dropout)

        if self.dropout > len(self.candidates):
            self.dropout = len(self.candidates)


    def generate_random_sample(self, mean=0.8, std_dev=0.15, step=0.05):
        self.range_list = [(x, x+step) for x in np.arange(0, 1, step)]
        self.range_dict = {}
        baseline_performance = self.performance_losses[-1]
        for i, performance in enumerate(self.performance_losses):
            performace_ratio = performance / baseline_performance

            bFound = False
            for range_bin in self.range_list:
                if range_bin[0] < performace_ratio and performace_ratio <= range_bin[1]:
                    if range_bin not in self.range_dict.keys():
                        self.range_dict[range_bin] = []
                    self.range_dict[range_bin].append(i)
                    bFound = True
                    break
            assert bFound

        self.range_dict = dict(sorted(self.range_dict.items()))
        self.range_prob = []

        for range_bin in self.range_dict.keys():
            self.range_prob.append(stats.norm.cdf(range_bin[1], loc=mean, scale=std_dev)-stats.norm.cdf(range_bin[0], loc=mean, scale=std_dev))                                                                     
        
        self.range_prob = self.range_prob / np.sum(self.range_prob)

    def sample(self, hard=True, mode="search"):
        if mode == "enumerate":
            candidate_index = self.enumerate_counter
        elif mode == "random":
            candidate_bin_index = np.random.choice(list(range(len(self.range_dict))), p=self.range_prob)
            candidate_bin = list(self.range_dict.keys())[candidate_bin_index]
            candidate_index = np.random.choice(self.range_dict[candidate_bin])
        else:
            if hard:
                candidate_index = np.argmax(self.super_theta.detach().cpu().numpy())
            else:
                distribution = softmax(self.super_theta.detach().cpu().numpy())
                candidate_index = np.random.choice(list(range(len(self.candidates))), p=distribution)

        selected_candidate, rank = copy.deepcopy(self.candidates[candidate_index])
        if selected_candidate not in [-1, -2]:
            selected_candidate.initialise_low_rank_module(rank)
            selected_candidate.generate_low_rank_weight()

        return selected_candidate

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        distribution = softmax(self.super_theta.detach().cpu().numpy()/self.temperature)
        self.sampled_candidates = np.random.choice(list(range(len(self.candidates))), self.dropout, replace=False, p=distribution)
        
        self.super_theta_last_iteration = self.super_theta.detach().clone()

        theta = self.super_theta[self.sampled_candidates].repeat(batch_size, 1)
        mask = nn.functional.gumbel_softmax(theta, self.temperature)
        
        self.weighted_performance_losses = mask * self.performance_losses[self.sampled_candidates].repeat(batch_size, 1)
        self.weighted_performance_losses = torch.sum(self.weighted_performance_losses) / batch_size

        for i, candidate_index in enumerate(self.sampled_candidates):
            simulate_low_rank_module = self.simulated_candidates[candidate_index]

            y = simulate_low_rank_module(x)
            y = mask[:,i].reshape((batch_size,1,1,1)) * y

            if i == 0:
                out = y
            else:
                out = out + y

        return out

# pruning the design space before NAS
def decomposition_proposer(model_name, original_model, data_loader, proposer_image_num, use_cached=True):
    targets = []
    candidates = []
    conv_layer_index = 0
    current_output_feature_map_size = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    for name, module in original_model.named_modules(): 
        current_input_feature_map_size = current_output_feature_map_size
        current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
        if svd_target_module_filter(model_name, name, module):       
            targets.append((name, module, current_output_feature_map_size))                          
            conv_layer_index += 1   

    # empirical_proposer prunes the design space by FLOPs only
    if model_name == "resnet18":
        sweep_list = [0.5, 0.9, 0.05]
        empirical_proposer = True
    elif model_name == "mobilenetv2":
        sweep_list = [0.6, 0.95, 0.05]
        empirical_proposer = False
    elif model_name == "efficientnetb0":
        sweep_list = [0.5, 0.95, 0.05]
        empirical_proposer = False
    else:
        assert False, "model not supported"

    if not empirical_proposer:
        baseline_model = copy.deepcopy(original_model)

        entropy_criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            entropy_criterion.cuda() 
            baseline_model.cuda()

        acc1, _ = validate(data_loader, baseline_model, entropy_criterion)
        baseline_acc1 = acc1.avg.item()
        print("proposer baseline:", baseline_acc1)

    for target_index, (module_name, original_module, original_ofm_size) in enumerate(targets):
        c = {}
        for scheme_wrapper in enumerate_scheme_candidates(original_module, original_ofm_size):
            if not empirical_proposer:
                max_rank = int(sweep_list[1] * scheme_wrapper.get_original_module_macs()/scheme_wrapper.get_per_rank_macs())
                step_rank = int(sweep_list[2] * scheme_wrapper.get_original_module_macs()/scheme_wrapper.get_per_rank_macs())
                min_rank = int(sweep_list[0] * scheme_wrapper.get_original_module_macs()/scheme_wrapper.get_per_rank_macs())

                acc_degrad_tole = 5

                if max_rank == 0:
                    max_rank == 1

                if step_rank == 0:
                    step_rank = 1

                if min_rank == 0:
                    min_rank = 1

                original_weight = scheme_wrapper.original_module.weight.detach().clone().numpy()
                unfolded_weight = scheme_wrapper.unfold_original_weight(original_weight)
                [u, s, vh] = np.linalg.svd(unfolded_weight, full_matrices=False)
                for i in reversed(range(min_rank, max_rank+1, step_rank)):
                    print("scheme:", scheme_wrapper.scheme, "group:", scheme_wrapper.groups, "rank:", i)

                    module_indentifier = "_".join([str(proposer_image_num), model_name, module_name, str(scheme_wrapper.scheme),str(scheme_wrapper.groups), str(i)])
                    acc_path = "checkpoint/nas_cached/{}_{}_{}_proposer_acc".format(proposer_image_num, model_name, module_name)
                    find_cached = False
                    if use_cached:
                        with SqliteDict(acc_path, autocommit=True) as db:
                            if module_indentifier in db.keys():
                                find_cached = True
                                cached_data = db[module_indentifier]

                    if not find_cached:
                        s_masked = copy.deepcopy(s)
                        s_masked[:, i:] = 0

                        u_s_sqrt = np.zeros_like(u)
                        vh_s_sqrt = np.zeros_like(vh)

                        for j in range(scheme_wrapper.groups):
                            s_sqrt_diag = np.diag(np.sqrt(s_masked[j]))
                            u_s_sqrt[j] = u[j] @ s_sqrt_diag
                            vh_s_sqrt[j] = s_sqrt_diag @ vh[j]

                        approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                        approximated_weight = scheme_wrapper.fold_original_weight(approximated_unfolded_weight)

                        low_rank_model = copy.deepcopy(original_model)

                        conv_layer_index = 0
                        replace_dict = {}
                        for name, module in low_rank_model.named_modules(): 
                            if svd_target_module_filter(model_name, name, module): 
                                if conv_layer_index == target_index:                                    
                                    module.weight.data.copy_(torch.from_numpy(approximated_weight)) 
                                conv_layer_index += 1  

                        entropy_criterion = nn.CrossEntropyLoss()
                        if torch.cuda.is_available():
                            entropy_criterion.cuda() 
                            low_rank_model.cuda()


                        acc1, _ = validate(data_loader, low_rank_model, entropy_criterion)
                        acc1 = acc1.avg.item()

                        if use_cached:
                            with SqliteDict(acc_path, autocommit=True) as db:
                                db[module_indentifier] = acc1
                    else:
                        acc1 = cached_data

                    if acc1 < baseline_acc1 - acc_degrad_tole:
                        print("reject")
                        break
                    else:
                        print("accept")

                rank_range = list(reversed(range(i+1, max_rank+1, step_rank)))
                
                c[scheme_wrapper] = rank_range
            else:
                sweep_range = np.arange(sweep_list[0], sweep_list[1], sweep_list[2])

                rank_list = []
                for proportion in sweep_range:
                    rank = int((proportion*scheme_wrapper.get_original_module_macs())/scheme_wrapper.get_per_rank_macs())
                    if rank not in rank_list and rank > 0:
                        rank_list.append(rank)
                        
                c[scheme_wrapper] = rank_list

        candidates.append(c)

    print("candidates:", candidates)
    return candidates            

def residual_iterative_decomposition_proposer(model_name, module, original_ofm_size, candidate_bound):
    candidates = {}

    for scheme_wrapper in enumerate_scheme_candidates(module, original_ofm_size): 
        rank_list = []

        if (scheme_wrapper.scheme, scheme_wrapper.groups) not in candidate_bound.keys():
            continue

        last_iteration_proportion = candidate_bound[(scheme_wrapper.scheme, scheme_wrapper.groups)]

        sweep_range = np.arange(0.05, min(1-last_iteration_proportion, 0.40), 0.05)
        for proportion in sweep_range:
            rank = int((proportion*scheme_wrapper.get_original_module_macs())/scheme_wrapper.get_per_rank_macs())
            if rank not in rank_list and rank > 0:
                rank_list.append(rank)
        candidates[scheme_wrapper] = rank_list

    return candidates