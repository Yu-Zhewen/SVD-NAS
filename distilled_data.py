import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import DatasetFolder
import numpy as np
from hybrid_svd.common.utils import *


def numpy_to_tensor_loader(path):
    return np.load(path)

class NumpyFolderDataset(DatasetFolder):
    def __init__(self, root):
        super(NumpyFolderDataset, self).__init__(root, numpy_to_tensor_loader, (".npy"))

class ZeroqUniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = (torch.randint(high=255, size=self.size).float() -
                  127.5) / 5418.75
        return sample

class ZeroqDatasetGenerator():
    def __init__(self, model_name, model, img_num, batch_size):
        self.image_num = img_num
        self.batch_size = batch_size

        if torch.cuda.is_available():
            model = model.cuda()
        model = model.eval()

        self.eps = 1e-6
        self.bn_inputs_dict = {}
        self.bn_stats_dict = {}
        self.handler_collection = []

        def log_bn_input(m, input, output):
            self.bn_inputs_dict[m] = input[0]

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # get the statistics in the BatchNorm layers
                mean = module.running_mean.detach().clone()
                std  = torch.sqrt(module.running_var + self.eps).detach().clone()

                if torch.cuda.is_available():
                    mean = mean.cuda()
                    std = std.cuda()
                
                self.bn_stats_dict[module] = (mean, std)
                self.handler_collection.append(module.register_forward_hook(log_bn_input))

        self.model = model

        self.data_loader = DataLoader(ZeroqUniformDataset(img_num, (INPUT_IMAGE_CHANNEL, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)

        if model_name == "resnet18":
            self.lr = 0.5
            self.bn_loss_reduce = False
            self.filter_list = []
            self.input_loss_factor = 1
        elif model_name == "mobilenetv2":
            self.lr = 0.25
            self.bn_loss_reduce = True
            self.filter_list = []
            self.input_loss_factor = 1
        elif model_name == "efficientnetb0":
            self.lr = 0.5
            self.bn_loss_reduce = True
            self.filter_list = [6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
            self.input_loss_factor = 100
        else:
            assert False
                
    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):

        for batch_idx, images in enumerate(self.data_loader):
            images_mean = torch.zeros(images.size(0), 3)
            images_std = torch.ones(images.size(0), 3)

            if torch.cuda.is_available():
                images = images.cuda()
                images_mean = images_mean.cuda()
                images_std = images_std.cuda()

            images.requires_grad = True
            optimizer = torch.optim.Adam([images], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-4, verbose=False, patience=100)
            
            iteration_num = 500
            for it_index in range(iteration_num):
                self.model(images)
                
                bn_mean_loss = 0
                bn_std_loss = 0
                input_mean_loss = 0
                input_std_loss = 0

                for layer_index, (bn_layer, (bn_mean, bn_std)) in enumerate(self.bn_stats_dict.items()):

                    if layer_index in self.filter_list:
                        bn_loss_factor = 0
                    else:
                        bn_loss_factor = 1

                    bn_input = self.bn_inputs_dict[bn_layer]

                    current_mean = torch.mean(bn_input.view(bn_input.size(0), bn_input.size(1), -1), dim=2)
                    current_std = torch.sqrt(torch.var(bn_input.view(bn_input.size(0), bn_input.size(1), -1), dim=2) + self.eps)

                    if self.bn_loss_reduce:
                        bn_mean_loss += nn.MSELoss()(bn_mean.expand(bn_input.size(0),-1), current_mean)*bn_loss_factor
                        bn_std_loss += nn.MSELoss()(bn_std.expand(bn_input.size(0),-1), current_std)*bn_loss_factor
                    else:
                        bn_mean_loss += nn.MSELoss(reduction='sum')(bn_mean.expand(bn_input.size(0),-1), current_mean) / bn_input.size(0)*bn_loss_factor
                        bn_std_loss += nn.MSELoss(reduction='sum')(bn_std.expand(bn_input.size(0),-1), current_std) / bn_input.size(0)*bn_loss_factor

                    #print(layer_index, bn_mean_loss, bn_std_loss)

                current_mean = torch.mean(images.view(images.size(0), images.size(1),-1), dim=2)
                current_std = torch.sqrt(torch.var(images.view(images.size(0), images.size(1), -1), dim=2) + self.eps)
                
                if self.bn_loss_reduce:
                    input_mean_loss += nn.MSELoss()(images_mean, current_mean)
                    input_std_loss += nn.MSELoss()(images_std, current_std)
                else:                          
                    input_mean_loss += nn.MSELoss(reduction='sum')(images_mean, current_mean) / images.size(0)
                    input_std_loss += nn.MSELoss(reduction='sum')(images_std, current_std) / images.size(0)

                total_loss = bn_mean_loss + bn_std_loss + input_mean_loss*self.input_loss_factor + input_std_loss*self.input_loss_factor

                #print(total_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step(total_loss.item())

            yield images.detach().clone(), torch.tensor([])

    def random_seed_reset(self):
        torch.manual_seed(0)
        np.random.seed(0)

def generate_zeroq_distilled_dataset(model_name, model, img_num, batch_size, output_path):
    data_loader = ZeroqDatasetGenerator(model_name, model, img_num, batch_size)
    distilled_data = []

    img_count = 0
    for batch_index, (images, target) in enumerate(data_loader):
        print(batch_index)
        for image in images:
            np.save(os.path.join(output_path, "{}.npy".format(img_count)), image.cpu().numpy())
            img_count += 1
            
class GaussianDataset(Dataset):
    """
    get random gaussian samples with mean 0 and variance 1
    """
    def __init__(self, length, size):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(self.size)
        target = 0

        return image, target

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Distilled Dataset')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id (main) to use.')
    parser.add_argument('--output_path', default='output', type=str,
                        help='output path')
    parser.add_argument('--batch_size', default='32', type=int, 
                        help='')
    parser.add_argument('--image_num', default=25000, type=int,
                        help='')
    parser.add_argument('--model_name', default='resnet18', choices=['resnet18', "mobilenetv2", "efficientnetb0"], type=str,
                        help='output path')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    model = load_torch_vision_model(args.model_name)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    datset_dir = "zeroq_img{}_distilled_data_{}_{}_dir".format(args.image_num, args.model_name, args.batch_size)
    datset_dir = os.path.join(args.output_path, datset_dir)
    os.mkdir(datset_dir)
    generate_zeroq_distilled_dataset(args.model_name, model, args.image_num, args.batch_size, datset_dir)
