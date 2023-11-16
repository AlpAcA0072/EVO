import sys
sys.path.insert(0, './')

import time
from tqdm import tqdm
from functools import partial
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

import os
sys.path.append('f:\\EVO\\seg_nas_codes')

from supernet.ofa_fanet import OFAFANet
from supernet.ofa_fanetplus import OFAFANetPlus
from evaluation.evaluate import MscEval
from train.train_ofafanet_basic import SUB_SEED, set_running_statistics

from ofa.utils import get_net_device
from torchprofile import profile_macs

import pynvml, psutil
from psutil._common import bytes2human
import time, json

class OFAFANetEvaluator(ABC):
    def __init__(self,
                 supernet: OFAFANet,
                 dl: DataLoader,  # validation dataloader for measuring mIoU
                 sdl: DataLoader,  # a subset train dataloader for re-calibrating BN stats
                 input_size=(1, 3, 512, 1024),  # input data scale for measuring latency and flops
                 num_classes=19  # 19 classes for Cityscapes, 12 for CamVid
                 ):
        self.supernet = supernet
        self.dl = dl
        self.sdl = sdl
        self.input_size = input_size
        self.num_classes = num_classes

    @staticmethod
    def _calc_params(subnet):
        return sum(p.numel() for p in subnet.parameters() if p.requires_grad) / 1e6  # in unit of Million

    @staticmethod
    def _calc_flops(subnet, dummy_data):
        dummy_data = dummy_data.to(get_net_device(subnet))
        return profile_macs(subnet, dummy_data) / 1e9  # in unit of GFLOPs

    @staticmethod
    def measure_latency(subnet, input_size, iterations=2000):
        """ Be aware that latency will fluctuate depending on the hardware operating condition,
        e.g., loading, temperature, etc. """

        print("measuring latency....")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        subnet.eval()
        model = subnet.cuda()
        input = torch.randn(*input_size).cuda()

        GPU_util = []
        GPU_power = []
        GPU_temp = []

        pynvml.nvmlInit()  
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)


        with torch.no_grad():
            for _ in range(10):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            print('=========Speed Testing=========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in tqdm(range(iterations)):
                model(input)
            elapsed_time = time.time() - t_start
                                
            utilization = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_util.append(utilization.used)

            powerusage = pynvml.nvmlDeviceGetPowerUsage(handle)
            # 单位瓦，原始单位毫瓦
            GPU_power.append(powerusage/1000)

            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            GPU_temp.append(temp)

            GPU_util = sum(GPU_util) / len(GPU_util)
            GPU_power = sum(GPU_power) / len(GPU_power)
            GPU_temp = sum(GPU_temp) / len(GPU_temp)

            torch.cuda.synchronize()
            torch.cuda.synchronize()

            consumption = GPU_power * elapsed_time
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        # FPS = 1000 / latency (in ms)
        return latency, GPU_util, consumption, GPU_temp, elapsed_time

    def _measure_latency(self, subnet):
        return self.measure_latency(subnet, self.input_size)

    @staticmethod
    def eval_mIoU(subnet, input_size, dl, sdl, num_classes):

        # reset BN running statistics
        subnet.train()
        set_running_statistics(subnet, sdl)
        subnet.eval()
        # measure mIoU
        with torch.no_grad():
            single_scale = MscEval(input_size[-2:])
            # 进度条
            mIoU, GPU_util, energy_consumption, GPU_temp, time = single_scale(subnet, dl, num_classes)

        return mIoU, GPU_util, energy_consumption, GPU_temp, time

    def evaluate(self, data, report_latency=True):
        """ high-fidelity evaluation by inference on validation data """
        # create dummy data for measuring flops
        dummy_data = torch.rand(*self.input_size)
        # print(dummy_data.shape)

        pynvml.nvmlInit()  
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode()

        partial_name = name.split()[-1]
        
        file = open(os.path.join('/zhaoyifan/EVO/sampled_result/4090', 'ofa_fanet_plus_bottleneck_rtx_fps@0.5_4090.json'), 'w+')
        monitor = open(os.path.join('/zhaoyifan/EVO/sampled_result/4090', 'monitor.log'), 'w+')
        monitor.write("Total number of subnets: {}\n".format(len(data)))
        monitor.write("index, intial latency, latency, memory utilization, energy consumption, GPU temperature, time\n")
        monitor.flush()
        file.write("[\n")

        batch_mIoU, batch_params, batch_flops, batch_latency = [], [], [], []
        batch_util, batch_consumption, batch_temperature = [], [], []
        for i, subnet_all in enumerate(data):
            subnet_str = subnet_all['config']
            print("evaluating subnet {}:".format(i))
            print(subnet_str)

            # set subnet accordingly
            self.supernet.set_active_subnet(**subnet_str)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)
            subnet.cuda()

            
            # compute mean IoU
            mIoU = 0.0
            # util = 0.0
            # consumption = 0.0
            # temperature = 0.0
            mIoU, util, consumption, temperature, time = self.eval_mIoU(subnet, self.input_size[-2:], self.dl, self.sdl, self.num_classes)
            # calculate #params and #flops
            # params = self._calc_params(subnet)
            # flops = self._calc_flops(subnet, dummy_data)

            batch_mIoU.append(mIoU)
            # batch_params.append(params)
            # batch_flops.append(flops)

            # GPU_mem = bytes2human(meminfo.used)

            if report_latency:
                latency = 1.0
                latency, la_util, la_consumption, la_temperature, elapsed_time = self._measure_latency(subnet)
                batch_latency.append(latency)
                util = (util * time + la_util * elapsed_time) / (time + elapsed_time)
                consumption = consumption + la_consumption
                temperature = (temperature * time + la_temperature * elapsed_time) / (time + elapsed_time)

                # print("mIoU = {:.4f}, Params = {:.2f}M , FLOPs = {:.2f}G, FPS = {:d}".format(
                #     mIoU, params, flops, int(1000 / latency)))
            else:
                print("mIoU = {:.4f}, Params = {:.2f}M, FLOPs = {:.2f}".format(mIoU, params, flops))

            batch_util.append(util)
            batch_consumption.append(consumption)
            batch_temperature.append(temperature)

            subnet_all[partial_name + '_latency'] = latency
            subnet_all['GPU_memory_utilization'] = util
            subnet_all['energy_consumption'] = consumption
            subnet_all['GPU_temperature'] = temperature

            json.dump(subnet_all, file, indent=4)
            file.write(",\n")
            file.flush()

            monitor.write(str(i + 1) + " " + str(subnet_all['latency'])+ " " + str(latency) + " " + str(util) + " " + str(consumption) + " " + str(temperature) + " " + str(elapsed_time + time) +"\n")
            monitor.flush()

        file.write("\n]")
        file.close()

        monitor.close()
        return batch_mIoU, batch_params, batch_flops, batch_latency, batch_util, batch_consumption, batch_temperature


class CityscapesEvaluator(OFAFANetEvaluator):
    def __init__(self,
                 supernet: OFAFANet,
                 data_root='../data',  # path to the data folder
                 scale=0.5,  # image size scale, 0.5 -> 512x1024, 0.75 -> 768x1536
                 batchsize=8, n_workers=8):

        input_size = 1, 3, int(scale * 1024), int(scale * 2048)

        # build Cityscapes dataset and dataloader
        from data_providers.cityscapes import CityScapes

        # build the validation dataloader
        dsval = CityScapes(data_root, mode='val')
        dl = DataLoader(dsval, batch_size=batchsize, shuffle=False, num_workers=n_workers, drop_last=False)

        # build a subset train loader for resetting BN running statistics
        cropsize = [1024, 512]
        randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
        ds = CityScapes(data_root, cropsize=cropsize, mode='train', randomscale=randomscale)
        g = torch.Generator()
        g.manual_seed(SUB_SEED)
        rand_indexes = torch.randperm(ds.len, generator=g).tolist()[:500]
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_indexes)
        sdl = DataLoader(ds, batch_size=24, shuffle=False, sampler=sub_sampler,
                         num_workers=10, pin_memory=False, drop_last=True)

        super().__init__(supernet, dl, sdl, input_size, num_classes=19)


class CamVidEvaluator(OFAFANetEvaluator):
    def __init__(self,
                 supernet: OFAFANet,
                 data_root='../data',  # path to the data folder
                 scale=0.75,  # image size scale, 0.75 -> 576x768, 0.9375 -> 720x960, 1.0 -> 768x1024
                 batchsize=8, n_workers=8):

        input_size = 1, 3, int(scale * 768), int(scale * 1024)

        # build CamVid dataset and dataloader
        from data_providers.camvid import CamVidDataset
        from data_providers.transform import LargerEdgeResize, ToTensor, Normalize, Compose, \
            RandomResize, RandomCrop, RandomHorizontalFlip

        # build the validation dataloader
        dsval = partial(CamVidDataset, data_root, 'test')
        val_transforms = Compose([
            LargerEdgeResize(input_size[-2:]), ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dsval = dsval(transforms=val_transforms)

        dl = DataLoader(dsval, batch_size=batchsize, shuffle=False,
                        num_workers=n_workers, pin_memory=True, drop_last=False)

        # build a subset train loader for resetting BN running statistics
        randomscale = (0.5, 2.0)
        cropsize = [576, 576]
        ds = partial(CamVidDataset, data_root, ['train', 'val'])
        train_transforms = Compose([
            RandomResize(scale_range=randomscale),
            RandomCrop(cropsize, pad_if_needed=True, lbl_fill=255),
            RandomHorizontalFlip(), ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ds = ds(transforms=train_transforms)

        g = torch.Generator()
        g.manual_seed(SUB_SEED)
        rand_indexes = torch.randperm(ds.len, generator=g).tolist()[:500]
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_indexes)
        sdl = DataLoader(ds, batch_size=16, shuffle=False, sampler=sub_sampler,
                         num_workers=10, pin_memory=False, drop_last=True)

        super().__init__(supernet, dl, sdl, input_size, num_classes=12)


if __name__ == '__main__':

    # ---------------------------- Basic search space ---------------------------- #
    # construct the basic supernet
    ofa_network = OFAFANetPlus(
        backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
        width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
        output_aux=False, n_classes=12)

    # load checkpoints weights
    # ofa_network.load_state_dict(torch.load(
    #     './pretrained/ofa_fanet/camvid/ofa_plus_basic_camvid_depth-expand-width@phase2/model_maxmIOUFull.pth',
    #     map_location='cpu'))
    # ofa_network.cuda()

    # construct the search space
    import numpy as np
    from search_space import BasicSearchSpace
    search_space = BasicSearchSpace()

    # # ---------------------------- Bottleneck search space ---------------------------- #
    # ofa_network = OFAFANetPlus(
    #     backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
    #     width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
    #     output_aux=False)
    #
    # ofa_network.load_state_dict(torch.load('./pretrained/ofa_fanet/bottleneck_plus/'
    #                                        'ofa_plus_bottleneck_depth-expand-width@phase2/model_maxmIOU50.pth',
    #                                        map_location='cpu'))
    # # ofa_network.cuda()
    #
    # from search.search_space import BottleneckSearchSpace
    # search_space = BottleneckSearchSpace()

    # construct the evaluator
    evaluator = CityscapesEvaluator(ofa_network,
                                    data_root='F:\\EVO\\data\\cityscapes\\leftImg8bit_trainvaltest', scale=0.5)

    # evaluator = CamVidEvaluator(ofa_network,
    #                             data_root='/home/cseadmin/datasets/CamVid/', scale=1.0)

    subnets = search_space.sample(5)

    from itertools import product
    subnets = []
    depth_list = [2, 2, 3, 4, 2] # SUM = 13
    expand_ratio_list = [0.2, 0.25, 0.35]
    width_mult_list = [2, 2, 2, 2, 2] # 第一位不影响结果
    # width_mult_list = [2, 2, 2, 2, 2, 2]

    current_depth = np.zeros(5)
    counter = 0
    for i in range(len(depth_list)):
        while current_depth[i] < depth_list[i]:
            current_depth[i] += 1
            idx_base = int(np.sum(depth_list[0:i]))
            poss = list(product(expand_ratio_list, repeat = depth_list[i]))
            for ele in poss:
                current_ratio = np.zeros(13)    
                current_width = np.zeros(6)
                for j in range(depth_list[i]):
                    current_ratio[j + idx_base] = ele[j]
                for j in range(3):
                    current_width[i + 1] = j
                    current_width[0] = current_width[1]
                    counter += 1
                    subnets.append({"d" : list(current_depth.astype(int)), "e": list(current_ratio), "w": list(current_width.astype(int))})
        else: current_depth[i] = 0
    print("total number of sampled subnets: {}", counter)

    _, batch_params, batch_flops, _ = evaluator.evaluate(subnets, report_latency=True)
    with open("ofa_fanet_plus_rtx_params_flops.json", 'w') as f:
        for i in range(len(subnets)):
            config = subnets[i]
            params = batch_params[i] * 1e6
            flops = batch_flops[i] * 1e9
            f.write("{{\"config\":{}, \"params\":{}, \"flops\":{}}},\n".format(config, params, flops))
            # f.flush()

