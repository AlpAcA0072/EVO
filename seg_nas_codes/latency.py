from __future__ import division

import time
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchprofile import profile_macs

from supernet.ofa_fanet import OFAFANet
from supernet import ofa_fanetplus
from supernet.ofa_fanetplus import OFAFANetPlus

ofa_fanetplus.BENCHMARK_LATENCY = True


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--sample-size', type=int, default=200)
    parse.add_argument('--size_hw', nargs='+', type=int, default=[512, 1024])
    parse.add_argument('--save', type=str, default='meta_data.json', help='file name to save json file')
    parse.add_argument('--backbone', type=str, default='basic', help='which backbone search space to benchmark')
    return parse.parse_args()


def compute_latency(model, input_size, iterations=None, device=None):
    print("use PyTorch for latency test")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    input = torch.randn(*input_size).cuda()

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
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency


def main():
    args = parse_args()
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args.n_classes = 19
    args.use_conv_last = False
    args.output_aux = False
    args.pretrain_model = False

    inputDimension = (1, 3, args.size_hw[0], args.size_hw[1])

    meta_data = []
    data = torch.rand(*inputDimension)
    for _ in range(args.sample_size):

        if args.backbone == 'basic':
            # ----------- Basic block ------------------ #
            ofa_network = OFAFANetPlus(
                backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
                width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]])

        elif args.backbone == 'bottleneck':
            # ----------- Bottleneck block ------------------ #
            ofa_network = OFAFANetPlus(
                backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
                width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]])

        elif args.backbone == 'inverted':
            # ----------- Inverted bottleneck block ------------------ #
            ofa_network = OFAFANetPlus(
                backbone_option='inverted', depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6], width_mult_list=1.0,
                feature_dim_list=[[40], [96], [320]])
        else:
            raise NotImplementedError

        backbone_config = ofa_network.sample_active_subnet()
        subnet = ofa_network.get_active_subnet(preserve_weight=True)

        # print(subnet)
        # calculate # of params
        param_count = sum(p.numel() for p in subnet.parameters() if p.requires_grad)
        # calculate # of flops
        flops_count = profile_macs(subnet, data)
        # measure latency
        latency = compute_latency(subnet, inputDimension)
        # calculate FPS
        fps = 1000. / latency

        print("subnet:")
        print(backbone_config)
        print("#Params = {}, #FLOPs = {}, FPS = {}".format(param_count, flops_count, fps))
        meta_data.append(
            {'config': backbone_config, 'params': param_count, 'flops': flops_count,
             'latency': latency, 'FPS': fps})

    # save meta_data
    json.dump(meta_data, open(args.save, 'w'), indent=4)


if __name__ == '__main__':
    main()