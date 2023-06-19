#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import os
import json
import argparse

import torch

from search.msunas import MSuNAS
from supernet.ofa_fanet import OFAFANet
from supernet.ofa_fanetplus import OFAFANetPlus


def main(args):

    # construct search space
    if args.search_space == 'basic':

        from search.search_space import BasicSearchSpace

        # construct the supernet
        supernet = OFAFANetPlus(
            backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=True)
        # load supernet checkpoints weights
        supernet.load_state_dict(torch.load(args.supernet_weights, map_location='cpu'))

        search_space = BasicSearchSpace()

        data = json.load(open(args.meta_data, 'r'))

    elif args.search_space == 'bottleneck':

        from search.search_space import BottleneckSearchSpace

        # construct the supernet
        supernet = OFAFANetPlus(
            backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=True)
        # load supernet checkpoints weights
        supernet.load_state_dict(torch.load(args.supernet_weights, map_location='cpu'))

        search_space = BottleneckSearchSpace()

        data = json.load(open(args.meta_data, 'r'))

    elif args.search_space == 'inverted':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # construct the evaluator
    if args.dataset == 'cityscapes':
        from search.evaluator import CityscapesEvaluator
        evaluator = CityscapesEvaluator(supernet, data_root=args.data_root, scale=args.scale)
    else:
        raise NotImplementedError

    # re-evaluate the mIoU
    subnets = [d['config'] for d in data]
    mIoUs = evaluator.evaluate(subnets)

    for subnet, mIoU in zip(data, mIoUs):
        subnet['mIoU'] = mIoU

    json.dump(data, open(args.meta_data, 'w'), indent=4)

    # # ----------------------- Random Search ----------------------- #
    # subnets = search_space.sample(n_samples=1000)
    # batch_mIoU, batch_params, batch_flops, batch_latency = evaluator.evaluate(subnets, report_latency=True)
    # data = []
    # for subnet, mIoU, params, flops, latency in zip(subnets, batch_mIoU, batch_params, batch_flops, batch_latency):
    #     data.append(
    #         {'config': subnet, 'mIoU': mIoU, 'params': params, 'flops': flops, 'latency': latency})
    #
    # json.dump(data, open(os.path.join(args.save_path, 'RSearch-' + args.dataset + '-'
    #                                   + search_space.name + '-scale@{}.json'.format(args.scale)), 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-space', type=str, default='basic', choices=['basic', 'bottleneck', 'inverted'],
                        help='which search space to run search')
    parser.add_argument('--supernet-weights', type=str,
                        default='pretrained/ofa_fanet/basic/ofa_basic_depth-expand-width@phase2/model_maxmIOU50.pth',
                        help='path to the pretrained supernet weights')
    parser.add_argument('--meta-data', type=str, default='data/ofa_fanet_plus_basic_rtx_fps@0.5.json',
                        help='path to the meta data file of latency, flops, and params')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes'],
                        help='which dataset to perform search')
    parser.add_argument('--data-root', type=str, help='path to the dataset')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='resolution scale 0.5 -> 512x1024 for Cityscapes')
    parser.add_argument('--sec-obj', type=str, default='latency', choices=['latency', 'flops', 'params'],
                        help='the additional objective to be optimized')
    parser.add_argument('--surrogate', type=str, default='lgb',
                        choices=['lgb', 'mlp', 'e2epp', 'carts', 'gp', 'svr', 'ridge', 'knn', 'bayesian', 'ranknet'],
                        help='which surrogate model to use')
    parser.add_argument('--save-path', type=str, default='.tmp',
                        help='path to the folder for saving')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to a previous experiment folder to resume')
    cfgs = parser.parse_args()
    main(cfgs)