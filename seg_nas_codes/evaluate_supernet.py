#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import json
import argparse
import numpy as np

import torch

from supernet.ofa_fanetplus import OFAFANetPlus


def main(args):

    if args.dataset == 'cityscapes':
        num_classes = 19
    elif args.dataset == 'camvid':
        num_classes = 12
    else:
        raise NotImplementedError

    # construct search space
    if args.search_space == 'basic':

        from search.search_space import BasicSearchSpace

        # construct the supernet
        supernet = OFAFANetPlus(
            backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=False, n_classes=num_classes)
        # load supernet checkpoints weights
        supernet.load_state_dict(torch.load(args.supernet_weights, map_location='cpu'))

        search_space = BasicSearchSpace()

    elif args.search_space == 'bottleneck':

        from search.search_space import BottleneckSearchSpace

        # construct the supernet
        supernet = OFAFANetPlus(
            backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=False, n_classes=num_classes)
        # load supernet checkpoints weights
        supernet.load_state_dict(torch.load(args.supernet_weights, map_location='cpu'))

        search_space = BottleneckSearchSpace()

    elif args.search_space == 'inverted':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # construct the evaluator
    if args.dataset == 'cityscapes':
        from search.evaluator import CityscapesEvaluator
        evaluator = CityscapesEvaluator(supernet, data_root=args.data_root, scale=args.scale)

    elif args.dataset == 'camvid':
        from search.evaluator import CamVidEvaluator
        evaluator = CamVidEvaluator(supernet, data_root=args.data_root, scale=args.scale)

    else:
        raise NotImplementedError

    data = {
        '<100': [],

    }


    # re-evaluate the mIoU
    data = json.load(open("tmp.json", 'r'))
    idx = np.argsort([d['mIoU'] for d in data])
    data = [data[i] for i in idx[::-1]]

    subnets = [d['config'] for d in data]
    batch_mIoU, _, _, batch_latency = evaluator.evaluate(subnets, report_latency=True)

    for idx, (pred_mIoU, pred_FPS, mIoU, latency) in enumerate(zip(
            [d['mIoU'] for d in data], [d['FPS'] for d in data], batch_mIoU, batch_latency)):
        print("Subnet {}: predicted mIoU = {:.4f}, FPS = {:.0f}; evaluated mIoU = {:.4f}, FPS = {:.0f}".format(
            idx, pred_mIoU, pred_FPS, mIoU, 1000 / latency))

    # data = []
    # exp_root = "./SearchExps/" \
    #            "BasicSearchSpaceMSuEvo-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30-scale@720x960/"
    # for i in range(1, 35):
    #     data.append(json.load(open(os.path.join(
    #         exp_root, 'non_dominated_subnets', 'subnet_{}.json'.format(i)), 'r')))

    # idx = np.argsort([d[2] for d in data])
    # data = [data[i] for i in idx[::-1]]

    # subnets = [d[0] for d in data]
    # batch_mIoU, _, _, batch_latency = evaluator.evaluate(subnets, report_latency=True)
    #
    # for idx, (pred_mIoU, pred_FPS, mIoU, latency) in enumerate(zip(
    #         [d[1] for d in data], [1000 / d[2] for d in data], batch_mIoU, batch_latency)):
    #     print("Subnet {}: predicted mIoU = {:.4f}, FPS = {:.0f}; evaluated mIoU = {:.4f}, FPS = {:.0f}".format(
    #         idx, pred_mIoU, pred_FPS, mIoU, 1000 / latency))

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
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'camvid'],
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