#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import argparse


def main(args):

    # construct search space
    if args.search_space == 'basic':

        from search.search_space import BasicSearchSpace

        search_space = BasicSearchSpace()

    elif args.search_space == 'bottleneck':

        from search.search_space import BottleneckSearchSpace

        search_space = BottleneckSearchSpace()

    elif args.search_space == 'inverted':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # ------------------ offline Multi-Objective Surrogate-Assisted Evolutionary Search ----------------- #
    # construct the search engine
    from search.data_driven_msunas import DataDrivenMSuNAS
    engine = DataDrivenMSuNAS(search_space, args.sec_obj, args.surrogate, args.meta_data,
                              save_path=args.save_path)
    # kick-off the search
    engine.search()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-space', type=str, default='basic', choices=['basic', 'bottleneck', 'inverted'],
                        help='which search space to run search')
    parser.add_argument('--meta-data', type=str, default='data/ofa_fanet_plus_basic_rtx_fps@0.5.json',
                        help='path to the meta data file of latency, flops, and params')
    parser.add_argument('--sec-obj', type=str, default='latency', choices=['latency', 'flops', 'params'],
                        help='the additional objective to be optimized')
    parser.add_argument('--surrogate', type=str, default='lgb',
                        choices=['lgb', 'mlp', 'e2epp', 'carts', 'gp', 'svr', 'ridge', 'knn', 'bayesian', 'ranknet'],
                        help='which surrogate model to use')
    parser.add_argument('--save-path', type=str, default='.tmp',
                        help='path to the folder for saving')
    cfgs = parser.parse_args()
    main(cfgs)