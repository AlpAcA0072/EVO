from __future__ import division
import argparse
import os
import sys
import logging
import torch
import numpy as np

sys.path.append("../")

try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")

from models.FANet_trt import FANet


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='stdc')
    parse.add_argument('--backbone', type=str, default='STDCNet813')
    parse.add_argument('--inputratio', type=int, default=50)
    return parse.parse_args()


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
    
    methodName = '{}-{}'.format(args.model, args.backbone)
    inputDimension = (1, 3, 512, 1024) if args.inputratio == 50 else (1, 3, 768, 1536)
    model = FANet(args)
    model = model.cuda()
    #####################################################

    latency = compute_latency(model, inputDimension)
    print("{}{} FPS:".format(methodName, args.inputScale) + str(1000./latency))
    logging.info("{}{} FPS:".format(methodName, args.inputScale) + str(1000./latency))

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''


if __name__ == '__main__':
    main() 
