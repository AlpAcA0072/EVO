import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import MscEval
from train.utils import setup_logger
from train.train_ofafanet_basic import SUB_SEED
from supernet.ofa_fanet import OFAFANet
from data_providers.cityscapes import CityScapes

from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics


def main(cfgs):

    # ----------------- dataset loading ------------------- #
    batchsize = 8
    n_workers = 8
    dsval = CityScapes(cfgs.data, mode='val')
    dl = DataLoader(dsval, batch_size=batchsize, shuffle=False, num_workers=n_workers, drop_last=False)

    # build a subset train loader for resetting BN running statistics
    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
    ds = CityScapes(args.data, cropsize=cropsize, mode='train', randomscale=randomscale)
    g = torch.Generator()
    g.manual_seed(SUB_SEED)
    rand_indexes = torch.randperm(ds.len, generator=g).tolist()[:500]
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_indexes)
    sdl = DataLoader(ds, batch_size=24, shuffle=False, sampler=sub_sampler,
                     num_workers=10, pin_memory=False, drop_last=True)

    # -------------------------------------------------- #
    # OFA-FANet with different backbone
    if cfgs.backbone == 'basic':
        ofa_network = OFAFANet(
            backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=True)

    elif cfgs.backbone == 'bottleneck':
        ofa_network = OFAFANet(
            backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=True)

    elif cfgs.backbone == 'inverted':
        ofa_network = OFAFANet(
            backbone_option='inverted', depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6], width_mult_list=1.0,
            feature_dim_list=[[40], [96], [320]], output_aux=True)

    else:
        raise NotImplementedError

    # load checkpoints weights
    ofa_network.load_state_dict(torch.load(cfgs.pretrained, map_location='cpu'))

    # randomly sample a subnet
    subnet_settings = ofa_network.sample_active_subnet()
    print("sampled subnet: ")
    print(subnet_settings)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)

    # reset BN running statistics of the sampled subnet
    subnet.cuda()
    set_running_statistics(subnet, sdl)

    subnet.eval()

    with torch.no_grad():
        single_scale = MscEval(scale=cfgs.scale)
        mIOU = single_scale(subnet, dl, 19)

    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/cseadmin/huangsh/datasets/Cityscapes')
    parser.add_argument('--backbone', dest='backbone', type=str, default='basic',
                        help="which backbone architecture to construct FANet")
    parser.add_argument('--pretrained', type=str,
                        default='pretrained/ofa_fanet/basic/ofa_basic_depth-expand-width@phase2/model_maxmIOU50.pth',
                        help='path to pretrained weights')
    parser.add_argument('--scale', type=float, default=0.5, help='scale of input resolution')
    parser.add_argument('--log', type=str, default='.tmp', help='evaluation log directory')
    args = parser.parse_args()

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)

    main(args)