#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import os
import logging
import argparse
from tqdm import tqdm


from models.fanet import FANet
from train.utils import setup_logger
from data_providers.cityscapes import CityScapes


class MscEval(object):
    def __init__(self, size_hw, ignore_label=255):
        self.ignore_label = ignore_label
        # self.scale = scale
        # self.size_hw = [int(1024 * scale), int(2048 * scale)] if size_hw == None else size_hw
        self.size_hw = size_hw

    def __call__(self, net, dl, n_classes):
        # evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            # N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            imgs = imgs.cuda()
            new_hw = self.size_hw
            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
            logits = net(imgs)[0]
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2).view(
                n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()


def evaluate(pretrained='./pretrained', dspth='./data', scale=0.75):
    print('scale', scale)
    # dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval, batch_size=batchsize, shuffle=False, num_workers=n_workers, drop_last=False)

    # ofa_fanet = OFAFANet(
    #     depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0], width_mult_list=[0.65, 0.8, 1.0],
    #     feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]])
    #
    # # ofa_fanet.load_backbone_state_dict(pretrained)
    # state_dict = torch.load(pretrained, map_location='cpu')
    # ofa_fanet.load_state_dict(state_dict)
    #
    # ofa_fanet.set_active_subnet(d=[2, 3, 4, 2], e=[1.0] * 16, w=[2] * 5)
    # net = ofa_fanet.get_active_subnet()

    net = FANet()
    state_dict = torch.load(pretrained, map_location='cpu')
    net.load_state_dict(state_dict)

    net.cuda()
    net.eval()

    with torch.no_grad():
        single_scale = MscEval(scale=scale)
        mIOU = single_scale(net, dl, 19)

    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/cseadmin/huangsh/datasets/Cityscapes')
    parser.add_argument('--pretrained', type=str, default='pretrained/ofaresnet34d.pth')
    parser.add_argument('--scale', type=float, default=0.5, help='scale of input resolution')
    parser.add_argument('--log', type=str, default='.tmp', help='evaluation log directory')
    args = parser.parse_args()

    log_dir = args.log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)

    evaluate(pretrained=args.pretrained, dspth=args.data, scale=args.scale)


