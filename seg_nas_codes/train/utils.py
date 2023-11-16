import time
import logging
import numpy as np
from tqdm import tqdm
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    _format = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=_format, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr


def enet_weighing(label, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0

    label = label.cpu().numpy()

    # Flatten label
    flat_label = label.flatten()

    # Sum up the number of pixels of each class and the total pixel
    # counts for each label
    class_count += np.bincount(flat_label, minlength=num_classes)
    total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    class_weights = torch.from_numpy(class_weights).float()
    # print(class_weights)
    return class_weights


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class OhemCELoss(nn.Module):
    # def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
    def __init__(self, thresh, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        # self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels, n_min):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)


class WeightedOhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, num_classes, ignore_lb=255, *args, **kwargs):
        super(WeightedOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.num_classes = num_classes
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(),
                                       ignore_index=self.ignore_lb, reduction='none')
        loss = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(
            False).type(torch.cuda.FloatTensor)
        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(1, 3, 1, 1).type(
                torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(boundary_logits, boundary_targets.shape[2:], mode='bilinear',
                                            align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class MscEvalV0(object):
    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        # evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            imgs = imgs.cuda()
            N, C, H, W = imgs.size()
            new_hw = [int(H * self.scale), int(W * self.scale)]
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


class Optimizer(object):
    def __init__(self, model, loss=None, opti='sgd', lr0=0.001, momentum=0.9, wd=0.005, warmup_steps=1000, warmup_start_lr=0.0001,
                 max_iter=80000, power=0.98, *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        loss_nowd_params = loss.get_params() if loss is not None else []
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True},
                {'params': loss_nowd_params}, ]

        for name, param in model.named_parameters():
            if 'embed' in name:
                param_list.append({"params": param, 'weight_decay': 0}, )

        if opti == 'sgd':
            self.optim = torch.optim.SGD(param_list, lr=lr0, momentum=momentum, weight_decay=wd)
        elif opti == 'adam':
            self.optim = torch.optim.Adam(param_list, lr=lr0, betas=(0.5, 0.999))
        else:
            raise NotImplementedError

        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            print('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()