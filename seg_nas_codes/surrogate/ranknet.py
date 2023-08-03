#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import copy
import torch
import warnings
import numpy as np
import torch.nn as nn
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import get_correlation

_DEBUG = False

warnings.filterwarnings('ignore')


# class NumbersDataset(Dataset):
#     def __init__(self, data, max_combs=10000000):
#         # assuming data = [x, f] -> N x d+1 matrix
#         # create all combinations of x
#         comb = list(combinations(data, 2))[:max_combs]
#         d1, d2 = np.vstack([v[0]] for v in comb), np.vstack([v[1]] for v in comb)
#         perm = torch.randperm(d1.shape[0])
#         d1 = d1[perm, :]
#         d2 = d2[perm, :]
#         self._target = torch.from_numpy(1. * (d1[:, -1] > d2[:, -1])[:, None]).float()
#         self._x1 = torch.from_numpy(d1[:, :-1]).float()
#         self._x2 = torch.from_numpy(d2[:, :-1]).float()
#         self._size = len(comb)
#
#     def __len__(self):
#         return self._size
#
#     def __getitem__(self, idx):
#         return self._x1[idx], self._x2[idx], self._target[idx]


class NumbersDataset(Dataset):
    def __init__(self, data, target):
        # self._data = torch.from_numpy(data).float()
        self._data = torch.from_numpy(data).float()
        self._target = torch.from_numpy(target).float()
        self._size = len(data)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self._data[idx], self._target[idx]


class Net(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers=2, n_hidden=300, n_output=1, drop=0.2):
        super(Net, self).__init__()

        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.regressor = nn.Linear(n_hidden, n_output)  # output layer
        self.sigmoid = nn.Sigmoid()  # convert output to be [0, 1]
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)  # linear output
        # return self.sigmoid(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


class RankNet:
    """ Ranking based on Multi Layer Perceptron """
    def __init__(self, n_layers=2, n_hidden=400,
                 n_output=1, drop=0.2, device='cpu', trn_split=0.8,
                 lr=8e-4, epochs=300, loss='mse', kwargs=None):

        self.model = None
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.drop = drop
        self.name = 'ranknet'
        self.device = device
        self.trn_split = trn_split
        self.lr = lr
        self.epochs = epochs
        self.loss = loss

        if kwargs is not None: self.kwargs = kwargs
        if _DEBUG: print(self.model)

    def fit(self, x, y, pretrained=None):

        self.model = Net(x.shape[1], self.n_layers, self.n_hidden, self.n_output, self.drop)

        if pretrained:
            self.model.load_state_dict(pretrained)
        else:
            self.model = train(
                self.model, x, y, self.trn_split, self.lr,
                self.epochs, self.device, self.loss)

        return self

    def predict(self, test_data):
        return predict(self.model, test_data, device=self.device)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def train(net, x, y,
          trn_split=0.8, lr=8e-4, epochs=2000, device='cuda', loss='mse'):

    n_samples = x.shape[0]
    perm = torch.randperm(n_samples)
    trn_idx = perm[:int(n_samples * trn_split)]
    vld_idx = perm[int(n_samples * trn_split):]

    # trn_data = NumbersDataset(data=np.hstack((x[trn_idx, :], y[trn_idx, np.newaxis])))
    trn_data = NumbersDataset(data=x[trn_idx, :], target=y[trn_idx, np.newaxis])

    trn_loader = torch.utils.data.DataLoader(
        trn_data, batch_size=2000, shuffle=True, pin_memory=True, num_workers=2)

    # inputs = torch.from_numpy(x).float()
    vld_inputs = torch.from_numpy(x).float()[vld_idx, :]
    try:
        targets = torch.zeros(n_samples, 1)
        targets[:, 0] = torch.from_numpy(y).float()
    except RuntimeError:
        targets = torch.from_numpy(y).float()
    vld_targets = targets[vld_idx]

    # # ------------- for debug purpose ------------- #
    # trn_data = NumbersDataset(data=np.hstack((x, y)))
    # trn_loader = torch.utils.data.DataLoader(
    #     trn_data, batch_size=2000, shuffle=True, pin_memory=True, num_workers=2)
    # vld_data = NumbersDataset(data=np.hstack((kwargs['tst_data'], kwargs['tst_target'])))
    # # vld_loader = torch.utils.data.DataLoader(
    # #     vld_data, batch_size=2000, shuffle=False, pin_memory=True, num_workers=2)
    # vld_inputs = torch.from_numpy(kwargs['tst_data']).float()
    # vld_targets = torch.from_numpy(kwargs['tst_target']).float()
    # # --------------------------------------------- #

    net.apply(net.init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # if loss == 'bce':
    #     criterion = nn.BCELoss()
    # elif loss == 'margin':
    #     criterion = nn.MarginRankingLoss(margin=0.05)
    # else:
    #     raise NotImplementedError

    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'l1':
        criterion = nn.SmoothL1Loss()
    elif loss == 'rank':
        criterion = nn.MarginRankingLoss(margin=0.05)
    elif loss == 'mse+rank':
        criterion = {'mse': nn.MSELoss(), 'rank': nn.MarginRankingLoss(margin=0.05)}
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=0)

    best_tau = 0
    best_net = copy.deepcopy(net)
    for epoch in range(epochs):
        # trn_inputs = inputs[trn_idx]
        # trn_labels = targets[trn_idx]
        loss_trn = train_one_epoch(net, trn_loader, criterion, optimizer, device)
        # loss_vld = infer(net, vld_loader, criterion, device)
        rmse, rho, tau, _, _ = validate(net, vld_inputs, vld_targets, device=device)
        scheduler.step()

        if _DEBUG:
            print("epoch = {}, rmse = {}, rho = {}, tau = {}".format(epoch, rmse, rho, tau))

        if tau > best_tau:
            best_tau = tau
            best_net = copy.deepcopy(net)

    # validate(best_net, inputs[vld_idx, :], targets[vld_idx, :], device=device)
    validate(best_net, vld_inputs, vld_targets, device=device)

    return best_net.to('cpu')


def train_one_epoch(net, trn_loader, criterion, optimizer, device):
    net.train()
    train_loss = 0
    total = 0
    dynanmic_batch_size = 4

    for data, target in trn_loader:

        # data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if isinstance(criterion, nn.MSELoss) or isinstance(criterion, nn.SmoothL1Loss):
            pred = net(data)
            loss = criterion(pred, target)
        elif isinstance(criterion, nn.MarginRankingLoss):
            loss = 0
            for _ in range(dynanmic_batch_size):
                idx = torch.randperm(len(data))
                rank_target = 1. * (target > target[idx])
                rank_target[rank_target < 1] = -1.
                loss += criterion(net(data), net(data[idx]), rank_target)
        elif isinstance(criterion, dict):
            pred = net(data)
            mse = criterion['mse'](pred, target)
            idx = torch.randperm(len(data))
            rank_target = 1. * (target > target[idx])
            rank_target[rank_target < 1] = -1.
            rank = criterion['rank'](pred, net(data[idx]), rank_target)
            loss = mse + rank
        else:
            raise NotImplementedError

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += target.size(0)

    # if _DEBUG:
    #     print('train loss = {:.2E}'.format(train_loss / total))

    return train_loss/total


def infer(net, vld_loader, criterion, device):
    net.eval()
    vld_loss = 0
    total = 0

    with torch.no_grad():
        for data1, data2, target in vld_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            pred = net(data1, data2)
            loss = criterion(pred, target)

            vld_loss += loss.item()
            total += target.size(0)

    if _DEBUG:
        print('valid loss = {:.4E}'.format(vld_loss / total))

    return vld_loss/total


def validate(net, data, target, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()
        # print(np.mean(pred), np.std(pred))
        # print(np.mean(target), np.std(target))
        rmse, _, rho, tau = get_correlation(pred, target)

    return rmse, rho, tau, pred, target


def predict(net, query, device):

    if query.ndim < 2:
        data = torch.zeros(1, query.shape[0])
        data[0, :] = torch.from_numpy(query).float()
    else:
        data = torch.from_numpy(query).float()

    net = net.to(device)
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = net(data)

    return pred.cpu().detach().numpy()[:, 0]


if __name__ == '__main__':
    import json
    # from seg_nas_codes.search.search_space import BasicSearchSpace
    sys.path.append('F:\\EVO')
    from seg_nas_codes.search.search_space import BasicSearchSpace
    from lightgbm import LGBMRegressor

    # define the search space
    search_space = BasicSearchSpace()

    # meta_data = json.load(open("../data/ofa_fanet_plus_basic_rtx_fps@0.5.json", "r"))
    meta_data = json.load(open("f:/EVO/seg_nas_codes/data/ofa_fanet_plus_bottleneck_rtx_fps@0.5.json", "r"))
    subnet_str = [d['config'] for d in meta_data]
    features = search_space.features(search_space.encode(subnet_str))
    # targets = np.array([d['mIoU'] for d in meta_data])
    targets = np.array([d['latency'] for d in meta_data])

    train_inputs = features[:10000, :]
    test_inputs = features[10000:, :]
    train_targets = targets[:10000]
    test_targets = targets[10000:]

    perm = np.random.permutation(len(train_targets))

    test_preds, lgb_test_preds = 0, 0
    # state_dicts = []
    # state_dicts = torch.load('../surrogate/ranknet_mIoU.pth', map_location='cpu')
    # state_dicts = torch.load('../surrogate/ranknet_latency.pth', map_location='cpu')

    for i, test_split in enumerate(np.array_split(perm, 10)):

        train_split = np.setdiff1d(perm, test_split, assume_unique=True)

        # ranknet
        predictor = RankNet(loss='rank', epochs=300, device='cpu')

        # no pretrained model
        # predictor.fit(train_inputs[train_split, :], train_targets[train_split], pretrained=state_dicts[i])
        predictor.fit(train_inputs[train_split, :], train_targets[train_split])
        
        pred = predictor.predict(train_inputs[test_split, :])
        rmse, r, rho, tau = get_correlation(pred, train_targets[test_split])
        print("Fold {} RankNet: rmse = {:.4f}, pearson = {:.4f}, spearman = {:.4f}, kendall = {:.4f}".format(
            i, rmse, r, rho, tau))

        # state_dicts.append(predictor.model.state_dict())

        pred = predictor.predict(test_inputs)
        test_preds += pred

        # LGB
        lgb = LGBMRegressor(objective='huber').fit(train_inputs[train_split, :], train_targets[train_split])
        pred_lgb = lgb.predict(train_inputs[test_split, :])
        rmse, r, rho, tau = get_correlation(pred_lgb, train_targets[test_split])
        print("Fold {} LGB: rmse = {:.4f}, pearson = {:.4f}, spearman = {:.4f}, kendall = {:.4f}".format(
            i, rmse, r, rho, tau))

        pred_lgb = lgb.predict(test_inputs)
        lgb_test_preds += pred_lgb

    avg_pred = test_preds / 10
    lgb_avg_pred = lgb_test_preds /10

    rmse, r, rho, tau = get_correlation(avg_pred, test_targets)
    print("ensemble RankNet: rmse = {:.4f}, pearson = {:.4f}, spearman = {:.4f}, kendall = {:.4f}".format(
        rmse, r, rho, tau))

    rmse, r, rho, tau = get_correlation(lgb_avg_pred, test_targets)
    print("ensemble LGB: rmse = {:.4f}, pearson = {:.4f}, spearman = {:.4f}, kendall = {:.4f}".format(
        rmse, r, rho, tau))

    # save
    # torch.save(state_dicts, "ranknet_latency.pth")