import os
import json
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation

from surrogate import SurrogateModel
from search.search_space import OFAFANetSearchSpace
from search.utils import setup_logger

__all__ = ['DataDrivenMSuNAS']


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 search_space: OFAFANetSearchSpace,
                 mIoU_predictor: SurrogateModel,
                 sec_obj_predictor: SurrogateModel,
                 ):
        super().__init__(
            n_var=search_space.n_var, n_obj=2, n_constr=0, xl=search_space.lb, xu=search_space.ub, type_var=np.int)

        self.search_space = search_space
        self.mIoU_predictor = mIoU_predictor
        self.sec_obj_predictor = sec_obj_predictor

    def _evaluate(self, x, out, *args, **kwargs):

        features = self.search_space.features(x)
        mIoUs = SurrogateModel.predict(self.mIoU_predictor, features)
        sec_objs = SurrogateModel.predict(self.sec_obj_predictor, features)

        # assuming minimization, we need to negate mIoU
        out["F"] = np.column_stack((-mIoUs, sec_objs))


class DataDrivenMSuNAS:
    """ Multi-Objective Evolutionary Search """
    def __init__(self,
                 search_space: OFAFANetSearchSpace,
                 sec_obj='latency',  # additional objectives to be optimized, choices=['latency', 'flops', 'params']
                 surrogate='lgb',  # surrogate model method
                 meta_data='data/ofa_fanet_plus_basic_rtx_fps@0.5.json',  # path to the pre-collect json data file
                 pop_size=100,  # population size
                 max_iter=30,  # maximum number of iterations to search
                 save_path='.tmp',  # path to the folder for saving stats
                 num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
                 ):

        self.search_space = search_space
        self.sec_obj = sec_obj
        self.surrogate = surrogate
        self.meta_data = meta_data
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_subnets_to_report = num_subnets

        # create the save dir and setup logger
        self.save_path = os.path.join(save_path, search_space.name + "-mIoU&{}-{}-pop_size@{}-max_iter@{}".format(
            sec_obj, surrogate, pop_size, max_iter))

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

        # build offline surrogate models for all objectives, i.e., 'mIoU' + 'latency' or 'flops' or'params'
        self.logger.info("Constructing one-time off-line surrogate model for predicting mIoU and {}".format(sec_obj))

        meta_data = json.load(open(meta_data, "r"))
        subnet_str = [d['config'] for d in meta_data]
        features = search_space.features(search_space.encode(subnet_str))
        mIoU_targets = np.array([d['mIoU'] for d in meta_data])
        sec_obj_targets = np.array([d[sec_obj] for d in meta_data])

        # calculate normalization factors
        self.mIoU_range = (np.min(mIoU_targets), np.max(mIoU_targets))
        self.sec_obj_range = (np.min(sec_obj_targets), np.max(sec_obj_targets))

        if surrogate == 'ranknet':
            mIoU_state_dicts = torch.load('./surrogate/ranknet_mIoU.pth', map_location='cpu')
            self.mIoU_predictor = SurrogateModel(surrogate).fit(
                features, mIoU_targets, pretrained=mIoU_state_dicts, ensemble=True)

            sec_obj_state_dicts = torch.load('./surrogate/ranknet_{}.pth'.format(sec_obj), map_location='cpu')
            self.sec_obj_predictor = SurrogateModel(surrogate).fit(
                features, sec_obj_targets, pretrained=sec_obj_state_dicts, ensemble=True)
        else:
            self.mIoU_predictor = SurrogateModel(surrogate).fit(features, mIoU_targets, ensemble=True)
            self.sec_obj_predictor = SurrogateModel(surrogate).fit(features, sec_obj_targets, ensemble=True)

    def search(self):

        # define the auxiliary problem of surrogate models
        problem = AuxiliarySingleLevelProblem(self.search_space, self.mIoU_predictor, self.sec_obj_predictor)

        # this problem is a regular discrete-variable multi-objective problem
        # which can be exhaustively searched by regular EMO algorithms such as NSGA-II, MOEA/D, etc.
        emo_method = get_algorithm(
            "nsga2", pop_size=200, sampling=get_sampling('int_lhs'),
            crossover=get_crossover('int_two_point', prob=0.9),
            mutation=get_mutation('int_pm', eta=1.0),
            eliminate_duplicates=True)
        res = minimize(problem, emo_method, termination=('n_gen', 500), verbose=True)

        if self.surrogate == 'ranknet':
            # normalize the surrogate predicted objectives
            mIoU_scaler = MinMaxScaler(feature_range=self.mIoU_range)
            mIoU_scaler.fit(-res.F[:, 0].reshape(-1, 1))

            sec_obj_scaler = MinMaxScaler(feature_range=self.sec_obj_range)
            sec_obj_scaler.fit(res.F[:, 1].reshape(-1, 1))

            F = np.column_stack((1000 / sec_obj_scaler.transform(res.F[:, 1].reshape(-1, 1))[:, 0],
                                 mIoU_scaler.transform(-res.F[:, 0].reshape(-1, 1))[:, 0]))
        else:
            F = np.column_stack((1000 / res.F[:, 1], -res.F[:, 0]))

        # find model around [180 - 190] FPS and [73.2 - 74] mIoU
        valid_idx = np.logical_and(F[:, 0] > 200, F[:, 1] > 0.71)
        print(np.sum(valid_idx))

        subnets = self.search_space.decode(res.X[valid_idx, :])
        print(subnets)

        data = []
        for subnet, mIoU, FPS in zip(subnets, F[valid_idx, 1], F[valid_idx, 0]):
            data.append({'config': subnet, 'mIoU': mIoU, 'FPS': FPS})

        json.dump(data, open('tmp.json', 'w'), indent=4)

        from pymoo.visualization.scatter import Scatter
        plot = Scatter()
        plot.add(F, s=50, facecolors='none', edgecolors='tab:blue')
        plot.add(F[valid_idx, :], s=30, facecolors='tab:red', edgecolors='tab:red')
        plot.show()

