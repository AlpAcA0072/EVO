import os
import json
import logging
import numpy as np

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation

from surrogate import SurrogateModel
from surrogate.utils import get_correlation
from search.evaluator import OFAFANetEvaluator
from search.search_space import OFAFANetSearchSpace
from search.utils import MySampling, BinaryCrossover, MyMutation, calc_hv, setup_logger, HighTradeoffPoints


__all__ = ['MSuNAS']


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


class SubsetSelectionProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=np.bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # s, p = stats.kstest(np.concatenate((self.archive, self.candidates[_x])), 'uniform')
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # f[i, 0] = s
            # we penalize if the number of selected candidates is not exactly K
            # g[i, 0] = (self.n_max - np.sum(_x)) ** 2
            g[i, 0] = np.sum(_x) - self.n_max  # as long as the selected individual is less than K

        out["F"] = f
        out["G"] = g


class MSuNAS:
    def __init__(self,
                 search_space: OFAFANetSearchSpace,
                 evaluator: OFAFANetEvaluator,
                 sec_obj='latency',  # additional objectives to be optimized, choices=['latency', 'flops', 'params']
                 surrogate='lgb',  # surrogate model method
                 # meta_data='data/ofa_fanet_plus_basic_rtx_fps@0.5.json',  # path to the pre-collect json data file
                 n_doe=100,  # design of experiment points, i.e., number of initial (usually randomly sampled) points
                 n_iter=8,  # number of high-fidelity evaluations per iteration
                 max_iter=30,  # maximum number of iterations to search
                 save_path='.tmp',   # path to the folder for saving stats
                 num_subnets=4,  # number of subnets spanning the Pareto front that you would like find
                 resume=None,  # path to a search experiment folder to resume search
                 ):

        self.search_space = search_space
        self.evaluator = evaluator
        self.sec_obj = sec_obj
        self.surrogate = surrogate
        # self.meta_data = meta_data  # all surrogate models are learned online
        self.n_doe = n_doe
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.num_subnets_to_report = num_subnets
        self.resume = resume
        self.ref_pt = None

        # create the save dir and setup logger
        self.save_path = os.path.join(save_path, search_space.name
                                      + "MSuEvo-mIoU&{}-{}-n_doe@{}-n_iter@{}-max_iter@{}-scale@{}x{}".format(
            sec_obj, surrogate, n_doe, n_iter, max_iter, evaluator.input_size[-2], evaluator.input_size[-1]))

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = logging.getLogger()
        setup_logger(self.save_path)

        # # build offline surrogate models for the second objective, i.e., 'latency', 'flops', 'params'
        # self.logger.info("Constructing one-time off-line surrogate model for predicting {}".format(sec_obj))
        # meta_data = json.load(open(meta_data, "r"))
        # subnet_str = [d['config'] for d in meta_data]
        # features = search_space.features(search_space.encode(subnet_str))
        # targets = np.array([d[sec_obj] for d in meta_data])
        # self.sec_obj_predictor = SurrogateModel(surrogate).fit(features, targets, ensemble=True)

    def _eval(self, subnets):

        batch_mIoU, batch_params, batch_flops, batch_latency = self.evaluator.evaluate(subnets, report_latency=True)

        if self.sec_obj == 'latency':
            # if len(batch_latency) < 1:  # in case evaluator does not return latency
            #     features = self.search_space.features(self.search_space.encode(subnets))
            #     batch_latency = SurrogateModel.predict(self.sec_obj_predictor, features)
            return batch_mIoU, batch_latency

        elif self.sec_obj == 'flops':
            return batch_mIoU, batch_flops

        elif self.sec_obj == 'params':
            return batch_mIoU, batch_params

        else:
            raise NotImplementedError

    def search(self):
        # ----------------------- initialization ----------------------- #
        if self.resume:
            archive = json.load(open(self.resume, 'r'))
        else:
            archive = []  # initialize an empty archive to store all trained CNNs
            doe_subnets = self.search_space.sample(self.n_doe - 2)
            # add the lower and upper bound subnets
            doe_subnets.extend(self.search_space.decode([np.array(self.search_space.lb),
                                                         np.array(self.search_space.ub)]))
            mIoUs, sec_objs = self._eval(doe_subnets)
            # store evaluated / trained architectures
            for member in zip(doe_subnets, mIoUs, sec_objs):
                archive.append(member)

        # setup reference point for calculating hypervolume
        self.ref_pt = np.array([np.max([-d[1] for d in archive]), np.max([d[2] for d in archive])])

        self.logger.info("Iter 0: hv = {:.4f}".format(self._calc_hv(archive, self.ref_pt)))
        self.save_iteration("iter_0", archive)  # dump the initial population

        # ----------------------- main search loop ----------------------- #
        for it in range(1, self.max_iter + 1):
            # construct mIoU surrogate model from archive
            mIoU_predictor, sec_obj_predictor = self._fit_predictors(archive)

            # construct an auxiliary problem of surrogate objectives and
            # search for the next set of candidates for high-fidelity evaluation
            candidates = self._next(archive, mIoU_predictor, sec_obj_predictor)

            # high-fidelity evaluate the selected candidates (lower level)
            mIoUs, sec_objs = self._eval(candidates)

            # evaluate the performance of mIoU predictor
            mIoU_pred = SurrogateModel.predict(
                mIoU_predictor, self.search_space.features(self.search_space.encode(candidates)))
            mIoU_rmse, mIoU_r, mIoU_rho, mIoU_tau = get_correlation(mIoU_pred, mIoUs)

            # evaluate the performance of sec obj predictor
            sec_obj_pred = SurrogateModel.predict(
                sec_obj_predictor, self.search_space.features(self.search_space.encode(candidates)))
            sec_obj_rmse, sec_obj_r, sec_obj_rho, sec_obj_tau = get_correlation(sec_obj_pred, sec_objs)

            # add the evaluated subnets to archive
            for member in zip(candidates, mIoUs, sec_objs):
                archive.append(member)

            # print iteration-wise statistics
            hv = self._calc_hv(archive, self.ref_pt)
            self.logger.info("Iter {}: hv = {:.4f}".format(it, hv))
            self.logger.info("Surrogate model {} performance:".format(self.surrogate))
            self.logger.info("For predicting mIoU: RMSE = {:.4f}, Spearman's Rho = {:.4f}, "
                             "Kendall’s Tau = {:.4f}".format(mIoU_rmse, mIoU_rho, mIoU_tau))
            self.logger.info("For predicting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, "
                             "Kendall’s Tau = {:.4f}".format(self.sec_obj, sec_obj_rmse, sec_obj_rho, sec_obj_tau))

            meta_stats = {'iteration': it, 'hv': hv, 'mIoU_tau': mIoU_tau, 'sec_obj_tau': sec_obj_tau}
            self.save_iteration("iter_{}".format(it), archive, meta_stats)  # dump the current iteration

        # ----------------------- report search result ----------------------- #
        # dump non-dominated architectures from the archive first
        nd_front = NonDominatedSorting().do(np.column_stack(
            ([-x[1] for x in archive], [x[2] for x in archive])), only_non_dominated_front=True)
        nd_archive = [archive[idx] for idx in nd_front]
        self.save_subnets("non_dominated_subnets", nd_archive)

        # select a subset from non-dominated set in case further fine-tuning
        selected = HighTradeoffPoints(n_survive=self.num_subnets_to_report).do(
            np.column_stack(([-x[1] for x in nd_archive], [x[2] for x in nd_archive])))

        self.save_subnets("high_tradeoff_subnets", [nd_archive[i] for i in selected])

    @staticmethod
    def _calc_hv(archive, ref_pt):
        # reference point (nadir point) for calculating hypervolume
        # ref_pt = np.array([-np.min([x[1] for x in archive]), np.max([x[2] for x in archive])])
        hv = calc_hv(ref_pt, np.column_stack(([-x[1] for x in archive], [x[2] for x in archive])))
        return hv

    def _fit_predictors(self, archive):
        self.logger.info("fitting mIoU surrogate model...")
        features = self.search_space.features(self.search_space.encode([x[0] for x in archive]))
        mIoU_targets = np.array([x[1] for x in archive])
        mIoU_predictor = SurrogateModel(self.surrogate).fit(features, mIoU_targets, ensemble=True)

        self.logger.info("fitting second objective surrogate model...")
        sec_obj_targets = np.array([x[2] for x in archive])
        sec_obj_predictor = SurrogateModel(self.surrogate).fit(features, sec_obj_targets, ensemble=True)

        return mIoU_predictor, sec_obj_predictor

    @staticmethod
    def subset_selection(pop, archive, K):
        # get non-dominated subnets from archive
        F = np.column_stack(([-x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # select based on the cheap (second) objective
        subset_problem = SubsetSelectionProblem(pop.get("F")[:, 1], F[front, 1], K)
        # define a solver
        ea_method = get_algorithm(
            'ga', pop_size=500, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)
        # start solving
        res = minimize(subset_problem, ea_method, termination=('n_gen', 200), verbose=False)
        # in case the number of solutions selected is less than K
        if np.sum(res.X) < K:
            for idx in np.argsort(pop.get("F")[:, 0]):
                res.X[idx] = True
                if np.sum(res.X) >= K:
                    break
        return res.X

    def _next(self, archive, mIoU_predictor, sec_obj_predictor):
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self.search_space, mIoU_predictor, sec_obj_predictor)

        # this problem is a regular discrete-variable multi-objective problem
        # which can be exhaustively searched by regular EMO algorithms such as NSGA-II, MOEA/D, etc.
        emo_method = get_algorithm(
            "nsga2", pop_size=200, sampling=get_sampling('int_lhs'),
            crossover=get_crossover('int_two_point', prob=0.9),
            mutation=get_mutation('int_pm', eta=1.0),
            eliminate_duplicates=True)
        res = minimize(problem, emo_method, termination=('n_gen', 500), verbose=False)

        # check against archive to eliminate any already evaluated subnets to be re-evaluated
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in self.search_space.decode(res.pop.get("X"))])

        # form a subset selection problem to short list K from pop_size
        indices = self.subset_selection(res.pop[not_duplicate], archive, self.n_iter)

        candidates = self.search_space.decode(res.pop[not_duplicate][indices].get("X"))

        return candidates

    def save_iteration(self, _save_dir, archive, meta_stats=None):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        json.dump(archive, open(os.path.join(save_dir, 'archive.json'), 'w'), indent=4)
        if meta_stats:
            json.dump(meta_stats, open(os.path.join(save_dir, 'stats.json'), 'w'), indent=4)

    def save_subnets(self, _save_dir, archive):
        save_dir = os.path.join(self.save_path, _save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for i, subnet in enumerate(archive):
            json.dump(subnet, open(os.path.join(save_dir, 'subnet_{}.json'.format(i + 1)), 'w'), indent=4)
