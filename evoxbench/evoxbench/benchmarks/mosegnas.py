import copy
import json
import hashlib
import itertools
import os
import random
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import Callable, Sequence, cast, List, AnyStr, Any, Tuple, Union, Set

from evoxbench.modules import SearchSpace, Evaluator, Benchmark,SurrogateModel
from evoxbench.modules.evaluator import Evaluator
from evoxbench.modules.search_space import SearchSpace

__all__ = ['MoSegNASSearchSpace', 'MoSegNASEvaluator', 'MoSegNASBenchmark']

# HASH = {'conv1x1-compression': 0, 'conv3x3': 1, 'conv1x1-expansion': 2}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "moseg" / name)

class MoSegNASSearchSpace(SearchSpace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #number of MAX layers of each stage:
        # TODO [2, 2, 3, 4, 2] ?
        self.depth_list = [2, 3, 4, 2]

        #choice of the layer except input stem
        self.expand_ratio_list = [0.2, 0.25, 0.35]

        self.categories = [list(range(d + 1)) for d in self.depth_list]
        self.categories += [list(range(3))] * 13
        self.categories += [list(range(3))] * 6
        
    @property
    def name(self):
        return 'MoSegNASSearchSpace'

    def _sample(self, subnet_str=True):
        x = np.array([random.choice(options) for options in self.categories])
        if subnet_str:
            return self._decode(x)
        else:
            return x
    
    def _encode(self, subnet_str):
        e = [np.where(_e == np.array(self.expand_ratio_list))[0][0] for _e in subnet_str['e']]
        return subnet_str['d'] + e + subnet_str['w']
    
    def _decode(self, x):
        e = [self.expand_ratio_list[i] for i in x[4:-5]]
        return {'d': x[:4].tolist(), 'e': e, 'w': x[-5:].tolist()}
    
    def visualize(self):
        raise NotImplementedError

class MoSegNASEvaluator(Evaluator):
    def __init__(self, objs='err&params', **kwargs):
        super().__init__(objs, **kwargs)
        
    @property
    def name(self):
        return 'MoSegNASEvaluator'
    
    def evaluate(self, archs, **kwargs):
        pass

class MoSegNASBenchmark(Benchmark):
    def __init__(self, normalized_objectives=False, **kwargs):
        self.search_space = MoSegNASSearchSpace()
        self.evaluator = MoSegNASEvaluator()
        super().__init__(self.search_space, self.evaluator, normalized_objectives, **kwargs)

    @property
    def name(self):
        return 'MoSegNASBenchmark'

    def debug(self, samples=10):
        archs = self.search_space.sample(samples)
        X = self.search_space.encode(archs)
        F = self.evaluator.evaluate(X, true_eval=True)

        print(archs)
        print(X)
        print(F)
 
class MoSegSurrogateModel(SurrogateModel):
    def __init__(self, **kwargs):
        pass

    def name(self):
        """ name of the surrogate model """
        raise NotImplementedError

    def fit(self, X, **kwargs):
        """ method to fit/learn/train a surrogate model from data """
        raise NotImplementedError

    def predict(self, features, **kwargs):
        """ method to predict performance from architecture features """
        raise NotImplementedError
