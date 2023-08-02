import os
import random
import json
# import torch
from collections import OrderedDict
from pathlib import Path
import numpy as np

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, SurrogateModel
# from mosegnas.models import MoSegNASResult  # has to be imported after the init method

__all__ = ['MoSegNASSearchSpace', 'MoSegNASEvaluator', 'MoSegNASBenchmark', 'MoSegNASSurrogateModel']

# HASH = {'conv1x1-compression': 0, 'conv3x3': 1, 'conv1x1-expansion': 2}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "moseg" / name)

class MoSegNASSearchSpace(SearchSpace):
    def __init__(self, subnet_str=True, **kwargs):
        super().__init__(**kwargs)
        self.subnet_str = subnet_str

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

    def _sample(self):
        x = np.array([random.choice(options) for options in self.categories])
        if self.subnet_str:
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
        """ method to visualize an architecture """
        raise NotImplementedError            

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

class MoSegNASEvaluator(Evaluator):
    def __init__(self,
                 pretrained,
                 input_size=(1, 3, 512, 1024),
                  **kwargs):
        super().__init__(**kwargs)
        self.input_size= input_size
        self.pretrained = pretrained

        self.feature_encoder = MoSegNASSearchSpace()
        self.surrogate_model = MoSegNASSurrogateModel(pretrained=self.pretrained)
        
    @property
    def name(self):
        return 'MoSegNASEvaluator'
    
    def evaluate(self, 
                 archs, # archs = subnets
                 true_eval = False, # true_eval = if evaluate based on data or true inference result
                 objs='err&params&flops&latency&FPS&mIoU', # objectives to be minimized/maximized
                 **kwargs):
    
        batch_stats = []

        for index, subnet_encoded in enumerate(archs):
            print("evaluating subnet index {}, subnet {}:".format(index, subnet_encoded))

            accs = self.surrogate_model.fit(subnet_encoded, true_eval = true_eval)
            stats = {}
            # objs='err&params&flops&latency&FPS&mIoU'
            if 'err' in objs:
               stats['err'] = 1 - accs
            if 'params' in objs:
                stats['err'] = params
            if 'flops' in objs:
                stats['err'] = flops
            if 'latency' in objs:
                stats['err'] = latency
            if 'FPS' in objs:
                stats['err'] = FPS
            if 'mIoU' in objs:
                stats['err'] = mIoU
            batch_stats.append(stats)
            
        return batch_stats


 
class MoSegNASSurrogateModel(SurrogateModel):
    def __init__(self, 
                 pretrained, 
                 **kwargs):
       super().__init__()
       self.pretrain_model = json.load(open(pretrained, 'r'))

    def name(self):
        return 'MoSegNASSurrogateModel'

    def fit(self, X, **kwargs):
        """ method to perform forward in a surrogate model from data """
        raise NotImplementedError

    def predict(self, features, true_eval, **kwargs):
        """ method to predict performance including ? from architecture features """
        if not true_eval:
            pred = self.forward(features, np.random.choice(self.models))
        else:
            pred = 0
            for model in self.models:
                pred += self.forward(features, model)

            pred = pred / len(self.models)
        return pred
