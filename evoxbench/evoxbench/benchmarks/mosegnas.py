import os
import random
import json
# import torch
from collections import OrderedDict
from pathlib import Path
import numpy as np

# TODO: 解决import问题，测试subnet结构
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
        # stride_list = [1, 2, 2, 2]
        # number of MAX layers of each stage:
        # [2, 2, 3, 4, 2] ?
        # range of number of each layer estimated
        # [0/2,
        # 0~2,
        # 0~2,
        # 0~2,
        # 0~2
        # ]
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
                #  pretrained,
                 input_size=(1, 3, 512, 1024),
                  **kwargs):
        super().__init__(**kwargs)
        self.input_size= input_size
        # self.pretrained = pretrained

        self.feature_encoder = MoSegNASSearchSpace()
        # self.surrogate_model = MoSegNASSurrogateModel(pretrained=self.pretrained)
        self.surrogate_model = MoSegNASSurrogateModel()
        
    @property
    def name(self):
        return 'MoSegNASEvaluator'
    
    def evaluate(self, 
                 archs, # archs = subnets
                 true_eval = False, # true_eval = if evaluate based on data or true inference result
                 objs='err&params&flops&latency&FPS&mIoU', # objectives to be minimized/maximized
                 **kwargs):
        """ evalute the performance of the given subnets """
        batch_stats = []

        for index, subnet_encoded in enumerate(archs):
            print("evaluating subnet index {}, subnet {}:".format(index, subnet_encoded))

            # results contains err&params&flops at most
            results = self.surrogate_model.predict(subnet = subnet_encoded,
                                                   true_eval = true_eval, 
                                                   objs = objs)
            stats = {}
            # objs='err&params&flops&latency&FPS&mIoU'

            # surrogate models' returns
            if 'err' in objs:
                stats['err'] = 1 - acc
            if 'params' in objs:
                stats['err'] = params
            if 'flops' in objs:
                stats['err'] = flops

            # TODO: inference result, 如果给定的model和json file中已有的model完全一致也可以直接使用json file中已有的数据
            if 'latency' or 'FPS' or 'mIoU' in objs:
                # TODO: model构建
                if 'latency'  in objs:
                    stats['err'] = latency
                if 'FPS' in objs:
                    stats['err'] = 1000.0 / latency
                if 'mIoU' in objs:
                    stats['err'] = mIoU
                batch_stats.append(stats)
            
        return batch_stats

# TODO: model implementation
class MosegNASTempModels():
    def __init__(self) -> None:
        pass
 
class MoSegNASSurrogateModel(SurrogateModel):
    def __init__(self,
                 pretrained_json, 
                #  pretrained_model,
                 **kwargs):
        super().__init__()
        # TODO: 根据已有数据从代理模型返回params和flops指标
        self.pretrained_result = json.load(open(pretrained_json, 'r'))
        searchSpace = MoSegNASSearchSpace()
        self.pretrained_result = searchSpace._encode(self.pretrained_result)

        model = MosegNASTempModels()
        # load pretrained weights
        # self.pretrained_model = open(pretrained_model, 'r')

        # self.search_space = MoSegNASSearchSpace()
        # self.subnet  = self.search_space.encode(self.pretrain_model)


        # params&flops&latency&FPS&mIoU
        # 求和
        # self.params = self.pretrained_model['params']
        # self.flops = self.pretrained_model['flops']

        # 实测
        # self.latency = self.pretrain_model['latency']
        # self.FPS = self.pretrain_model['FPS']
        # self.mIoU = self.pretrain_model['mIoU']

    def name(self):
        return 'MoSegNASSurrogateModel'

    def fit(self, **kwargs):
        """ method to perform forward in a surrogate model from data """
        # TODO: 从表中采params和flops数据

        raise NotImplementedError
    

    # TODO
    def params_predictor(self, subnet):
        # if subnet in self.pretrained_result:
        pass

    def flops_predictor(self, subnet):
        pass

    def acc_predictor(self, subnet):
        pass

    def predict(self, subnet, true_eval, objs, **kwargs):
        """ method to predict performance including acc&params&flops from given architecture features(subnets) """
        pred = {}

        # acc result is not included in the json file
        if 'err' in objs:
            pred['acc'] = self.acc_predictor(subnet = subnet)

        if true_eval:
            if 'params' in objs:
                pred['params'] = self.params_predictor(subnet = subnet)
            if 'flops' in objs:
                pred['flops'] = self.flops_predictor(subnet = subnet)
        else:
            # TODO: 从表中采数据
            pred['params'], pred['flops'] = self.fit(subnet = subnet)
            pass

        return pred


if __name__ == '__main__':
    surrogateModel = MoSegNASSurrogateModel(pretrained_json = 'F:\EVO\data\moseg\ofa_fanet_plus_bottleneck_rtx_fps@0.5.json')
    pass