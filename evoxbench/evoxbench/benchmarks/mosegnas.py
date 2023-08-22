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

        # d e w
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
        # subnet_str只包括config部分不包括params部分
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
                 latency_pretrained,
                 mIoU_pretrained,
                #  input_size=(1, 3, 512, 1024),
                  **kwargs):
        super().__init__(**kwargs)
        # self.input_size= input_size
        self.latency_pretrained = latency_pretrained,
        self.mIoU_pretrained = mIoU_pretrained,


        self.feature_encoder = MoSegNASSearchSpace()
        self.latency_surrogate_model = MoSegNASSurrogateModel(pretrained_weights=self.latency_pretrained)
        self.mIoU_surrogate_model = MoSegNASSurrogateModel(pretrained_weights=self.mIoU_pretrained)
        
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

            # pred contains err&params&flops at most
            pred = self.surrogate_model.predict(subnet = subnet_encoded,
                                                   true_eval = true_eval, 
                                                   objs = objs)
            # objs='err&params&flops&latency&FPS&mIoU'

            if 'err' in objs:
                pred['err'] = 1 - pred['acc']
            if 'FPS' in objs:
                pred['FPS'] = 1000.0 / pred['latency']
            batch_stats.append(pred)
            
        return batch_stats

# TODO: model implementation
class MosegNASRankNet():
    def __init__(self, 
                 pretrained = None,
                 n_layers=2, n_hidden=400,
                 n_output=1, drop=0.2, trn_split=0.8,
                 lr=8e-4, epochs=300, loss='mse'):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.drop = drop
        self.trn_split = trn_split
        self.lr = lr
        self.epochs = epochs
        self.n_feature = None
        self.loss = loss

        self.pretrained = pretrained
        if pretrained is not None:
            self.init_weights()
        else: 
            self.randomly_init_weights()
        self.weights = []
        self.biases = []

        self.name = 'RankNet'


    def init_weights(self, x):
        # 根据x初始化weights和biases
        self.n_feature = x.shape[1]
        #TODO: 根据x初始化weights和biases
        return self
    
    def randomly_init_weights(self, x):
        # 随机初始化weights和bias
        self.n_feature = x.shape[1]

        #Input layer
        self.weights.append([self.fill(self.n_hidden) for _ in range(self.n_feature)])
        self.biases.append(self.fill(self.n_hidden))

        # Hidden layers
        for _ in range(self.n_layers):
            self.weights.append([self.fill(self.n_hidden) for _ in range(self.n_hidden)])
            self.biases.append(self.fill(self.n_hidden))

        # Output layer
        self.weights.append([self.fill(self.n_output) for _ in range(self.n_hidden)])
        self.biases.append(self.fill(self.n_output))

    @staticmethod
    def fill(x):
        return [random.uniform(-1, 1) for _ in range(x)]

    def predict(self, x):
        if x.ndim < 2:
            data = np.zeros((1, x.shape[0]), dtype=np.float32)
            data[0, :] = x
        else:
            data = x.astype(np.float32)
        data = data.T
        pred = self.forward(data)
        return pred[:, 0]
    
    def relu(self, x):
        return max(0, x)

    def dropout(self, x):
        return 0 if random.random() < self.drop else x
    
    def linear(self, inputs, weights, biases):
        return [sum(x * w for x, w in zip(inputs, weights_row)) + b for weights_row, b in zip(weights, biases)]

    def forward(self, x):
        # Input layer
        outputs = self.linear(x, self.weights[0], self.biases[0])
        outputs = [self.relu(x) for x in outputs]
        outputs = [self.dropout(x) for x in outputs]

        # Hidden layers
        for layer in range(1, len(self.weights) - 1):
            outputs = self.linear(outputs, self.weights[layer], self.biases[layer])
            outputs = [self.relu(x) for x in outputs]
            outputs = [self.dropout(x) for x in outputs]

        # Output layer
        outputs = self.linear(outputs, self.weights[-1], self.biases[-1])
        outputs = [self.sigmoid(x) for x in outputs]

        return outputs

    def train():
        pass

class MoSegNASSurrogateModel(SurrogateModel):
    def __init__(self,
                 pretrained, 
                 **kwargs):
        super().__init__()
        # [(depth/layers)1, 3, 0, 1,
        #  (expand ratio/area of the layer)1, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0,
        #  (width mult/channels)2, 2, 2, 0, 0]

        #pretrained中记录了10个model，取均值
        model = json.load(open(pretrained, 'r'))
        searchSpace = MoSegNASSearchSpace()
        model = MosegNASRankNet()
        model.fit()


    def name(self):
        return 'MoSegNASSurrogateModel'

    def fit(self, subnet):
        # subnet = [{'d': [...], 'e': [...], 'w': [...]}]
        # self.pretrained result = [{'config': {'d': [...], 'e': [...], 'w': [...]}, 'params': 2762960, 'flops': 6400445327, 'latency': 4.957451937742715, 'FPS': 201.71652949102148, 'mIoU': 0.6482}, {...}, {...}]
        """ method to perform forward in a surrogate model from data """
        for result in self.pretrained_result:
            if 'config' in result and isinstance(result['config'], dict):
                config = result['config']
                if all(key in config and config[key] == value for key, value in subnet[0].items()):
                    return result['params'], result['flops'], result['latency'], result['mIoU']
        # 不存在时直接返回空值 or ？
        return None

    def params_predictor(self, subnet):
        pass

    def flops_predictor(self, subnet):
        pass

    def acc_predictor(self, subnet):
        pass

    def latency_predictor(self, subnet):
        pass

    def mIoU_predictor(self, subnet):
        pass

    def predict(self, subnet, true_eval, objs, **kwargs):
        """ method to predict performance including acc&params&flops from given architecture features(subnets) """
        pred = {}

        # acc result is not included in the json file
        if 'err' or 'flops' or 'params' in objs:
            # TODO: model构建实测
            pass

        if true_eval:
            # 可用各个子模块求和，不存在则实测
            if 'params' in objs:
                pred['params'] = self.params_predictor(subnet = subnet)

            # MLP预测
            if 'latency' in objs:
                pred['latency'] = self.latency_predictor(subnet = subnet)
            if 'mIoU' in objs:
                pred['mIoU'] = self.mIoU_predictor(subnet = subnet)

            # 实测
            if 'flops' in objs:
                pred['flops'] = self.flops_predictor(subnet = subnet)
            if 'err' in objs:
                pred['acc'] = self.acc_predictor(subnet = subnet)
        else:
            pred['params'], pred['flops'], pred['latency'], pred['mIoU'] = self.fit(subnet = subnet)
        
        return pred