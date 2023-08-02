import random
import numpy as np
from abc import ABC, abstractmethod


class OFAFANetSearchSpace(ABC):
    def __init__(self, **kwargs):
        self.feat_enc = None
        # attributes below have to be filled by child class
        self.n_var = None
        self.lb = None
        self.ub = None
        self.categories = None

    @property
    def name(self):
        return NotImplementedError

    @abstractmethod
    def _sample(self, subnet_str=True):
        """ method to randomly create a solution """
        raise NotImplementedError

    def sample(self, n_samples, subnet_str=True):
        subnets = []
        for _ in range(n_samples):
            subnets.append(self._sample(subnet_str=subnet_str))
        return subnets

    @abstractmethod
    def _encode(self, subnet):
        """ method to convert architectural string to search decision variable vector """
        raise NotImplementedError

    def encode(self, subnets):
        X = []
        for subnet in subnets:
            X.append(self._encode(subnet))
        return np.array(X)

    @abstractmethod
    def _decode(self, x):
        """ method to convert decision variable vector to architectural string """
        raise NotImplementedError

    def decode(self, X):
        subnets = []
        for x in X:
            subnets.append(self._decode(x))
        return subnets

    @abstractmethod
    def _features(self, X):
        """ method to convert decision variable vector to feature vector for surrogate model / predictor """
        raise NotImplementedError

    def features(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim < 2:
            X = X.reshape(1, -1)
        return self._features(X)


class BasicSearchSpace(OFAFANetSearchSpace):

    def __init__(self, feature_encoding='one-hot', **kwargs):

        super().__init__(**kwargs)

        self.depth_list = [2, 3, 4, 2]
        self.expand_ratio_list = [0.2, 0.25, 0.35]
        # self.expand_ratio_list = [0.65, 0.8, 1.0]
        self.width_mult_list = [0.65, 0.8, 1.0]
        self.feature_encoding = feature_encoding

        # upper and lower bound on the decision variables
        self.n_var = 25
        self.lb = [0] * self.n_var
        self.ub = self.depth_list + [2] * 21

        # create the categories for each variable
        self.categories = [list(range(d + 1)) for d in self.depth_list]
        self.categories += [list(range(3))] * 16
        self.categories += [list(range(3))] * 5

    @property
    def name(self):
        return 'BasicSearchSpace'

    def _sample(self, subnet_str=True):
        x = np.array([random.choice(options) for options in self.categories])
        if subnet_str:
            return self._decode(x)
        else:
            return x

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'d': [1, 3, 0, 1],
        # 'e': [0.8, 0.65, 0.8, 0.8, 1.0, 0.65, 1.0, 0.65, 0.8, 1.0, 0.8, 0.65, 0.8, 1.0, 1.0, 0.65],
        # 'w': [2, 2, 2, 0, 0]}
        # both 'd' and 'w' indicate choice index already, we just need to encode 'e''
        # self.expand_ratio_list = [0.2, 0.25, 0.35]
        e = [np.where(_e == np.array(self.expand_ratio_list))[0][0] for _e in subnet_str['e']]
        return subnet_str['d'] + e + subnet_str['w']

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(depth)1, 3, 0, 1,
        #  (expand ratio)1, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0,
        #  (width mult)2, 2, 2, 0, 0]
        # both 'd' and 'w' are in subnet string format already, we just need to decode 'e'
        e = [self.expand_ratio_list[i] for i in x[4:-5]]
        return {'d': x[:4].tolist(), 'e': e, 'w': x[-5:].tolist()}

    def _features(self, X):
        # X should be a 2D matrix with each row being a decision variable vector
        if self.feat_enc is None:
            # in case the feature encoder is not initialized
            if self.feature_encoding == 'one-hot':
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(categories=self.categories)
                self.feat_enc = encoder.fit(X)
            else:
                raise NotImplementedError

        return self.feat_enc.transform(X).toarray()


class BottleneckSearchSpace(OFAFANetSearchSpace):

    def __init__(self, feature_encoding='one-hot', **kwargs):

        super().__init__(**kwargs)

        self.depth_list = [0, 1, 2]
        self.expand_ratio_list = [0.2, 0.25, 0.35]
        self.width_mult_list = [0.65, 0.8, 1.0]
        self.feature_encoding = feature_encoding

        # upper and lower bound on the decision variables
        self.n_var = 24
        self.lb = [0] * self.n_var
        self.ub = [1] + [2] * (self.n_var - 1)

        # create the categories for each variable
        self.categories = [list(range(3))] * self.n_var

    @property
    def name(self):
        return 'BottleneckSearchSpace'

    def _sample(self, subnet_str=True):
        x = np.array([random.choice(options) for options in self.categories])
        # x[0] =
        x[-5] = x[-6]  # ad-hoc fix to make sure the first two stem width mult are the same TODO FIXME
        if subnet_str:
            return self._decode(x)
        else:
            return x

    def _encode(self, subnet_str):
        # a sample subnet string
        # {'d': [0, 0, 2, 1, 2],
        #  'e': [0.25, 0.35, 0.2, 0.35, 0.2, 0.25, 0.2, 0.2, 0.35, 0.2, 0.25, 0.25, 0.25],
        #  'w': [2, 1, 1, 2, 2, 2]}
        # both 'd' and 'w' indicate choice index already, we just need to encode 'e'
        e = [np.where(_e == np.array(self.expand_ratio_list))[0][0] for _e in subnet_str['e']]
        return subnet_str['d'] + e + subnet_str['w']

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(depth)1, 3, 0, 1,
        #  (expand ratio)1, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0,
        #  (width mult)2, 2, 2, 0, 0]
        # both 'd' and 'w' are in subnet string format already, we just need to decode 'e'
        e = [self.expand_ratio_list[i] for i in x[4:-6]]
        return {'d': x[:5].tolist(), 'e': e, 'w': x[-6:].tolist()}

    def _features(self, X):
        # X should be a 2D matrix with each row being a decision variable vector
        if self.feat_enc is None:
            # in case the feature encoder is not initialized
            if self.feature_encoding == 'one-hot':
                from sklearn.preprocessing import OneHotEncoder
                self.feat_enc = OneHotEncoder(categories=self.categories).fit(X)
            else:
                raise NotImplementedError

        return self.feat_enc.transform(X).toarray()