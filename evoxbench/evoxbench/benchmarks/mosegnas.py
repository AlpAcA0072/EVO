import copy
import json
import hashlib
import itertools
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import Callable, Sequence, cast, List, AnyStr, Any, Tuple, Union, Set

from evoxbench.modules import SearchSpace, Evaluator, Benchmark
from evoxbench.modules.evaluator import Evaluator
from evoxbench.modules.search_space import SearchSpace

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "moseg" / name)

class MoSegNASSearchSpace(SearchSpace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return 'MoSegNASSearchSpace'
    
class MoSegNASEvaluator(Evaluator):
    def __init__(self, objs='err&params', **kwargs):
        super().__init__(objs, **kwargs)
        
    @property
    def name(self):
        return 'MoSegNASEvaluator'

class MoSegNASBenchmark(Benchmark):
    def __init__(self, search_space: SearchSpace, evaluator: Evaluator, normalized_objectives=False, **kwargs):
        super().__init__(search_space, evaluator, normalized_objectives, **kwargs)

    @property
    def name(self):
        return 'MoSegNASBenchmark'

    @property
    def pareto_front(self):
        raise NotImplementedError

    @property
    def pareto_set(self):
        raise NotImplementedError

