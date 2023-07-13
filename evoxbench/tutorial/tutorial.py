from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import sys
sys.path.append(r"F:\EVO\evoxbench")
# sys.path.append(r"F:\EVO\evoxbench\evoxbench\database")
sys.path.append(r"F:\EVO\database")
sys.path.append(r"F:\EVO\data")
sys.path.append(r"F:\EVO\nasbench")
sys.path.append(r"F:\EVO\evoxbench\evoxbench\database\ORM")
# print(sys.path)

print('Configurating EvoXBench...')
from evoxbench.database.init import config
# config("F:\EVO\evoxbench\evoxbench\database", "F:\EVO\data\data")
config("F:\EVO\database\database", "F:\EVO\data\data")


# NAS-Bench-101 search space
from evoxbench.benchmarks import NASBench101Benchmark, NASBench101Evaluator
objs = 'err&params'  # ['err&params', 'err&flops', 'err&params&flops']
benchmark = NASBench101Benchmark(objs=objs, normalized_objectives=False)
evaluator =  NASBench101Evaluator()
print("Benchmaking on NB101 search space with objectives: {}".format(objs))


N = 10
archs = benchmark.search_space.sample(N)
print('Randomly create {} architectures:'.format(N))
print(archs)

# encode architecture (phenotype) to decision variables (genotypes)
X = benchmark.search_space.encode(archs)
print('Encode architectures to decision variables X: ')
print(X)

decoded_X = benchmark.search_space.decode(X)
results = evaluator.evaluate(archs=decoded_X,true_eval=True)
print(results)