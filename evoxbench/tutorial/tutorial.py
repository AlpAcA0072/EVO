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
# from evoxbench.benchmarks import NASBench101Benchmark, NASBench101Evaluator
# objs = 'err&params'
# benchmark = NASBench101Benchmark(objs=objs, normalized_objectives=False)
# evaluator =  NASBench101Evaluator()
# print("Benchmaking on NB101 search space with objectives: {}".format(objs))


# N = 10
# archs = benchmark.search_space.sample(N)
# print('Randomly create {} architectures:'.format(N))
# print(archs)

# # encode architecture (phenotype) to decision variables (genotypes)
# X = benchmark.search_space.encode(archs)
# print('Encode architectures to decision variables X: ')
# print(X)

# decoded_X = benchmark.search_space.decode(X)
# results = evaluator.evaluate(archs=decoded_X,true_eval=True)
# print(results)


from evoxbench.benchmarks import MoSegNASSearchSpace, MoSegNASEvaluator, MoSegNASBenchmark, MoSegNASSurrogateModel

searchSpace = MoSegNASSearchSpace(subnet_str=True)
surrogateModel = MoSegNASSurrogateModel(pretrained_json = 'F:\EVO\data\moseg\ofa_fanet_plus_bottleneck_rtx_fps@0.5.json')
randomSubnet = searchSpace.sample(n_samples=1)
randomSubnet = [{
            "d": [
                0,
                0,
                0,
                0,
                1
            ],
            "e": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.35,
                0.25,
                0.25,
                0.2,
                0.35,
                0.2,
                0.2,
                0.2,
                0.25
            ],
            "w": [
                1,
                0,
                0,
                1,
                2,
                2
            ]
        }]
params = surrogateModel.params_predictor(subnet=randomSubnet)

