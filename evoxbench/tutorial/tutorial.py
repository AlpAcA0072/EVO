from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import sys
sys.path.append(r"D:\evo\evoxbench")
sys.path.append(r"D:\evo\database")
sys.path.append(r"D:\evo\data")
sys.path.append(r"D:\evo\nasbench")
# print(sys.path)

print('Configurating EvoXBench...')
from evoxbench.database.init import config
config("D:\evo\database\database", "D:\evo\data\data")


# NAS-Bench-101 search space
from evoxbench.benchmarks import NASBench101Benchmark
objs = 'err&params'  # ['err&params', 'err&flops', 'err&params&flops']
benchmark = NASBench101Benchmark(objs=objs, normalized_objectives=False)
print("Benchmaking on NB101 search space with objectives: {}".format(objs))

# # NAS-Bench-201 search space
# from evoxbench.benchmarks import NASBench201Benchmark
# # hardware = 'edgegpu'  # ['edgegpu', 'raspi4', 'edgetpu', 'pixel3', 'eyeriss', 'fpga']
# # ['err&params', 'err&flops', 'err&latency', 'err&params&flops', 'err&params&latency', ...]
# objs = 'err&params&flops&edgegpu_latency&edgegpu_energy'
# benchmark = NASBench201Benchmark(objs=objs, normalized_objectives=False)
# print("Benchmaking on NB201 search space with objectives: {}".format(objs))

# # NATS size search space 
# from evoxbench.benchmarks import NATSBenchmark
# objs = 'err&params&flops&latency'
# # ['err&params', 'err&flops', 'err&latency', 'err&params&flops', 'err&params&latency', ...]
# benchmark = NATSBenchmark(objs=objs, normalized_objectives=False)
# print("Benchmaking on NATS search space with objectives: {}".format(objs))

# # DARTS search space
# from evoxbench.benchmarks import DARTSBenchmark
# objs = 'err&params'  # ['err&params', 'err&flops', 'err&params&flops']
# benchmark = DARTSBenchmark(objs=objs, normalized_objectives=False)
# print("Benchmaking on DARTS search space with objectives: {}".format(objs))

# let's randomly create N architectures
N = 1
archs = benchmark.search_space.sample(N)
print('Randomly create {} architectures:'.format(N))
print(archs)

# encode architecture (phenotype) to decision variables (genotypes)
X = benchmark.search_space.encode(archs)
print('Encode architectures to decision variables X: ')
print(X)