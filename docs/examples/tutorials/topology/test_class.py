import pyswarms as ps
from pyswarms.utils.functions import constrained as fx
cop = fx.Sphere()
optimizer = ps.single.DynamicTopologyOptimizer(cop = cop,
                                                N = 200,
                                                iterations = 100,
                                                c1 = 0.6,
                                                c2 = 0.3,
                                                w = 0.4,
                                                dim = 2)

results_2k, results_10k, results_20k = optimizer.optimize()

print(results_2k)
print(results_10k)
print(results_20k)
