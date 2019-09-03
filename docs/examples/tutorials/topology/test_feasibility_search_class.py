import pyswarms as ps
from pyswarms.utils.functions import constrained as fx
cop = fx.C11()
optimizer = ps.single.SearchFeasibleRegion(cop = cop,
                                            N = 200,
                                            iterations = 100,
                                            c1 = 0.6,
                                            c2 = 0.3,
                                            w = 0.4,
                                            dim = 2)

best_cost, results_2k, results_10k, results_20k, success = optimizer.optimize()

print(best_cost)
print(results_2k)
print(results_10k)
print(results_20k)
print(success)
