import pyswarms as ps
from pyswarms.utils.functions import constrained as cops
from pyswarms.backend.topology import AdaptiveRing

class TestAlgorithm():
    def __init__(self, algorithm_name, list_cops, runs, dims, N = 200, iterations = 100, c1 = 0.6, c2 = 0.3, w = 0.4):
        self.algorithm_name = algorithm_name
        self.list_cops = list_cops
        self.runs = runs
        self.dims = dims
        self.N = N
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.my_topology = AdaptiveRing()

    def run(self):
        if self.algorithm_name == 'dynamic topology':
            for cop in self.list_cops:
                for dim in self.dims:
                    optimizer = ps.single.DynamicTopologyOptimizer(cop = cop,
                                                                    N = self.N,
                                                                    iterations = self.iterations,
                                                                    c1 = self.c1,
                                                                    c2 = self.c2,
                                                                    w = self.w,
                                                                    dim = dim)
                    results_2k, results_10k, results_20k = optimizer.optimize()
                    print('COP: ' + str(cop))
                    print('DIM: ' + str(dim))
                    print('2k: ' + str(results_2k))
                    print('10k: ' + str(results_10k))
                    print('20k: ' + str(results_20k))
            return True
        if self.algorithm_name == 'search feasible region':
            for cop in self.list_cops:
                for dim in self.dims:
                    optimizer = ps.single.SearchFeasibleRegion(cop = cop,
                                                                N = self.N,
                                                                iterations = self.iterations,
                                                                c1 = self.c1,
                                                                c2 = self.c2,
                                                                w = self.w,
                                                                dim = dim)
                    best_cost, results_2k, results_10k, results_20k, success = optimizer.optimize()
                    print('COP: ' + str(cop))
                    print('DIM: ' + str(dim))
                    print('2k: ' + str(results_2k))
                    print('10k: ' + str(results_10k))
                    print('20k: ' + str(results_20k))
                    print('Best cost: ' + str(best_cost))
            return True
        raise ValueError('The algorithm is not supported')
