import pyswarms as ps
from pyswarms.utils.functions import constrained as cops
from pyswarms.backend.topology import AdaptiveRing
from pylatex import Tabular, Document, Section

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
            geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
            doc = Document(geometry_options=geometry_options)
            for cop in self.list_cops:
                cop_str = str(cop)
                cop_str = cop_str[cop_str.index('C'):cop_str.index('C') + 3]
                results = []
                for dim in self.dims:
                    optimizer = ps.single.DynamicTopologyOptimizer(cop = cop,
                                                                    N = self.N,
                                                                    iterations = self.iterations,
                                                                    c1 = self.c1,
                                                                    c2 = self.c2,
                                                                    w = self.w,
                                                                    dim = dim)
                    results_2k, results_10k, results_20k = optimizer.optimize()
                    print('COP: ' + cop_str)
                    print('DIM: ' + str(dim))
                    print('2k: ' + str(results_2k))
                    print('10k: ' + str(results_10k))
                    print('20k: ' + str(results_20k))
                    results.append(('DIM' + str(dim), results_2k, results_10k, results_20k))
                with doc.create(Section('Results for ' + 'COP ' + cop_str)):
                    with doc.create(Tabular('|l|l|l|l')) as table:
                        head = tuple([cop_str] + ['2K FES', '10K FES', '20K FES'])
                        table.add_row(head)
                        for result in results:
                            table.add_hline()
                            table.add_row(result)
            doc.generate_pdf('results/dynamic_topology', clean_tex=False, compiler = 'pdflatex')
            return True
        if self.algorithm_name == 'search feasible region':
            geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
            doc = Document(geometry_options=geometry_options)
            for cop in self.list_cops:
                cop_str = str(cop)
                cop_str = cop_str[cop_str.index('C'):cop_str.index('C') + 3]
                results = []
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
                    if success == True:
                        print('The feasible region was found!')
                    else:
                        print('The feasible region was not found :(')
                    print('2k: ' + str(results_2k))
                    print('10k: ' + str(results_10k))
                    print('20k: ' + str(results_20k))
                    print('Best cost: ' + str(best_cost))
                    results.append(('DIM' + str(dim), success, results_2k, results_10k, results_20k, best_cost))
                with doc.create(Section('Results for ' + 'COP ' + cop_str)):
                    with doc.create(Tabular('|l|l|l|l|l|l')) as table:
                        head = tuple([cop_str] + ['Success', '2K FES', '10K FES', '20K FES', 'Best Cost'])
                        table.add_row(head)
                        for result in results:
                            table.add_hline()
                            table.add_row(result)
            doc.generate_pdf('results/feasible_region_search', clean_tex=False, compiler = 'pdflatex')
            return True
        raise ValueError('The algorithm is not supported')
