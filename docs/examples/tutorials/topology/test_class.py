import pyswarms as ps
from pyswarms.utils.functions import constrained as fx
cop = fx.C01()
optimizer = ps.single.DynamicTopologyOptimizer(cop = cop,
                                                N = 20,
                                                iterations = 20,
                                                c1 = 0.6,
                                                c2 = 0.3,
                                                w = 0.4,
                                                dim = 4)

optimizer.optimize()
