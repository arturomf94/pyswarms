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

optimizer.optimize()
