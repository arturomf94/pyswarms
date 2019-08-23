# Import modules
import numpy as np
import numpy.ma as ma
import math
import pyswarms.backend as P
from pyswarms.backend.topology import AdaptiveRing
from pyswarms.utils.functions.constrained import C13
cop = C13()
N = 100
dim = 6
l_lim = cop.l_lim
u_lim = cop.u_lim
if l_lim == None or u_lim == None:
    bounds = None
else:
    l_lims = np.asarray([l_lim] * dim)
    u_lims = np.asarray([u_lim] * dim)
    bounds = (l_lims, u_lims)
my_topology = AdaptiveRing()
my_options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4,
                'feasibility': np.zeros(N, dtype = bool),
                'best_position': None}
my_swarm = P.create_swarm(n_particles = N, dimensions=dim, options=my_options, bounds = bounds)
print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))

iterations = 200
k_delta = math.ceil( (N - 1) / iterations)
k = k_delta

my_swarm.options['feasibility'] = cop.constraints(my_swarm.position)
my_swarm.current_cost = cop.sum_violations(my_swarm.position)
my_swarm.pbest_pos = my_swarm.position
my_swarm.pbest_cost = my_swarm.current_cost


for i in range(iterations):
    my_swarm.options['feasibility'] = cop.constraints(my_swarm.position)
    if np.all(my_swarm.options['feasibility'] == False) == False:
        break
    my_swarm.current_cost = cop.sum_violations(my_swarm.position)
    my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)

    my_swarm.best_cost = min(x for x in my_swarm.pbest_cost if x is not None)
    min_pos_id = np.where(my_swarm.pbest_cost == my_swarm.best_cost)[0][0]
    my_swarm.options['best_position'] = my_swarm.pbest_pos[min_pos_id]
    my_swarm.best_pos = my_topology.compute_gbest(my_swarm, p = 2, k = (N - 1))


    if i%50==0:
        print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

    new_best_pos = np.empty([N, dim])
    for n in range(N):
        for d in range(dim):
            new_best_pos[n][d] = my_swarm.best_pos[n][d]
    my_swarm.best_pos = new_best_pos
    my_swarm.velocity = my_topology.compute_velocity(my_swarm)
    my_swarm.position = my_topology.compute_position(my_swarm)


if np.all(my_swarm.options['feasibility'] == False) == False:
    print('The feasible region was found!')
    print('The following are all the known feasible points:')
    print(my_swarm.position[my_swarm.options['feasibility'] == True])
else:
    print('The feasible region was not found :(')
    print('The best sum of violations found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
    print('The best position found by our swarm is: {}'.format(my_swarm.options['best_position']))
