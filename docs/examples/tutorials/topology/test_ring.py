# Import modules
import numpy as np
import numpy.ma as ma
import math
import pyswarms.backend as P
from pyswarms.backend.topology import AdaptiveRing
from pyswarms.utils.functions.constrained import Sphere
cop = Sphere()
N = 100
dim = 2
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
my_swarm.pbest_pos = np.asarray([None] * N)
my_swarm.pbest_cost = np.asarray([None] * N)
iterations = 100
k_delta = math.ceil( N / iterations)
k = k_delta
my_swarm.options['feasibility'] = cop.constraints(my_swarm.position)
my_swarm.current_cost = cop.objective(my_swarm.position)
for particle_id in range(N):
    if my_swarm.options['feasibility'][particle_id] == True:
        my_swarm.pbest_pos[particle_id] = my_swarm.position[particle_id]
        my_swarm.pbest_cost[particle_id] = my_swarm.current_cost[particle_id]

for i in range(iterations):
    if np.all(my_swarm.pbest_cost == None) == False:
        my_swarm.best_cost = min(x for x in my_swarm.pbest_cost if x is not None)
        min_pos_id = np.where(my_swarm.pbest_cost == my_swarm.best_cost)[0][0]
        my_swarm.options['best_position'] = my_swarm.pbest_pos[min_pos_id]
    if k < N:
        my_swarm.best_pos = my_topology.compute_gbest(my_swarm, p = 2, k = k)
    else:
        my_swarm.best_pos = my_topology.compute_gbest(my_swarm, p = 2, k = N)
    k += k_delta
    if i%50==0:
        print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))
    my_swarm.velocity = my_topology.compute_constrained_velocity(my_swarm)
    my_swarm.position = my_topology.compute_position(my_swarm)
    my_swarm.options['feasibility'] = cop.constraints(my_swarm.position)
    my_swarm.current_cost = cop.objective(my_swarm.position)
    my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_constrained_pbest(my_swarm)

print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
print('The best position found by our swarm is: {}'.format(my_swarm.options['best_position']))
