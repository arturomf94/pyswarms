# Import modules
import numpy as np

# Import sphere function as objective function
from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
import pyswarms.backend as P
from pyswarms.backend.topology import AdaptiveRing

N = 6
my_topology = AdaptiveRing() # The Topology Class
my_options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'feasibility': np.zeros(N, dtype = bool)} # arbitrarily set
my_swarm = P.create_swarm(n_particles = N, dimensions=2, options=my_options) # The Swarm Class

print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))

# Define constraints:

def all_constraints(x):
    #feasible = np.logical_and(con1(x), con2(X))
    feasible = con1(x)
    return feasible

def con1(x):
    feasible = x > .5
    feasible = np.asarray([var[0] for var in feasible])
    return feasible

iterations = 1 # Set 100 iterations
for i in range(iterations):
    # Part 1: Update personal best
    my_swarm.current_cost = f(my_swarm.position) # Compute current cost
    my_swarm.pbest_cost = f(my_swarm.pbest_pos)  # Compute personal best pos
    my_swarm.options['feasibility'] = all_constraints(my_swarm.position)
    my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store

    # Part 2: Update global best
    # Note that gbest computation is dependent on your topology
    if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
        my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm, p = 2, k = 5)

    # Let's print our output
    if i%20==0:
        print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

    # Part 3: Update position and velocity matrices
    # Note that position and velocity updates are dependent on your topology
    my_swarm.velocity = my_topology.compute_velocity(my_swarm)
    my_swarm.position = my_topology.compute_position(my_swarm)

print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
print('The best position found by our swarm is: {}'.format(my_swarm.best_pos))
