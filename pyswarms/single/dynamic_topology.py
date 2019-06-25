import numpy as np
import numpy.ma as ma
import math
import pyswarms.backend as P
from pyswarms.backend.topology import AdaptiveRing


class DynamicTopologyOptimizer():

    def __init__(self, cop, N, iterations, c1, c2, w, dim):
        self.cop = cop # Constrained function class
        self.N = N
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.my_topology = AdaptiveRing()

    def optimize(self):
        fes = 0 # Function evaluations
        l_lim = self.cop.l_lim
        u_lim = self.cop.u_lim
        if l_lim == None or u_lim == None:
            bounds = None
        else:
            l_lims = np.asarray([l_lim] * self.dim)
            u_lims = np.asarray([u_lim] * self.dim)
            bounds = (l_lims, u_lims)

        my_options = {'c1': self.c1, 'c2': self.c2, 'w': self.w,
                        'feasibility': np.zeros(self.N, dtype = bool),
                        'best_position': None}
        my_swarm = P.create_swarm(n_particles = self.N, dimensions = self.dim, options = my_options, bounds = bounds)
        my_swarm.pbest_pos = np.asarray([None] * self.N)
        my_swarm.pbest_cost = np.asarray([None] * self.N)
        #print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))
        k_delta = math.ceil( (self.N - 1) / self.iterations) # additional neighbors each gen
        k = k_delta

        # Check feasibility and update personal best only if feasible
        my_swarm.options['feasibility'] = self.cop.constraints(my_swarm.position)
        my_swarm.current_cost = self.cop.objective(my_swarm.position)
        fes += self.N
        for particle_id in range(self.N):
            if my_swarm.options['feasibility'][particle_id] == True:
                my_swarm.pbest_pos[particle_id] = my_swarm.position[particle_id]
                # my_swarm.pbest_cost[particle_id] = self.cop.objective(np.array([my_swarm.pbest_pos[particle_id]]))[0] # Compute personal best pos
                # fes += self.dim
                my_swarm.pbest_cost[particle_id] = my_swarm.current_cost[particle_id]

        for i in range(self.iterations):
            # Part 1: Update personal best if feasible
            my_swarm.options['feasibility'] = self.cop.constraints(my_swarm.position)
            my_swarm.current_cost = self.cop.objective(my_swarm.position) # Compute current cost
            fes += self.N
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_constrained_pbest(my_swarm) # Update and store
            # Part 2: Update global best
            # Note that gbest computation is dependent on your topology
            if np.all(my_swarm.pbest_cost == None) == False:
                my_swarm.best_cost = min(x for x in my_swarm.pbest_cost if x is not None)
                min_pos_id = np.where(my_swarm.pbest_cost == my_swarm.best_cost)[0][0]
                my_swarm.options['best_position'] = my_swarm.pbest_pos[min_pos_id]

            if k < self.N - 1:
                my_swarm.best_pos = self.my_topology.compute_gbest(my_swarm, p = 2, k = k)
            else:
                my_swarm.best_pos = self.my_topology.compute_gbest(my_swarm, p = 2, k = (self.N - 1))
            k += k_delta
            # Let's print our output
            # if i%50==0:
            #     print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))
            if fes == 2000 or fes == 10000 or fes == 20000:
                print('FES: {} | my_swarm.best_cost: {:.4f}'.format(fes, my_swarm.best_cost))
                if fes == 20000:
                    break

            # Part 3: Update position and velocity matrices
            # Note that position and velocity updates are dependent on your topology
            my_swarm.velocity = self.my_topology.compute_constrained_velocity(my_swarm)
            my_swarm.position = self.my_topology.compute_position(my_swarm)

        print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
        print('The best position found by our swarm is: {}'.format(my_swarm.options['best_position']))
