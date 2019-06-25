# -*- coding: utf-8 -*-

"""single_obj.py: collection of single-objective constrained functions

Obtained from: http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2017/Constrained/Technical%20Report%20-%20CEC2017-%20Final.pdf
"""

# Import modules
import numpy as np
import math

class Sphere():
    def __init__(self):
        self.l_lim = None
        self.u_lim = None

    def objective(self, x):
        """Sphere objective function.

        Has a global minimum at :code:`0` and with a search domain of
            :code:`[-inf, inf]`

        Parameters
        ----------
        x : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`

        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        j = (x ** 2.0).sum(axis=1)

        return j

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        #feasible = con1(x)
        return feasible

    def con1(self, x):
        feasible = x > .5
        feasible = np.asarray([var[0] for var in feasible])
        return feasible

    def con2(self, x):
        feasible = x < .5
        feasible = np.asarray([var[1] for var in feasible])
        return feasible

class C01():
    def __init__(self):
        self.l_lim = -100
        self.u_lim = 100

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][:j].sum() ** 2
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = self.con1(x)
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] ** 2 - \
                        5000 * math.cos(0.1 * \
                        math.pi * x[i][j]) - 4000
            y[i] = sum
        feasible = y <= 0
        return feasible

class C03():
    def __init__(self):
        self.l_lim = -100
        self.u_lim = 100

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][:j].sum() ** 2
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] ** 2 - \
                        5000 * math.cos(0.1 * \
                        math.pi * x[i][j]) - 4000
            y[i] = sum
        feasible = y <= 0
        return feasible

    def con2(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(2 * x[i][j])
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

class C04():
    def __init__(self):
        self.l_lim = -10
        self.u_lim = 10

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] ** 2 - 10 * \
                        math.cos(2 * math.pi * \
                        x[i][j]) + 10
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(2 * x[i][j])
            y[i] = - sum
        feasible = y <= 0
        return feasible

    def con2(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(x[i][j])
            y[i] = sum
        feasible = y <= 0
        return feasible

class C06():
    def __init__(self):
        self.l_lim = -20
        self.u_lim = 20

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] ** 2 - 10 * \
                        math.cos(2 * math.pi * \
                        x[i][j]) + 10
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), \
                                    np.logical_and(self.con2(x), \
                                    np.logical_and(self.con3(x), \
                                    np.logical_and(self.con4(x), \
                                    np.logical_and(self.con5(x), \
                                    self.con6(x))))))
        return feasible

    def con1(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(x[i][j])
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

    def con2(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(math.pi * x[i][j])
            y[i] = math.fabs(sum) - e
        feasible = y <= 0
        return feasible

    def con3(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.cos(x[i][j])
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

    def con3(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.cos(x[i][j])
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

    def con4(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.cos(math.pi * x[i][j])
            y[i] = math.fabs(sum) - e
        feasible = y <= 0
        return feasible

    def con5(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(2 * \
                        math.sqrt(math.fabs(x[i][j])))
            y[i] = math.fabs(sum) - e
        feasible = y <= 0
        return feasible

    def con6(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(2 * \
                        math.sqrt(math.fabs(x[i][j])))
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

class C07():
    def __init__(self):
        self.l_lim = -50
        self.u_lim = 50

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] * math.sin(x[i][j])
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        return feasible

    def con1(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] - 100 * math.cos(0.5 * \
                        x[i][j]) + 100
            y[i] = math.fabs(sum) - e
        feasible = y <= 0
        return feasible

    def con2(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] - 100 * math.cos(0.5 * \
                        x[i][j]) + 100
            y[i] = math.fabs(-sum) - e
        feasible = y <= 0
        return feasible

class C11():
    def __init__(self):
        self.l_lim = -100
        self.u_lim = 100

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j]
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            prod = 0
            for j in range(dim):
                prod *= x[i][j]
            y[i] = prod
        feasible = y <= 0
        return feasible

    def con2(self, x):
        e = 0.0001
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim - 1):
                sum += (x[i][j] - x[i][j+1]) ** 2
            y[i] = math.fabs(sum) - e
        feasible = y <= 0
        return feasible

class C13():
    def __init__(self):
        self.l_lim = -100
        self.u_lim = 100

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim - 1):
                sum += 100 * (x[i][j] ** 2 - x[i][j + 1]) \
                        ** 2 + (x[i][j] - 1) ** 2
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), \
                                    np.logical_and(self.con2(x), \
                                    self.con3(x)))
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j] ** 2 - 10 * \
                        math.cos(2 * math.pi * \
                        x[i][j]) + 10
            y[i] = sum - 100
        feasible = y <= 0
        return feasible

    def con2(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j]
            y[i] = sum - 2 * dim
        feasible = y <= 0
        return feasible

    def con3(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += x[i][j]
            y[i] = 5 - sum
        feasible = y <= 0
        return feasible

class C19():
    def __init__(self):
        self.l_lim = -50
        self.u_lim = 50

    def objective(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += math.pow(math.fabs(x[i][j]), 0.5) + \
                        2 * math.sin(x[i][j] ** 3)
            y[i] = sum
        return y

    def constraints(self, x):
        feasible = np.logical_and(self.con1(x), self.con2(x))
        return feasible

    def con1(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim - 1):
                sum += -10 * math.exp(-0.2 * math.sqrt(x[i][j] ** 2 + \
                        x[i][j + 1] ** 2))
            y[i] = sum + (dim - 1) * 10 / math.exp(-5)
        feasible = y <= 0
        return feasible

    def con2(self, x):
        N = len(x)
        y = np.zeros(N)
        dim = len(x[0])
        for i in range(N):
            sum = 0
            for j in range(dim):
                sum += math.sin(2 * x[i][j]) ** 2
            y[i] = sum - 0.5 * dim
        feasible = y <= 0
        return feasible
