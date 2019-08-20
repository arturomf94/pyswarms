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
        feasible = x >= .5
        feasible = np.asarray([var[0] for var in feasible])
        return feasible

    def con2(self, x):
        feasible = x <= .5
        feasible = np.asarray([var[1] for var in feasible])
        return feasible

    def sum_violations(self, x):
        N = len(x)
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0.5 - x[i][0]
            g2 = x[i][1] - 0.5
            sum_inequalities[i] += max(0, g1) + \
                                max(0, g2)
            sum_equalities[i] += 0
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0
            for j in range(dim):
                g1 += x[i][j] ** 2 - \
                        5000 * math.cos(0.1 * \
                        math.pi * x[i][j]) - 4000
            sum_inequalities[i] += max(0, g1)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0
            for j in range(dim):
                g1 += x[i][j] ** 2 - \
                        5000 * math.cos(0.1 * \
                        math.pi * x[i][j]) - 4000
            sum_inequalities[i] += max(0, g1)
            h1 = 0
            for j in range(dim):
                h1 -= x[i][j] * math.sin(2 * x[i][j])
            sum_equalities[i] += math.fabs(h1)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0
            g2 = 0
            for j in range(dim):
                g1 -= x[i][j] * math.sin(2 * x[i][j])
                g2 += x[i][j] * math.sin(x[i][j])
            sum_inequalities[i] += max(0, g1) + \
                                    max(0, g2)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            h1 = 0
            h2 = 0
            h3 = 0
            h4 = 0
            h5 = 0
            h6 = 0
            for j in range(dim):
                h1 -= x[i][j] * math.sin(x[i][j])
                h2 += x[i][j] * math.sin(math.pi * x[i][j])
                h3 -= x[i][j] * math.cos(x[i][j])
                h4 += x[i][j] * math.cos(math.pi * x[i][j])
                h5 += x[i][j] * math.sin(2 * \
                        math.sqrt(math.fabs(x[i][j])))
                h6 -= x[i][j] * math.sin(2 * \
                        math.sqrt(math.fabs(x[i][j])))
            sum_equalities[i] += math.fabs(h1) + \
                                    math.fabs(h2) + \
                                    math.fabs(h3) + \
                                    math.fabs(h4) + \
                                    math.fabs(h5) + \
                                    math.fabs(h6)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            h1 = 0
            h2 = 0
            for j in range(dim):
                h1 += x[i][j] - 100 * math.cos(0.5 * \
                        x[i][j]) + 100
                h2 -= x[i][j] - 100 * math.cos(0.5 * \
                        x[i][j]) + 100
            sum_equalities[i] += math.fabs(h1) + \
                                    math.fabs(h2)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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
            prod = 1
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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 1
            for j in range(dim):
                g1 *= x[i][j]
            sum_inequalities[i] += max(0, g1)
            h1 = 0
            for j in range(dim - 1):
                h1 += (x[i][j] - x[i][j+1]) ** 2
            sum_equalities[i] += math.fabs(h1)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0
            g2 = 0
            g3 = 0
            for j in range(dim):
                g1 += x[i][j] ** 2 - 10 * \
                        math.cos(2 * math.pi * \
                        x[i][j]) + 10
                g2 += x[i][j]
                g3 += x[i][j]
            g1 -= 100
            g2 -= 2 * dim
            g3 = 5 - g3
            sum_inequalities[i] += max(0, g1) + \
                                    max(0, g2) + \
                                    max(0, g3)
        total_violations = sum_inequalities + sum_equalities
        return total_violations

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

    def sum_violations(self, x):
        N = len(x)
        dim = len(x[0])
        sum_equalities = np.zeros(N)
        sum_inequalities = np.zeros(N)
        for i in range(N):
            g1 = 0
            g2 = 0
            for j in range(dim - 1):
                g1 += -10 * math.exp(-0.2 * math.sqrt(x[i][j] ** 2 + \
                        x[i][j + 1] ** 2))
            for j in range(dim):
                g2 += math.sin(2 * x[i][j]) ** 2
            g1 += (dim - 1) * 10 / math.exp(-5)
            g2 -= 0.5 * dim
            sum_inequalities[i] += max(0, g1) + \
                                    max(0, g2)
        total_violations = sum_inequalities + sum_equalities
        return total_violations
