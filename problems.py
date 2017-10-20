import numpy as np
# =============================================================================
#   J.J. Liang, T.P. Runarsson, E. Mezura-Montes, M. Clerc, P.N. Suganthan,
#   C.A. Coello Coello, K. Deb. Problem Definitions and Evaluation Criteria
#   for the CEC 2006 Special Session on Constrained Real-Parameter Optimization,
#   Technical Report, 2006. http://www.ntu.edu.sg/home/EPNSugan
# =============================================================================


# G01
class G01:
    # The optimum solution is x* = (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1) where
    # where six constraints are active (g1, g2, g3, g7, g8 and g9) and f(x*) = -15.
    optimum = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])
    foptimum = -15

    def __init__(self):
        self.dimension = 13
        self.lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0, 0]
        self.upper_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1]
        self.g = np.empty(9)

    def fitness(self, x):

        return 5 * np.sum(x[0:4]) - 5 * np.sum(x[0:4]**2) - np.sum(x[4:13])

    def constraints(self, x):

        self.g[0] = 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
        self.g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10
        self.g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
        self.g[3] = -8 * x[0] + x[9]
        self.g[4] = -8 * x[1] + x[10]
        self.g[5] = -8 * x[2] + x[11]
        self.g[6] = -2 * x[3] - x[4] + x[9]
        self.g[7] = -2 * x[5] - x[6] + x[10]
        self.g[8] = -2 * x[7] - x[8] + x[11]

        return self.g


# G02
class G02:
    # The optimum solution is x* = (3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469,
    #                               3.02792915885555, 2.99382606701730, 2.95866871765285, 2.92184227312450,
    #                               0.49482511456933, 0.48835711005490, 0.48231642711865, 0.47664475092742,
    #                               0.47129550835493, 0.46623099264167, 0.46142004984199, 0.45683664767217,
    #                               0.45245876903267, 0.44826762241853, 0.44424700958760, 0.44038285956317)
    # where f(x*) = -0.80361910412559.

    def __init__(self):
        self.dimension = 20
        self.lower_bounds = self.dimension * [0]
        self.upper_bounds = self.dimension * [10]
        self.g = np.empty(2)

    def fitness(self, x):
        sum_jx = 0

        for j in range(self.dimension):
            sum_jx = sum_jx + (j + 1) * x[j]**2

        return -np.abs((np.sum(np.cos(x)**4) - 2 * np.prod(np.cos(x)**2))/np.sqrt(sum_jx))

    def constraints(self, x):
        self.g[0] = 0.75 - np.prod(x)
        self.g[1] = np.sum(x) - 7.5 * self.dimension

        return self.g


# G03
class G03:
    # The optimum solution is x* = (0.31624357647283069, 0.316243577414338339, 0.316243578012345927,
    #                               0.316243575664017895, 0.316243578205526066, 0.31624357738855069,
    #                               0.316243575472949512, 0.316243577164883938, 0.316243578155920302,
    #                               0.316243576147374916)
    # where f(x*) = -1.00050010001000.

    def __init__(self):
        self.dimension = 10
        self.lower_bounds = self.dimension * [0]
        self.upper_bounds = self.dimension * [1]
        self.g = np.empty(2)

    def fitness(self, x):

        return -(np.sqrt(self.dimension))**self.dimension * np.prod(x)

    def constraints(self, x):
        self.g[0] = np.sum(x**2) - 1 - 0.0001

        return self.g


# G04
class G04:
    # The optimum solution is x* = (78, 33, 29.9952560256815985, 45, 36.7758129057882073)
    #   where f(x*) = -3.066553867178332e+004.

    def __init__(self):
        self.dimension = 5
        self.lower_bounds = [ 78, 33, 27, 27, 27]
        self.upper_bounds = [102, 45, 45, 45, 45]
        self.g = np.empty(6)

    def fitness(self, x):

        return 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    def constraints(self, x):
        u = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        v = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2
        w = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

        self.g[0] = -u
        self.g[1] = u - 92
        self.g[2] = -v + 90
        self.g[3] = v - 110
        self.g[4] = -w + 20
        self.g[5] = w - 25

        return self.g


# G05
class G05:
    # The optimum solution is x* = (679.945148297028709, 1026.06697600004691,
    #                                 0.118876369094410433, -0.39623348521517826)
    # where f(x*) = 5126.4967140071.

    def __init__(self):
        self.dimension = 4
        self.lower_bounds = [0, 0, -0.55, -0.55]
        self.upper_bounds = [1200, 1200, 0.55, 0.55]
        self.g = np.empty(5)

    def fitness(self, x):

        return 3 * x[0] + 1e-6 * x[0]**3 + 2 * x[1] + 2e-6 / 3 * x[1]**3

    def constraints(self, x):

        self.g[0] = x[2] - x[3] - 0.55
        self.g[1] = x[3] - x[2] - 0.55
        self.g[2] = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0] - 0.0001
        self.g[3] = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1] - 0.0001
        self.g[4] = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8 - 0.0001

        return self.g


# G06
class G06:
    # The optimum solution is x* = (14.09500000000000064, 0.8429607892154795668
    # where f(x*) = -6961.81387558015.

    def __init__(self):
        self.dimension = 2
        self.lower_bounds = [0, 0]
        self.upper_bounds = [100, 100]
        self.g = np.empty(2)

    def fitness(self, x):

        return (x[0] - 10)**3 + (x[1] - 20)**3

    def constraints(self, x):
        self.g[0] = -(x[0] - 5)**2 - (x[1] - 5)**2 + 100
        self.g[1] = (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

        return self.g


# G07
class G07:
    # The optimum solution is x* = (2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173,
    #                               0.990654756560493, 1.43057392853463, 1.32164415364306, 9.82872576524495,
    #                               8.2800915887356, 8.3759266477347)
    # where f(x*) = 24.30620906818.

    def __init__(self):
        self.dimension = 10
        self.lower_bounds = np.zeros(self.dimension)
        self.upper_bounds = 10 * np.ones(self.dimension)
        self.g = np.empty(8)

    def fitness(self, x):

        f = x[0]**2 + x[1]**2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10)**2 + \
            4 * (x[3] - 5)**2 + (x[4] - 3)**2 + 2 * (x[5] - 1)**2 + 5 * x[6]**2 + \
            7 * (x[7] - 11)**2 + 2 * (x[8] - 10)**2 + (x[9] - 7)**2 + 45

        return f

    def constraints(self, x):
        self.g[0] = -105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7]
        self.g[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
        self.g[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
        self.g[3] = 3 * (x[0] - 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120
        self.g[4] = 5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40
        self.g[5] = x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
        self.g[6] = 0.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30
        self.g[7] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9]

        return self.g


# G08
class G08:
    # The optimum solution is x* = (1.22797135260752599, 4.24537336612274885) where f(x*) = -0.0958250414180359.

    def __init__(self):
        self.dimension = 2
        self.lower_bounds = np.zeros(self.dimension)
        self.upper_bounds = 10 * np.ones(self.dimension)
        self.g = np.empty(2)

    def fitness(self, x):

        return -(((np.sin(2 * np.pi * x[0]))**3) * np.sin(2 * np.pi * x[1]))/(x[0]**3 * (x[0] + x[1]))

    def constraints(self, x):
        self.g[0] = x[0]**2 - x[1] + 1
        self.g[1] = 1 - x[0] + (x[1] - 4)**2

        return self.g


# G09
class G09:
    # The optimum solution is x* = (2.33049935147405174, 1.95137236847114592, -0.477541399510615805,
    #                               4.36572624923625874, -0.624486959100388983, 1.03813099410962173,
    #                               1.5942266780671519)
    # where f(x*) = -680.630057374402.

    def __init__(self):
        self.dimension = 7
        self.lower_bounds = -10 * np.ones(self.dimension)
        self.upper_bounds = 10 * np.ones(self.dimension)
        self.g = np.empty(4)

    def fitness(self, x):

        f = (x[0] - 10)**2 + 5 * (x[1] - 12)**2 + x[2]**4 + 3 * (x[3] - 11)**2 + \
               10 * x[4]**6 + 7 * x[5]**2 + x[6]**4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]

        return f

    def constraints(self, x):
        self.g[0] = -127 + 2 * x[0]**2 + 3 * x[1]**4 + x[2] + 4 * x[3]**2 + 5 * x[4]
        self.g[1] = -282 + 7 * x[0] + 3 * x[1] + 10 * x[2]**2 + x[3] - x[4]
        self.g[2] = -196 + 23 * x[0] + x[1]**2 + 6 * x[5]**2 - 8 * x[6]
        self.g[3] = 4 * x[0]**2 + x[1]**2 - 3 * x[0] * x[1] + 2 * x[2]**2 + 5 * x[5] - 11 * x[6]

        return self.g

# G10
class G10:
    # The optimum solution is x* = (579.306685017979589, 1359.97067807935605, 5109.97065743133317,
    #                               182.01769963061534, 295.601173702746792, 217.982300369384632,
    #                               286.41652592786852, 395.601173702746735)
    # where f(x*) = 7049.24802052867.

    def __init__(self):
        self.dimension = 8
        self.lower_bounds = [100, 1000, 1000, 10, 10, 10, 10, 10]
        self.upper_bounds = [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]
        self.g = np.empty(6)

    def fitness(self, x):

        return x[0] + x[1] + x[2]

    def constraints(self, x):
        self.g[0] = -1 + 0.0025 * (x[3] + x[5])
        self.g[1] = -1 + 0.0025 * (x[4] + x[6] - x[3])
        self.g[2] = -1 + 0.01 * (x[7] - x[4])
        self.g[3] = - x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
        self.g[4] = - x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]
        self.g[5] = - x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]

        return self.g


# Test problem S-CRES
class S_CRES:

    def __init__(self):
        self.dimension = 2
        self.lower_bounds = [0, 0]
        self.upper_bounds = [6, 6]
        self.g = np.empty(2)

    def fitness(self, x):

        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def constraints(self, x):

        self.g[0] = (x[0] - 0.05)**2 + (x[1] - 2.5)**2 - 4.84
        self.g[1] = 4.84 - x[0]**2 - (x[1] - 2.5)**2

        return self.g