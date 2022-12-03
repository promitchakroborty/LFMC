import numpy as np
import scipy.stats as stats


# Four Branch Function
def lf1eval(x):
    res = -1*(3 + (((x[0]-x[1])**2)/10) - ((x[0]+x[1])/(2**0.5)))
    return res


def lf2eval(x):
    res = -1*(3 + (((x[0]-x[1])**2)/10) + ((x[0]+x[1])/(2**0.5)))
    return res


def lf3eval(x):
    res = -1*(x[0] - x[1] + (6/(2**0.5)))
    return res


def lf4eval(x):
    res = -1*(x[1] - x[0] + (6 / (2 ** 0.5)))
    return res


def hfeval(x):
    t1 = 3 + (((x[0]-x[1])**2)/10) - ((x[0]+x[1])/(2**0.5))
    t2 = 3 + (((x[0]-x[1])**2)/10) + ((x[0]+x[1])/(2**0.5))
    t3 = x[0] - x[1] + (6/(2**0.5))
    t4 = x[1] - x[0] + (6 / (2 ** 0.5))
    res = -1*(np.amin([t1, t2, t3, t4]))
    return res


# Rastrigin Function Type 1

# def lf1eval(x):
#     res = -1*(10 - (x[0]**2) + (5*np.cos(2*np.pi*x[0])))
#     return res
#
#
# def lf2eval(x):
#     res = -1*(10 - (x[1]**2) + (5*np.cos(2*np.pi*x[1])))
#     return res
#
#
# def hfeval(x):
#     res = -1 * (10 - (x[0] ** 2) - (x[1]**2) + (5 * np.cos(2 * np.pi * x[0])) + (5*np.cos(2*np.pi*x[1])))
#     return res


# Rastrigin Function Type 2

# def lf1eval(x):
#     res = -1*(10 - (x[0]**2) - (x[1]**2))
#     return res
#
#
# def lf2eval(x):
#     res = -1*(10 + (5*np.cos(2*np.pi*x[1])) + (5*np.cos(2*np.pi*x[0])))
#     return res
#
#
# def hfeval(x):
#     res = -1 * (10 - (x[0] ** 2) - (x[1]**2) + (5 * np.cos(2 * np.pi * x[0])) + (5*np.cos(2*np.pi*x[1])))
#     return res
