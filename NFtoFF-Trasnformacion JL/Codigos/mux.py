from math import factorial
import numpy as np
from numpy import cos, sin
from scipy import special


def legendre_polinom(n: int, m: int, z: int):
    """Function that calculate the legendre polinom"""
    if np.abs(m) > n:
        return 0
    else:
        return special.clpmn(m, n, z, 2)[0][np.abs(m)][n]
    
theta = np.linspace(-np.pi, np.pi, num=25)
n = 1
m = 1
legendre_polinom(n,m,cos(theta))