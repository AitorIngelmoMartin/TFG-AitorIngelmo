"""File used to program all the required functions to make the transformation"""

from math import factorial
import numpy as np
from scipy import special

def spherical_hankenl_H1(n: int,m: int):
    """Spherical Hankel function of first kind"""
    return special.spherical_jn(n,m)+1j*special.spherical_yn(n,m)

def hRDer(n: int, k: int, R: int):
    """Derivada de la función Hankel"""
    return ((1+n)*spherical_hankenl_H1(n,k*R)) - (k*R*spherical_hankenl_H1(1+n,k*R))
# print(hRDer(2, 3, 5))

def gamma(m: int, n: int):
    """Function gamma used in our transformation"""
    return np.sqrt(((2*n+1)*factorial(n-m))/(4*np.pi*n*(n+1)*factorial(n+m)))
# print(gamma(1,2))

def legendre_polinom(n,m,z):
    """Function that calculate the legendre polinom"""
    if m > n:
        return 0
    else:
        return special.clpmn(m, n, z, 2)[0][m][n]

# print(legendre_polinom(1,1,4))
# print(special.clpmn(1, 1, 4,2)[0])

# print(special.lpmn(1,1,3)[0][1][0]+1j*special.lpmn(1,1,3)[0][1][1])
# print((1+special.lpmn(1,1,3)[0][1][0]+1j*special.lpmn(1,1,3)[0][1][1])+(5+1j*4))
# def p_function(m: int, n: int, theta: int, phi: int):
#     """Functional base Pmn in our spherical system"""
#     return np.array([,0,0])
