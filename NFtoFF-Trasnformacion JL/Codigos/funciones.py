"""File used to program all the required functions to make the transformation"""

import numpy as np
import scipy

def spherical_hankenl_H1(n: int,m: int):
    """Spherical Hankel function of first kind"""
    return scipy.special.spherical_jn(n,m)+1j*scipy.special.spherical_yn(n,m)

def hRDer(n: int, k: int, R: int):
    """Derivada de la función Hankel"""
    return ((1+n)*spherical_hankenl_H1(n,k*R)) - (k*R*spherical_hankenl_H1(1+n,k*R))
print(hRDer(2, 3, 5))