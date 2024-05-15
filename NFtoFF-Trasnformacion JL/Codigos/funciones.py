"""File used to program all the required functions to make the transformation"""

from math import factorial, cos, sin
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

def legendre_polinom(n: int, m: int, z: int):
    """Function that calculate the legendre polinom"""
    if m > n:
        return 0
    else:
        return special.clpmn(m, n, z, 2)[0][m][n]

# print(legendre_polinom(1,1,4))
# print(special.clpmn(1, 1, 4,2)[0])

def p_function(m: int, n: int, theta: int, phi: int):
    """Functional base Pmn in our spherical system"""
    return np.array([(legendre_polinom(n,m,cos(theta))*np.exp(1j*m*phi)),0,0])
# print(p_function(1,1,5,5))

def legendre_polinom_derived(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom"""
    return -0.5*(((n+m)*(n-m+1)*legendre_polinom(n,m-1,cos(theta))) - (legendre_polinom(n, m+1, cos(theta))))    
# print(legendre_polinom_derived(1,1,5))    

def legendre_polinom_derived_multiplied_by_sin(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom multiplied by Sin(Theta)"""
    return sin(theta)*legendre_polinom_derived(n, m, theta)
# print(legendre_polinom_derived_multiplied_by_sin(1,1,5))
    
def m_legendre_polinom_derived_by_sin(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom multiplicated by Sin(Theta)"""
    return -0.5*((legendre_polinom(n-1,m+1,cos(theta))) + ((n+m-1)*(n+m)*legendre_polinom(n-1,m-1,cos(theta))))
# print(m_legendre_polinom_derived_by_sin(1,1,5))


