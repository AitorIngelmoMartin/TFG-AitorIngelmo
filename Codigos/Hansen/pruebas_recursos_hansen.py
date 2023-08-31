import scipy
import numpy as np
from scipy import special
from numpy import sin, cos, sign

def legendre_polinom(n,m,z):
    """Function that calculate the legendre polinom"""
    if m > n:
        return 0
    elif n <= m:
        legendre_associate_polynom = special.lpmn(n,m, z)
        return legendre_associate_polynom[0][n][m]
    else:
        legendre_associate_polynom = special.lpmn(m,n, z)
        return legendre_associate_polynom[0][m][n]

def legendre_division(n, m, theta):
    """Function that calculate a fraction that can be NaN"""
    return -0.5*(legendre_polinom(n + 1, m + 1,cos(theta)) + (n - m + 1)*(-n + m + 2)*legendre_polinom(n + 1, m - 1, cos(theta)))

def cot(rad):
    """Function that make the cotang of an angle given in radians"""
    return -np.tan(rad + np.pi/2)

def cosec(rad):
    """Dunction that calculate the cosecant of an angle given in radians"""
    return 1/sin(rad)

def legendre_cos_derivate(n,m,theta):
    """Function that calculate the cos(x) partial derivate of legendre polinom"""
    if not theta == 0 and not theta == np.pi:
        return -(1 + n)*cot(theta)*legendre_polinom(n, np.abs(m), cos(theta)) + \
            (1 + n - np.abs(m))*cosec(theta)*legendre_polinom(n+1, np.abs(m), cos(theta))

def expresion_sign(m):
    """Function that save the a NaN related with 0^0"""
    if m == 0:
        return -1
    return -sign(m)**np.abs(m)

def Ksmn(s, n, m, theta, phi):
    """Function that calculate the Ksmn expresion from hankel book"""
    if s == 1:
        np.sqrt(2/(2*(n+1)))*expresion_sign(m)* np.exp(1j * m * phi) * (-1j)**(n+1) *np.array[legendre_division(n, m, theta),-legendre_cos_derivate(n, m, theta)]
print(legendre_cos_derivate(2,2,5))