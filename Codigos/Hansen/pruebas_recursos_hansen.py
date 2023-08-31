import scipy
from scipy import special
from numpy import sin, cos

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


print(legendre_division(2,1,3.151617))

