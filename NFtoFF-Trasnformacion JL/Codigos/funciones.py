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
    if np.abs(m) > n:
        return 0
    else:
        return special.clpmn(m, n, z, 2)[0][np.abs(m)][n]
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

def legendre_polinom_derived_sin(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom multiplied by Sin(Theta)"""
    return sin(theta)*legendre_polinom_derived(n, m, theta)
# print(legendre_polinom_derived_sin(1,1,5))
    
def m_legendre_polinom_derived_by_sin(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom multiplicated by Sin(Theta)"""
    return -0.5*((legendre_polinom(n-1,m+1,cos(theta))) + ((n+m-1)*(n+m)*legendre_polinom(n-1,m-1,cos(theta))))
# print(m_legendre_polinom_derived_by_sin(1,1,5))

def b_sin_function(m: int, n: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Bmn*Sin(theta)"""
    return np.array([0, legendre_polinom_derived_sin(n, m, theta)*np.exp(1j*m*phi), 1j*m*legendre_polinom(n,m,cos(theta))*np.exp(1j*m*phi)])
# print(b_sin_function(2,2,5,5))

def c_sin_function(m: int, n: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Cmn"""
    return np.array([0, 1j*m*legendre_polinom(n, m, cos(theta))*np.exp(1j*m*phi),-legendre_polinom_derived_sin(n,m,theta)*np.exp(1j*m*phi)])
# print(c_sin_function(2,2,5,5))

def m_sin_function(m: int, n: int,k: int,R: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Mmn*sin(theta)"""
    return gamma(m,n)*spherical_hankenl_H1(n, k*R)*c_sin_function(m,n,theta,phi)
# print(m_sin_function(2,2,5,5,5,5))

def n_sin_function(m: int, n: int, k: int,R: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Nmn*sin(theta)"""
    return gamma(m,n)*((n*(n+1)*(spherical_hankenl_H1(n,k*R)/(k*R)))*(p_function(m,n,theta,phi)*sin(theta))+((hRDer(n,k,R)/(k*R))*b_sin_function(m, n, theta, phi)))
# print(n_sin_function(2,2,5,5,5,5))

def b_function(m: int, n: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Bmn"""
    return np.array([0, legendre_polinom_derived(n,m,theta)*np.exp(1j*m*phi),1j*m_legendre_polinom_derived_by_sin(n,m,theta)*np.exp(1j*m*phi)])
# print(b_function(3,3,4,4))

def c_function(m: int, n: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Cmn"""
    return np.array([0, 1j*m_legendre_polinom_derived_by_sin(n,m,theta)*np.exp(1j*m*phi),-legendre_polinom_derived(n,m,theta)*np.exp(1j*m*phi)])
# print(c_function(1,1,3,3))

def m_function(m: int, n: int, k: int, R: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Mmn"""
    return gamma(m,n)*spherical_hankenl_H1(n, k*R)*c_function(m,n,theta,phi)
# print(m_function(3,3,4,4,4,4))

def n_function(m: int, n: int, k: int, R: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Nmn"""
    return gamma(m,n)*((n*(n+1)*(spherical_hankenl_H1(n,k*R)/(k*R)))*p_function(m,n,theta,phi)+((hRDer(n,k,R)/(k*R))*b_function(m, n, theta, phi)))
# print(n_function(3,3,4,4,4,4))

def z1(m: int, n: int):
    """Function Z1mn"""
    return (-1)**m*((4*np.pi)/(2*n+1))
# print(z1(2,2))

def z2(m: int, n: int):
    """Function Z2mn"""
    return (-1)**m*(4*np.pi*n)*((n+1)/(2*n+1))
# print(z2(3,3))

def z3(m: int, n: int):
    """Function Z3mn, wich is the same than Z2mn"""
    return (-1)**m*(4*np.pi*n)*((n+1)/(2*n+1))
# print(z3(4,4))

def a_coef_function(g_data: list, m: int, n: int, k: int, R: int):
    """Function used to calculate our spherical base function Acoefmn"""
    return (((-1)**m)*(g_data[n-1][n+m])/(4*np.pi*gamma(n,m)))* (1/spherical_hankenl_H1(n, k*R))
# g_data = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# print(g_data[1][1+1+1])
# print(a_function(g_data,1,1,2,2))

def b_coef_function(e_data: list, m: int, n: int, k: int, R: int):
    """Function used to calculate our spherical base function Bcoefmn"""
    return (((-1)**m)*(e_data[n-1][n+m])/(4*np.pi*gamma(n,m)))* ((k*R)/hRDer(n, k, R))
# e_data = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# print(e_data[1][1+1+1])
# print(b_function(e_data,1,1,2,2))

# a_coeff_dummy = [[1.0, 1.0, 1.0], [0.25, 0.5, 1.0, 0.5, 0.25], [0.03703703703703703, 0.1111111111111111, 0.3333333333333333, 1.0, 0.3333333333333333, 0.1111111111111111, 0.03703703703703703], [0.00390625, 0.015625, 0.0625, 0.25, 1.0, 0.25, 0.0625, 0.015625, 0.00390625], [0.0003200000000000001, 0.0016000000000000003, 0.008000000000000002, 0.04000000000000001, 0.2, 1.0, 0.2, 0.04000000000000001, 0.008000000000000002, 0.0016000000000000003, 0.0003200000000000001]]
# b_coeff_dummy = [[1.0, 1.0, 1.0], [0.0625, 0.25, 1.0, 0.25, 0.0625], [0.001371742112482853, 0.012345679012345678, 0.1111111111111111, 1.0, 0.1111111111111111, 0.012345679012345678, 0.001371742112482853], [1.52587890625e-05, 0.000244140625, 0.00390625, 0.0625, 1.0, 0.0625, 0.00390625, 0.000244140625, 1.52587890625e-05], [1.0240000000000002e-07, 2.56e-06, 6.400000000000001e-05, 0.0016, 0.04, 1.0, 0.04, 0.0016, 6.400000000000001e-05, 2.56e-06, 1.0240000000000002e-07]]

N = 5
def make_dummy_a_coef(N):
    """Function used to test the orthogonality of Amn"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append((1/n)**np.abs(m))
        total_result.append(value_calculated)
    return total_result
# make_dummy_a_coef(5)

# Calculo de Bmncoef
def make_dummy_b_coef(N):
    """Function used to test the orthogonality of Bmn"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append((1/(n*n))**np.abs(m))
        total_result.append(value_calculated)
    return total_result

def e_field_sint(*,k: int, R: int, acoeff: list, bcoeff: list, theta: int, phi: int):
    """Function used to obtain a sintetic Field"""
    M = len(acoeff)
    total_result = 0
    for n in range(1, M + 1):
        values_calculated = 0
        for m in range(-n, n + 1):
            value = ((2*n+1)/(n*(n+1)))*(acoeff[n-1][m+n]*m_function(m,n,k,R,theta,phi) + bcoeff[n-1][m+n]*n_function(m,n,k,R,theta,phi))
            values_calculated += value
        total_result += values_calculated
    return total_result

# total_result = e_field_sint(k=2*np.pi,
#              R=1,
#              acoeff=make_dummy_a_coef(N),
#              bcoeff=make_dummy_b_coef(N),
#              theta=0,
#              phi=0)
# print(total_result)

