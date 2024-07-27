from math import factorial
import numpy as np
from numpy import cos, sin
from scipy import special


def e_dipole_theta_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((1j*eta*((k*Io*l*sin(theta))/(4*np.pi*r))) * (1+(1/(1j*k*r))-(1/(k*k*r*r))))*np.exp(-1j*k*r)

def legendre_polinom(n: int, m: int, z: int):
    """Function that calculate the legendre polinom"""
    if np.abs(m) > n:
        return 0
    else:
        return special.clpmn(m, n, z, 2)[0][np.abs(m)][n]

def legendre_polinom_derived(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom"""
    return -0.5*(((n+m)*(n-m+1)*legendre_polinom(n,m-1,cos(theta))) - (legendre_polinom(n, m+1, cos(theta))))    
# print(legendre_polinom_derived(1,1,5))    

def legendre_polinom_derived_sin(n: int, m: int, theta: int):
    """Function that calculate the derivate of the legendre polinom multiplied by Sin(Theta)"""
    return sin(theta)*legendre_polinom_derived(n, m, theta)

def b_sin_function(m: int, n: int, theta: int, phi: int):
    """Function used to calculate our spherical base function Bmn*Sin(theta)"""
    return np.array([0, legendre_polinom_derived_sin(n, m, theta)*np.exp(1j*m*phi), 1j*m*legendre_polinom(n,m,cos(theta))*np.exp(1j*m*phi)])
# print(b_sin_function(2,2,5,5))

def calculate_emn_from_dipole_field(number_of_points: int, m: int, n: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a single value of emn from our Dipole field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    total_result = np.array([np.complex128(0),np.complex128(0),np.complex128(0)])
    for theta in theta_values:
        for phi in phi_values:
            # mask = value >= 10e-10
            # value[mask] = np.complex128(0)
            # value = np.where(np.abs(value) < 10e-8, np.complex128(0), value)
            total_result += np.array([0,e_dipole_theta_field(r,theta,k,eta,l,Io),0])*b_sin_function(-m,n,theta,phi)

    return total_result*delta_theta*delta_phi

emn_dipole = calculate_emn_from_dipole_field(
    number_of_points=100,
    m=0,
    n=3,
    r=1,
    k=2*np.pi,
    eta=120*np.pi,
    l=1/50,
    Io=1)
print(emn_dipole)

# Obtenido :−0.0000000000367656364−0.000000000223229146j
# Obtenido :-0.0000000005934889083833803 - 0.000000003634537662391554j
# Obtenido: −0.000000386472918−0.00000236677214j (tras arreglar <<el sumatorio)
# Esperado: -0.0000353043 - 0.000216205 I


# m = 0
# n=3
# theta = 0.0031447373909807737
# phi = 0.37736848691769287
# result = b_sin_function(-m,n,theta,phi)

