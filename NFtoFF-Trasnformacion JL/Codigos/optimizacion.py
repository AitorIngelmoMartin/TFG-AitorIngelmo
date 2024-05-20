
import funciones
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

def e_dipole_r_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((eta * Io * cos(theta))/(2*np.pi*r*r)) * (1 + 1/(1j*k*r))*np.exp(-1j*k*r)
# print(e_dipole_r_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

def e_dipole_theta_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((1j*eta*((k*Io*l*sin(theta))/(4*np.pi*r))) * (1+(1/(1j*k*r))-(1/(k*k*r*r))))*np.exp(-1j*k*r)
# print(e_dipole_theta_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

def calculate_emn_from_dipole_field(number_of_points: int, m: int, n: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a single value of emn from our Dipole field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    total_result = 0
    for theta in theta_values:
        for phi in phi_values:
            total_result += ([0, e_dipole_theta_field(r,theta,k,eta,l,Io), 0]*funciones.b_function(-m,n,theta,phi))*sin(theta)*delta_theta*delta_phi
    return total_result

def calculate_emn_data_from_dipole_field(number_of_points: int, number_of_modes: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a all values of emn from our Dipole field"""
    total_result = []
    threshold = 1e-10
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            emn_dipole = calculate_emn_from_dipole_field(
                number_of_points=number_of_points,
                m=m,
                n=n,
                r=r,
                k=k,
                eta=eta,
                l=l,
                Io=Io)
            dummy = sum(emn_dipole)
            if abs(dummy) < threshold:
                dummy = 0.0
            value_calculated.append(dummy)
            print(dummy)
        total_result.append(value_calculated)
    return total_result
# print(calculate_emn_data_from_dipole_field(500, 2, 1, 2*np.pi, 120*np.pi, 1/50, 1))

