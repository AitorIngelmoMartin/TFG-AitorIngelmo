"""File used to test the orthogonality of our functions"""
import funciones
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

# Global variables
threshold = 1e-10

def e_dipole_r_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((eta * Io * cos(theta))/(2*np.pi*r*r)) * (1 + 1/(1j*k*r))*np.exp(-1j*k*r)
# print(e_dipole_r_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

def e_dipole_theta_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((1j*eta*((k*Io*l*sin(theta))/(4*np.pi*r))) * (1+(1/(1j*k*r))-(1/(k*k*r*r))))*np.exp(-1j*k*r)
# print(e_dipole_theta_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

def e_dipole_far_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric farfield using Balanis 4.10a and 4.10b"""
    return (1j*eta*((k*Io*l*sin(theta))/(4*np.pi*r)))*np.exp(-1j*k*r)
# print(e_dipole_far_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

# PolarPlot de la componente r del campo de un dipolo
def make_polarplot_of_EdipoloR():
    """Function used to draw the polar plor of EdipoloR from [-pi,pi]"""
    theta_values = np.linspace(-np.pi, np.pi, num=100)
    result_array = []
    for theta in theta_values:
        total_result = e_dipole_r_field(
            r=150,
            theta=theta,
            k=2*np.pi,
            eta=120*np.pi,
            l=0.05,
            Io=1)
        result_array.append(np.abs(total_result))

    # Crear el gráfico polar
    plt.figure(figsize=(8,8))
    plt.polar(theta_values, result_array)

    # Mostrar el gráfico
    plt.show()
# make_polarplot_of_EdipoloR()

# PolarPlot de la componente theta del campo de un dipolo
def make_polarplot_of_EdipoloTheta():
    """Function used to draw the polar plor of EdipoloTheta from [-pi,pi]"""
    theta_values = np.linspace(-np.pi, np.pi, num=100)
    result_array = []
    for theta in theta_values:
        total_result = e_dipole_theta_field(
            r=0.25,
            theta=theta,
            k=2*np.pi,
            eta=120*np.pi,
            l=0.05,
            Io=1)
        result_array.append(np.abs(total_result))

    # Crear el gráfico polar
    plt.figure(figsize=(8,8))
    plt.polar(theta_values, result_array)

    # Mostrar el gráfico
    plt.show()
# make_polarplot_of_EdipoloTheta()

# PolarPlot de la componente theta del campo de un dipolo
def make_polarplot_of_EdipoloTheta():
    """Function used to draw the polar plor of EdipoloTheta from [-pi,pi]"""
    theta_values = np.linspace(-np.pi, np.pi, num=100)
    result_array_theta = []
    result_array_far_field = []
    for theta in theta_values:
        theta_result = e_dipole_theta_field(
            r=0.25,
            theta=theta,
            k=2*np.pi,
            eta=120*np.pi,
            l=0.05,
            Io=1)
        result_array_theta.append(np.abs(theta_result))

        far_field_result = e_dipole_far_field(
            r=0.25,
            theta=theta,
            k=2*np.pi,
            eta=120*np.pi,
            l=0.05,
            Io=1)
        result_array_far_field.append(np.abs(far_field_result))

    # Mostrar el gráfico
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.polar(theta_values, result_array_theta)

    plt.polar(theta_values, result_array_far_field)

    # Mostrar el gráfico
    plt.show()
# make_polarplot_of_EdipoloTheta()

#####################
#### Verificación del campo del Dipolo cambiando Nintegrate por sumatorios
#####################

theta_array = np.arange(-60, 61, 1)
# print(len(theta_values))

def calculate_emn_from_dipole_field(number_of_points: int, m: int, n: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a single value of emn from our Dipole field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    total_result = 0
    for theta in theta_values:
        for phi in phi_values:
            # total_result += [e_dipole_r_field(r, theta, k, eta, l, Io),e_dipole_theta_field(r,theta,k,eta,l,Io),0]*funciones.b_sin_function(-m,n,theta,phi)
            total_result += np.array([0,e_dipole_theta_field(r,theta,k,eta,l,Io),0])*funciones.b_sin_function(-m,n,theta,phi)
    return total_result*delta_theta*delta_phi

# emn_dipole = calculate_emn_from_dipole_field(
#     number_of_points=1000,
#     m=0,
#     n=3,
#     r=1,
#     k=2*np.pi,
#     eta=120*np.pi,
#     l=1/50,
#     Io=1)
# print(emn_dipole)
# −0.0000000000367656364−0.000000000223229146j
# -0.0000353043 - 0.000216205 I

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

def calculate_gmn_from_dipole_field(number_of_points: int, m: int, n: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a single value of gmn from our Dipole field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    total_result = 0
    for theta in theta_values:
        for phi in phi_values:
            # total_result += [e_dipole_r_field(r, theta, k, eta, l, Io),e_dipole_theta_field(r,theta,k,eta,l,Io),0]*funciones.c_sin_function(-m,n,theta,phi)
            total_result += np.array([0,e_dipole_theta_field(r,theta,k,eta,l,Io),0])*funciones.c_sin_function(-m,n,theta,phi)
    return total_result*delta_theta*delta_phi

# gmn_dipole = calculate_gmn_from_dipole_field(
#     number_of_points=1000,
#     m=0,
#     n=1,
#     r=1,
#     k=2*np.pi,
#     eta=120*np.pi,
#     l=1/50,
#     Io=1)
# print(gmn_dipole)

def calculate_gmn_data_from_dipole_field(number_of_points: int, number_of_modes: int, r: int, k: int, eta: int, l: int, Io: int):
    """Function used to obtain a all values of gmn from our Dipole field"""
    total_result = []
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            emn_dipole = calculate_gmn_from_dipole_field(
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

def calculate_bmnffcoef_from_emn(number_of_modes: int, emn: list, r: int, k: int):
    """Function used to obtain a all values of bmnffcoef from bmn"""
    total_result = []
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            b_coef_value = funciones.b_coef_function(e_data=emn,
                                           m=m,
                                           n=n,
                                           k=k,
                                           R=r)
            if abs(b_coef_value) < threshold:
                b_coef_value = 0.0
            value_calculated.append(b_coef_value)
        total_result.append(value_calculated)
    return total_result

# dummy_emn = [[0,-5.02655 - 30.7828j,0]]
# b_coef = calculate_bmnffcoef_from_emn(
#     number_of_modes=1,
#     emn=dummy_emn,
#     r=1,
#     k=2*np.pi)
# print(b_coef)

def calculate_amnffcoef_from_gmn(number_of_modes: int, gmn: list, r: int, k: int):
    """Function used to obtain a all values of Amnffcoef from gmn"""
    total_result = []
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            b_coef_value = funciones.a_coef_function(g_data=gmn,
                                           m=m,
                                           n=n,
                                           k=k,
                                           R=r)
            if abs(b_coef_value) < threshold:
                b_coef_value = 0.0
            value_calculated.append(b_coef_value)
        total_result.append(value_calculated)
    return total_result

# dummy_gmn = [[0,0,0]]
# a_coef = calculate_amnffcoef_from_gmn(
#     number_of_modes=1,
#     gmn=dummy_gmn,
#     r=1,
#     k=2*np.pi)
# print(a_coef)

def calculate_far_field(number_of_modes: int, a_coef_value: list, b_coef_value: list, r: int, k: int, theta: int, phi: int, w: int, t: int):
    """Function used to obtainthe far field"""
    total_result = 0
    
    for n in range(1, number_of_modes + 1):
        for m in range(-n, n + 1):
            far_field_value = ((2*n+1)/(n*(n+1)))*(a_coef_value[n-1][n+m]*funciones.m_function_far_field_aproximation(m,n,k,r,theta,phi)+b_coef_value[n-1][n+m]*funciones.n_function_far_field_aproximation(m,n,k,r,theta,phi))
            # if abs(far_field_value) < threshold:
            #     far_field_value = 0.0
            total_result += far_field_value
    return total_result*np.exp(1j*w*t)

a_coef_dummy = [[0.0, 0.0, 0.0]]
b_coef_dummy = [[0.0, (43.332568590557884-14.539313370261027j), 0.0]]
ff_calculated = calculate_far_field(
    number_of_modes=1,
    a_coef_value=a_coef_dummy,
    b_coef_value=b_coef_dummy,
    r=1,
    k=2*np.pi,
    theta=1,
    phi=1,
    w=1,
    t=1)
print(ff_calculated)

def dipole_e_theta_far_field(k , I, l, theta, r):
    return 1j*(120*np.pi)*k*I*l*sin(theta)*(1/(4*np.pi*r))*np.exp(-1j*k*r)
print([0,dipole_e_theta_far_field(2*np.pi,1,1/50,1,1),0])