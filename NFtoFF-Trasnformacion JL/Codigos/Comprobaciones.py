"""File used to test the orthogonality of our functions"""
import funciones
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

# Global variables
threshold = 1e-10

#####################
#### EfieldSint
#####################

# Print de la matriz triangular obtenida al cacular Z1
def test_z1_matrix(N):
    """Function used to test the orthogonality of Z1"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append(funciones.z1(m,n))
        total_result.append(value_calculated)
    print(total_result)

# Calculo de Amncoef
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
# make_dummy_b_coef(5)

# Valores planos de Amncoef y Bmncoef hardcodeados para ahorrar cálculos
a_coeff_dummy = [[1.0, 1.0, 1.0], [0.25, 0.5, 1.0, 0.5, 0.25], [0.03703703703703703, 0.1111111111111111, 0.3333333333333333, 1.0, 0.3333333333333333, 0.1111111111111111, 0.03703703703703703], [0.00390625, 0.015625, 0.0625, 0.25, 1.0, 0.25, 0.0625, 0.015625, 0.00390625], [0.0003200000000000001, 0.0016000000000000003, 0.008000000000000002, 0.04000000000000001, 0.2, 1.0, 0.2, 0.04000000000000001, 0.008000000000000002, 0.0016000000000000003, 0.0003200000000000001]]
b_coeff_dummy = [[1.0, 1.0, 1.0], [0.0625, 0.25, 1.0, 0.25, 0.0625], [0.001371742112482853, 0.012345679012345678, 0.1111111111111111, 1.0, 0.1111111111111111, 0.012345679012345678, 0.001371742112482853], [1.52587890625e-05, 0.000244140625, 0.00390625, 0.0625, 1.0, 0.0625, 0.00390625, 0.000244140625, 1.52587890625e-05], [1.0240000000000002e-07, 2.56e-06, 6.400000000000001e-05, 0.0016, 0.04, 1.0, 0.04, 0.0016, 6.400000000000001e-05, 2.56e-06, 1.0240000000000002e-07]]


a_coeff_dummy = [[1.0, 1.0, 1.0]]
b_coeff_dummy = [[1.0, 1.0, 1.0]]
# Comprobación del cálculo de EfieldSint
# total_result = funciones.e_field_sint(k=2*np.pi,
#              R=1,
#              acoeff=a_coeff_dummy,
#              bcoeff=b_coeff_dummy,
#              theta=0,
#              phi=0)
# # print(total_result)

def calculate_emn_from_EfieldSint(number_of_points: int, m: int, n: int, r: int, k: int, N : int):
    """Function used to obtain a single value of emn from our sintetic field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    acoeff = make_dummy_a_coef(N)
    bcoeff = make_dummy_b_coef(N)

    total_result = 0
    for theta in theta_values:
        for phi in phi_values:
            total_result += funciones.e_field_sint(k=k, R=r, acoeff=acoeff,bcoeff=bcoeff,theta=0,phi=0)*funciones.b_sin_function(-m,n,theta,phi)*delta_theta*delta_phi
    return total_result
# print(calculate_emn_from_EfieldSint(number_of_points=500, m=0, n=1, r=1, k=2*np.pi,N=5))
# {-0.154949 + 0.951506 I, 0.109974 - 0.673486 I, -0.0778651 + 0.475551 I}

# PolarPlot de EfieldSint
def make_polarplot_of_EfieldSint():
    """Function used to draw the polar plor of EfieldSint from [-pi,pi]"""
    theta_values = np.linspace(-np.pi, np.pi, num=100)
    result_array = []
    for theta in theta_values:

        total_result = funciones.e_field_sint(  
            k=2*np.pi,
            R=50,
            acoeff=a_coeff_dummy,
            bcoeff=b_coeff_dummy,
            theta=theta,
            phi=0)
        result_array.append(np.abs(total_result))

    # Crear el gráfico polar
    plt.figure(figsize=(8,8))
    plt.polar(theta_values, result_array)

    # Mostrar el gráfico
    plt.show()
# make_polarplot_of_EfieldSint()

#####################
#### Edipolo
#####################

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

emn_dipole = calculate_emn_from_dipole_field(
    number_of_points=1000,
    m=0,
    n=3,
    r=1,
    k=2*np.pi,
    eta=120*np.pi,
    l=1/50,
    Io=1)
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

def calculate_bmnffcoef_from_dipole_field(number_of_modes: int, emn: list, r: int, k: int):
    """Function used to obtain a all values of bmnffcoef from our Dipole field"""
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
dummy_emn = [0,-5.02655 - 30.7828j,0]

b_coef = calculate_bmnffcoef_from_dipole_field(
    number_of_modes=1,
    emn=dummy_emn,
    r=1,
    k=2*np.pi)
print(b_coef)