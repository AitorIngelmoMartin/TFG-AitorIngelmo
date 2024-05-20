"""File used to test the orthogonality of our functions"""
import funciones
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

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
    """Function used to test the orthogonality of Z1"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append((1/n)**np.abs(m))
        total_result.append(value_calculated)
    # print(total_result)

# make_dummy_a_coef(5)

# Calculo de Bmncoef
def make_dummy_b_coef(N):
    """Function used to test the orthogonality of Z1"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append((1/(n*n))**np.abs(m))
        total_result.append(value_calculated)
    # print(total_result)

# make_dummy_b_coef(5)

# Valores planos de Amncoef y Bmncoef hardcodeados para ahorrar cálculos
a_coeff_dummy = [[1.0, 1.0, 1.0], [0.25, 0.5, 1.0, 0.5, 0.25], [0.03703703703703703, 0.1111111111111111, 0.3333333333333333, 1.0, 0.3333333333333333, 0.1111111111111111, 0.03703703703703703], [0.00390625, 0.015625, 0.0625, 0.25, 1.0, 0.25, 0.0625, 0.015625, 0.00390625], [0.0003200000000000001, 0.0016000000000000003, 0.008000000000000002, 0.04000000000000001, 0.2, 1.0, 0.2, 0.04000000000000001, 0.008000000000000002, 0.0016000000000000003, 0.0003200000000000001]]
b_coeff_dummy = [[1.0, 1.0, 1.0], [0.0625, 0.25, 1.0, 0.25, 0.0625], [0.001371742112482853, 0.012345679012345678, 0.1111111111111111, 1.0, 0.1111111111111111, 0.012345679012345678, 0.001371742112482853], [1.52587890625e-05, 0.000244140625, 0.00390625, 0.0625, 1.0, 0.0625, 0.00390625, 0.000244140625, 1.52587890625e-05], [1.0240000000000002e-07, 2.56e-06, 6.400000000000001e-05, 0.0016, 0.04, 1.0, 0.04, 0.0016, 6.400000000000001e-05, 2.56e-06, 1.0240000000000002e-07]]

# Comprobación del cálculo de EfieldSint
total_result = funciones.e_field_sint(k=2*np.pi,
             R=1,
             acoeff=a_coeff_dummy,
             bcoeff=b_coeff_dummy,
             theta=0,
             phi=0)
# print(total_result)

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
print(e_dipole_r_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

def e_dipole_theta_field(r: int, theta: int, k: int, eta: int, l: int, Io: int):
    """Function used to calculate the electric field using Balanis 4.10a and 4.10b"""
    return ((1j*eta*((k*Io*l*sin(theta))/(4*np.pi*r))) * (1+(1/(1j*k*r))-(1/(k*k*r*r))))*np.exp(-1j*k*r)
print(e_dipole_theta_field(r=150,k=2*np.pi,eta=120*np.pi,Io=1,l=0.05,theta=5))

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
            total_result += [e_dipole_r_field(r, theta, k, eta, l, Io),e_dipole_theta_field(r,theta,k,eta,l,Io),0]*funciones.b_function(-m,n,theta,phi)*sin(theta)*delta_theta*delta_phi
    return total_result

# emn_dipole = calculate_emn_from_dipole_field(
#     number_of_points=1000,
#     m=0,
#     n=1,
#     r=1,
#     k=2*np.pi,
#     eta=120*np.pi,
#     l=1/50,
#     Io=1)
# print(emn_dipole)
# -5.03157983-30.81354763j
# -5.02655 - 30.7828 I
