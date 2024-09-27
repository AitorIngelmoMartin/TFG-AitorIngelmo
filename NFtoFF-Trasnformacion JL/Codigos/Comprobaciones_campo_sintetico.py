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

def e_field_sint(*,k: int, R: int, acoeff: list, bcoeff: list, theta: int, phi: int):
    """Function used to obtain a sintetic Field"""
    M = len(acoeff)
    total_result = 0
    for n in range(1, M + 1):
        values_calculated = 0
        for m in range(-n, n + 1):
            value = ((2*n+1)/(n*(n+1)))*(acoeff[n-1][m+n]*funciones.m_function(m,n,k,R,theta,phi) + bcoeff[n-1][m+n]*funciones.n_function(m,n,k,R,theta,phi))
            values_calculated += value
        total_result += values_calculated
    return total_result

# Print de la matriz triangular obtenida al cacular Z1
def test_z1_matrix(N):
    """Function used to test the orthogonality of Z1"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append(funciones.z1(m,n))
        total_result.append(value_calculated)
    # print(total_result)

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
            # total_result += e_field_sint_calculated*funciones.b_sin_function(-m,n,theta,phi)*delta_theta*delta_phi
            total_result += (acoeff[n-1][m+n]*funciones.m_function(m,n,k,r,theta,phi) + bcoeff[n-1][m+n]*funciones.n_function(m,n,k,r,theta,phi))*funciones.b_sin_function(-m,n,theta,phi)
    return total_result*delta_theta*delta_phi

print(calculate_emn_from_EfieldSint(number_of_points=1500, m=0, n=1, r=1, k=2*np.pi, N=5))    
# MATEMATICA: {-0.154949 + 0.951506 I, 0.109974 - 0.673486 I, -0.0778651 + 0.475551 I}
#             [ 0.        +0.j        -0.32843936-0.9206222j  0.        +0.j       ]

# {-0.10356 + 0.634203 I, 0.0733162 - 0.448991 I, -0.0517798 + 0.317101 I}
# [0.        +0.j         0.07336505-0.44928977j 0.        +0.j        ]
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
