import funciones
import numpy as np
from math import cos, sin
from scipy import integrate

# Define las constantes
lambda_ = 1
r = 1 * lambda_
I0 = 1
l = lambda_ / 50
k = 2 * np.pi / lambda_
eta = 120 * np.pi
numberOfModes = 5

# Define las funciones a integrar
def Edipoler(r, theta, k, eta, l, I0):
    # Ejemplo simplificado; reemplaza con tu implementación
   return ((eta * I0 * cos(theta))/(2*np.pi*r*r)) * (1 + 1/(1j*k*r))*np.exp(-1j*k*r)

def Edipoletheta(r, theta, k, eta, l, I0):
    # Ejemplo simplificado; reemplaza con tu implementación
    return ((1j*eta*((k*I0*l*sin(theta))/(4*np.pi*r))) * (1+(1/(1j*k*r))-(1/(k*k*r*r))))*np.exp(-1j*k*r)

# Define la función de integrando para la integral doble
def integrand_r(phi, theta, m, n):
    # return Edipoler(r, theta, k, eta, l, I0)*funciones.b_sin_function(-m,n,theta,phi)[0]
    return 0

def integrand_theta(phi, theta, m, n):
    return Edipoletheta(r, theta, k, eta, l, I0)*funciones.b_sin_function(-m,n,theta,phi)[1]

# Calcula la integral para cada par (m, n)
results = []
for n in range(1, numberOfModes + 1):
    mode_results = []
    for m in range(-n, n + 1):
        integral_component_r, error_r = integrate.dblquad(integrand_r, 0, 2 * np.pi, 0, np.pi, args=(m, n))
        integral_component_theta, error_theta = integrate.dblquad(integrand_theta, 0, 2 * np.pi, 0, np.pi, args=(m, n))
        integral_component_phi = 0
        mode_results.append([integral_component_r,integral_component_theta,integral_component_phi])
    results.append(mode_results)

# Mostrar resultados
print("Resultados de la integral:")
print(results)
