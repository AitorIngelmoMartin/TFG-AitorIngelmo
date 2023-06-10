import numpy as np
import numpy.ma as ma
from scipy import special
#from scipy.special.basic import erf_zeros
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as image
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns
import time

mu0 = 4*np.pi*1e-7
c0 = 299792458
raw_data = 9
k0 = 2
eps = 1e-6

pauseinterval = 0.01

Nx = 100
Ny = 100
delta_x = 1
delta_kx = 2*np.pi/(delta_x*Nx)

kx_array = np.arange(-Nx/2,Nx/2)*delta_kx #jlap
ky_array = np.arange(-Ny/2,Ny/2)*delta_kx #jlap
kx_mesh, ky_mesh = np.meshgrid(kx_array, ky_array)
matrix_test = np.sin(kx_mesh**2+ky_mesh**2)/(kx_mesh**2+ky_mesh**2+eps)
"""
Hay que ver si el rango es -nx/2 hasta Nx/2-1 o si este "-1" se hace en el original. Si es -nx/2+1

M=(N-1)/2
range(-M,M+1,1)
"""


# Generación de los valores de los ángulos
theta = np.linspace(0, 2 * np.pi, Nx)
phi   = np.linspace(0, 2 * np.pi, Nx)
# Generación de la malla
phi_mesh, theta_mesh = np.meshgrid(phi, theta)
#Interpolación

#Cálculo de los Kx y Ky correspondientes
kx = k0*np.sin(theta_mesh)*np.cos(phi_mesh)
ky = k0*np.sin(theta_mesh)*np.sin(phi_mesh)

kz = k0*np.cos(theta_mesh)

#Estos valores corresponden a la "Ad(kx,ky)"
Ehatx = np.fft.fft2((matrix_test))
Ehatx_interp_func = RegularGridInterpolator((kx_array, ky_array), Ehatx)  #jlap
kxy = np.array([(kx[i,j],ky[i,j]) for i in range(kx.shape[0]) for j in range(ky.shape[0])])
print(len(kx[11]))
print(kxy[102])
Ehatx_interp_data = Ehatx_interp_func(kxy)

print(Ehatx_interp_data.shape[0])
Ehatx_interp_data_ordenado = np.random.random((Nx, Nx)) + np.random.random((Nx, Nx)) * 1j

posicion = 0
for id_array in range (Nx):
    for posicion_array in range (Nx):
        Ehatx_interp_data_ordenado[id_array,posicion_array] = Ehatx_interp_data[posicion,]
        posicion += 1
EhatxReconstruido = np.fft.ifft2(Ehatx_interp_data_ordenado)
pass