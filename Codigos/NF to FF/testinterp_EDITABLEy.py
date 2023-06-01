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



#Estos valores corresponden a la "Ad(kx,ky)"
Ehatx = np.fft.fft2((matrix_test))
Ehatx_interp_func = RegularGridInterpolator((kx_array, ky_array), Ehatx)  #jlap
kxy = np.array([[kx[i,j],ky[i,j]] for i in range(kx.shape[0]) for j in range(ky.shape[0])])
Ehatx_interp_data = Ehatx_interp_func(kxy)
pass