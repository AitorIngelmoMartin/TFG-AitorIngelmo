import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Puntos de origen y valores asociados
points = np.random.rand(100, 2)  # Puntos 2D aleatorios
values = np.sin(points[:, 0]) * np.cos(points[:, 1])  # Valores asociados

# Puntos de destino en una malla regular
xi = np.linspace(0, 1, 10)
yi = np.linspace(0, 1, 10)
xi_mesh, yi_mesh = np.meshgrid(xi, yi)
xi_points = np.vstack((xi_mesh.flatten(), yi_mesh.flatten())).T

# Interpolación de los valores en la malla regular
zi = griddata(points, values, xi_points, method='linear')

# Gráfico de los resultados
plt.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', label='Puntos de origen')
plt.scatter(xi_points[:, 0], xi_points[:, 1], c=zi, cmap='viridis', marker='s', label='Puntos interpolados')
plt.legend()
plt.colorbar()
plt.show()
