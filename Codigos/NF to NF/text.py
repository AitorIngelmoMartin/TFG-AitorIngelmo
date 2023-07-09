import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Datos de ejemplo del diagrama de radiación
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
radiation_pattern = np.abs(np.sin(phi) * np.cos(theta))

# Crear la figura y los ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Hacer el plot 3D
ax.plot_surface(theta, phi, radiation_pattern, cmap='viridis')

# Personalizar el plot
ax.set_xlabel('Theta')
ax.set_ylabel('Phi')
ax.set_zlabel('Radiación')
ax.set_title('Diagrama de Radiación')

# Mostrar el plot
plt.show()
