import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la onda
k = 1  # Número de onda

# Crear una cuadrícula de coordenadas
N = 1000
x = np.linspace(0, 39, N)
y = np.linspace(0, 39, N)
x, y = np.meshgrid(x, y)

# Calcular la distancia radial desde el epicentro
r = np.sqrt(x**2 + y**2)

# Calcular la fase de onda para un frente esférico
phase_spherical = k * r

# Calcular la fase de onda para un frente plano
phase_flat = np.zeros_like(r)

# Calcular una interpolación suave entre las fases esférica y plana
interpolation_factor = np.exp(-k * r)
smoothed_phase = (1 - interpolation_factor) * phase_spherical + interpolation_factor * phase_flat

# Calcular la amplitud de la onda (puede ser constante)
amplitude = np.ones_like(r)

# Crear el frente de onda suavizado
wavefront = amplitude * np.exp(1j * smoothed_phase)

# Dibujar el frente de onda en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, np.real(wavefront), cmap='viridis')

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Amplitud')

# Título del gráfico
plt.title('Frente de Ondas Esférico Isotrópico que se Vuelve Plano en 3D')
plt.show()
