import numpy as np
import matplotlib.pyplot as plt

# Crear una figura y un conjunto de ejes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear datos de coordenadas (rango de -50 a 50 en x y y) con menos puntos (100)
x = np.linspace(-40, 40, 1000)
y = np.linspace(-40, 40, 1000)
x, y = np.meshgrid(x, y)

# Calcular la distancia desde el origen
r = np.sqrt(x**2 + y**2)

# Calcular la fase de onda para un frente esférico
k = 1  # Número de onda
wavelength = 2 * np.pi / k
phase_spherical = k * r

# Calcular la fase de onda para un frente plano
phase_flat = np.zeros_like(r)

# Calcular una interpolación suave entre las fases esférica y plana
num_frames = 1000
interpolation_factor = np.linspace(0, 1, num_frames)
smoothed_phase = (1 - interpolation_factor) * phase_spherical + interpolation_factor * phase_flat

# Calcular la amplitud de la onda (puede ser constante)
amplitude = np.ones_like(r)

# Crear el frente de onda suavizado
wavefront = amplitude * np.exp(1j * smoothed_phase)

# Dibujar el frente de onda en 3D
ax.plot_surface(x, y, np.real(wavefront), cmap='viridis')

# Ajustar los límites de los ejes para una mejor visualización
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.set_zlim([0, 2])  # Ajusta esto según tus necesidades

# Mostrar el gráfico
plt.show()
