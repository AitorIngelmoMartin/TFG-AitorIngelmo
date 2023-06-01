from scipy.interpolate import RegularGridInterpolator
import numpy as np
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
data = f(xg, yg, zg)

interp = RegularGridInterpolator((x, y, z), data)

pts = np.array([[2.1, 6.2, 8.3],
                [3.3, 5.2, 7.1]])
salida_tras_interpretar = interp(pts)
print(salida_tras_interpretar)