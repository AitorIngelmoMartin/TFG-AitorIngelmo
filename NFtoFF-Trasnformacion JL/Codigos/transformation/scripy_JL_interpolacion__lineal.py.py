# Vamos a trabajar en cartesianas. Tú tendrás que generar una rejilla en esféricas (porque es donde se mide
# en realidad en la cámara anecoica) y 'traducirlas' a cartesianas para hacer lo que hacemos aquí

import numpy as np
from scipy.interpolate import LinearNDInterpolator
# Definimos las coordenadas de los datos originales
x = np.linspace(0, 1, num=3)
y = np.linspace(1, 2, num=3)
z = np.linspace(2, 3, num=3)
values0 = np.random.rand(3, 3, 3) 
values0[0,0,0] = np.float('nan') 
values0[0,0,1] = np.float('nan')
indices = ~np.isnan(values0)
values = values0[indices]
Y, Z, X = np.meshgrid(y, z, x)
X = X[indices]
Y = Y[indices]
Z = Z[indices]
points = np.column_stack((X, Y, Z))
x2 = np.unique(points[:,0])
y2 = np.unique(points[:,1])
z2 = np.unique(points[:,2])
# Estas nuevas coordenadas corresponden a los puntos donde queremos obtener el valor de la interpolación.
# En tu caso serán las coordenadas no equiespaciadas en cartesianas que resultan de la traducción de la rejilla
# regular que vas a definir en esféricas. Aquí por facilidad generamos una subrejilla regular.
x1 = np.linspace(x[0], x[2], num=5)
y1 = np.linspace(y[0],y[2], num=5)
z1= np.linspace(z[0],z[2], num=5)
Y1,Z1,X1= np.meshgrid(y1,z1,x1)
values_flat = values.flatten()
interfunc = LinearNDInterpolator(points, values_flat)
interpoints = np.column_stack((X1.flatten(), Y1.flatten(), Z1.flatten())) # Este array 'interpoints' llevará la
                                                                          # traducción a cartesisanas de tus puntos
                                                                          # elegidos en coordenadas esféricas
# Vas a tener (r_i,theta_i,phi_i) con i = 1,...,M (M es el número de puntos de tu rejilla en esféricas)
# Por ejemplo:
#  + (r=10,theta = -2 deg, phi = -1 deg)
#  + (r=10,theta = -1 deg, phi = -1 deg)
#  + (r=10,theta = -0 deg, phi = -1 deg)
#  + (r=10,theta = 1 deg, phi = -1 deg)
#  + (r=10,theta = 2 deg, phi = -1 deg)
#  + (r=10,theta = -2 deg, phi = 0 deg)
#  + (r=10,theta = -1 deg, phi = 0 deg)
#  + (r=10,theta = -0 deg, phi = 0 deg)
#  + (r=10,theta = 1 deg, phi = 0 deg)
#  + (r=10,theta = 2 deg, phi = 0 deg)
#  + (r=10,theta = -2 deg, phi = 1 deg)
#  + (r=10,theta = -1 deg, phi = 1 deg)
#  + (r=10,theta = -0 deg, phi = 1 deg)
#  + (r=10,theta = 1 deg, phi = 1 deg)
#  + (r=10,theta = 2 deg, phi = 1 deg)
#  Tenemos, en este ejemplo, 5 x 3 = 15 valores, al igual que
#  en el caso del ejemplo del código, que está directamente en cartesianas, teníamos 5 x 5 x 5 = 125
# 
#  Esto iría a un array (15,3) en esféricas (np.array([[10,-2,-1],[10,-1,-1],...) que tengo que traducir 
#  en un array (15,3) con las coordenadas cartesianas correspondientes 
#  interpoints = np.array([10 sin(-2 deg) cos(-1 deg),10 sin(-2 deg) sin(-1 deg),10 cos(-2 deg)],[],...))
interpolated_values_flat = interfunc(interpoints)
# Ahora obtendrás otro array, pero este con los valores interpolados para esos puntos
# detallados arriba. Tienes que hacer esto tres veces (para Ex, Ey, Ez). Recordamos que los 'values'
# que aquí son valores aleatorios, serán en tu caso los valores leídos de los ficheros de output de Comsol.
#  y que tendrás tres: Ex, Ey, Ez. Una vez interpolados en el sistema de referencia cartesiano tienes que pasar
#  estos valores interpolados a esféricas: E_r, E_theta, E_phi, porque son estos valores los que necesitas
#  como input en la descomposición modal
interpolated_values = interpolated_values_flat.reshape(X1.shape)
coordinates_nan = interpoints[np.isnan(interpolated_values_flat)]
print("The original values are:")
print(f"x: {x},\n y: {y},\n z: {z}\n values: {values} \n\n\n")
print("The interpolated values:")
print(interpolated_values)