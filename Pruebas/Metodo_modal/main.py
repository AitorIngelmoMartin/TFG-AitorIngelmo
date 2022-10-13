from cmath import pi
from math import cos, e, sin

freq = 1.2e9
landa = (3e8)/freq

# Distancia a la que quiero el campo lejanose mide. Expresada en metros.
Distancia = 12

# Numero de onda en espacio libre
K = (2*pi)/landa

# Leo el campo complejo para cada punto
Campo = []
X = Y = Theta = Phi = []

numero_muestras = 78
Funcion_espectral_onda_plana = 0
for i in range(numero_muestras):
    Kx = K * sin(Theta[i])*cos(Phi[i])
    Ky = K * sin(Theta[i])*sin(Phi[i])
    Kz = K*cos(Theta[i])
    Funcion_espectral_onda_plana += Campo[i]*(e**(Kx*X[i]))*(e**(Ky*Y[i]))

Funcion_espectral_onda_plana = (1/(Funcion_espectral_onda_plana*Funcion_espectral_onda_plana))*Funcion_espectral_onda_plana

E_campo_lejano = i*((e**(-i*K*Distancia))/Distancia)*Kz*Funcion_espectral_onda_plana