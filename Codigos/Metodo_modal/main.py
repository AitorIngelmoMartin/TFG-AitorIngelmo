from cmath import pi
from math import cos, e, sin
from numpy import fft, roll, arange,exp
freq = 1.2e9
landa = (3e8)/freq

# Distancia a la que quiero el campo lejanose mide. Expresada en metros.
Distancia = 12

# Numero de onda en espacio libre
K = (2*pi)/landa

# Leo el campo complejo para cada punto
Campo = [4,7,2]
X = [1,2,3]
Y = [7,4,6]
Theta = [1,4,5]
Phi   = [5,4,7]

N = len(Campo)
print("El valor de N es:",N)

M = (N-1)/2 #En mi caso M=N, ¿no?
print("El valor de M es:",M)

ckz1 = fft.fft(Campo)

N = len(Theta) 
r = 7
coordenada_X  = [0,1,2,3,4,5]
delta_X       = coordenada_X[1]-coordenada_X[0] 
Campo = [25,50,51,45,54,1,1,4,4,2,5,6]
#Modos en Ak

print("Los modos en Z son:\n",ckz1)
kvec = roll(arange(-M,M+1,1),int(M)+1)
print(kvec)
Zvalue1 = 5
Zvalue0 = 2

ckz2 = exp(-1j*r*(Zvalue1-Zvalue0))
field = fft.fft(ckz2)

""""
Funcion_espectral_onda_plana = 0
for i in range(N):
    Kx = K * sin(Theta[i])*cos(Phi[i])
    Ky = K * sin(Theta[i])*sin(Phi[i])
    Kz = K*cos(Theta[i])
    Funcion_espectral_onda_plana += Campo[i]*(e**(Kx*X[i]))*(e**(Ky*Y[i]))

Funcion_espectral_onda_plana = (1/(Funcion_espectral_onda_plana*Funcion_espectral_onda_plana))*Funcion_espectral_onda_plana

E_campo_lejano = i*((e**(-i*K*Distancia))/Distancia)*Kz*Funcion_espectral_onda_plana
"""