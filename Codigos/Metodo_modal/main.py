from cmath import pi
from math import cos, e, sin
from numpy import fft, roll, arange,exp

freq = 1.2e9
landa = (3e8)/freq

# Distancia a la que quiero el campo lejanose mide. Expresada en metros.
Distancia = 12

# Numero de onda en espacio libre
K = (2*pi)/landa

# Variables leidas----
Campo = [4 +1j,7-8j,2-1j,-2+9j] #
Theta = [1,4,5]      #
Phi   = [5,4,7]      #
                     #
X = [1,2,3]          #  
Y = [7,4,6]          #
# --------------------

N = len(Campo)
print("El valor de N es:",N)

M = (N-1)/2
print("El valor de M es:",M)

N_x = len(X)
M_x = (N_x-1)/2

N_y = len(Y)
M_y = (N_y-1)/2

ckz1 = fft.fft(Campo)

r = 7 #No sé cómo sacarlo
L = 6 #No sé cómo sacarlo


#Modos en Ak
print("Los modos en Z0 son:\n",ckz1)

#kvec = roll(arange(-M,M+1,1),int(M)+1)
#print(kvec)

Zvalue1 = 5
Zvalue0 = 2

ckz2 = ((L*L)/(N_x*N_y))*exp(-1j*r*(Zvalue1-Zvalue0))
print("Los modos en Z1 son:\n",ckz2)

ckz   = ckz1*ckz2
field = fft.fft(ckz)
print("El valor del campo es:\n",field)


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