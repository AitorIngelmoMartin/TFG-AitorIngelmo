import numpy as np
Theta = [1,2,3,4,5,6,7,8,9]

N = len(Theta) 
print("El valor de N es:",N)

M = (N-1)/2 #En mi caso M=N, ¿no?
print("El valor de M es:",M)

coordenada_X  = [0,1,2,3,4,5]
delta_X       = coordenada_X[1]-coordenada_X[0] # = L/N
Campo = [25,50,51,45,54,1,1,4,4,2,5,6]
#Modos en Ak
ckz1 = np.fft.fft(Campo)
print("Los modos en Z son:\n",ckz1)
kvec = np.roll(np.arange(-M,M+1,1),int(M)+1)
print(kvec)
Zvalue1 = 5
Zvalue0 = 2
r = 
ckz2 = np.exp(-1j*r*(Zvalue1-Zvalue0))