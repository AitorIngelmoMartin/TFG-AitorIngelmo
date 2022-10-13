import numpy as np


file  = open('archivo.ASC', 'r')
Theta = Phi = EPH_REAL = EPH_IMG = EPV_REAL = EPV_IMG = []
Theta = np.array
lines = file.readlines()
rawdata = lines[15:]

for i in range(len(rawdata)):
    columnas = rawdata[i].split()
    Theta.append(columnas[0])
datos_leidos = np.array(columnas)
print(Theta)


