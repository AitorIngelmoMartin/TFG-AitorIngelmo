
file  = open('archivo.ASC', 'r')

Theta    = []
Phi      = []
EPH_REAL = []
EPH_IMG  = []
EPV_REAL = []
EPV_IMG  = []

lines   = file.readlines()
rawdata = lines[15:]

for i in range(len(rawdata)):
    columnas = rawdata[i].split()
    Theta.append(float(columnas[0]))
    Phi.append(float(columnas[1]))
    EPH_REAL.append(float(columnas[2]))
    EPH_IMG.append(float(columnas[3]))
    EPV_REAL.append(float(columnas[4]))
    EPV_IMG.append(float(columnas[5]))

print(EPH_REAL)

lista = [Theta, Phi]

print(lista[1][:])