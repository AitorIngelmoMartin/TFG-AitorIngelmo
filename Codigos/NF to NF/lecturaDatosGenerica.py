import os
import numpy as np

# directorio de los archivos
Directorio = r'Datos_antena'

# lista de los archivos almacenados
lista_archivos = os.listdir(Directorio)

#Variables para almacenar los campos leidos
NF_40mmX = []
NF_40mmY = []
NF_40mmZ = []
NF_50mmX = []
NF_50mmY = []
NF_50mmZ = []

CoordenadaX_40mm = []
CoordenadaY_40mm = []
CoordenadaZ_40mm = []
CoordenadaX_50mm = []
CoordenadaY_50mm = []
CoordenadaZ_50mm = []

NF_40mmNORMA = []
NF_50mmNORMA = []

Array_Campos   = [NF_40mmX, NF_40mmY, NF_40mmZ,NF_50mmX,NF_50mmY,NF_50mmZ]
Coordenadas_muestras = [CoordenadaX_40mm, CoordenadaY_40mm]

#A partir de qué líneas hay datos
inicio_datos = 8

#Leo todos los archivos uno a uno
for i in range(len(lista_archivos)):
    print(rf"{Directorio}\{lista_archivos[i]}")
    archivo  = open(rf"{Directorio}\{lista_archivos[i]}", 'r')

    lines   = archivo.readlines()
    rawdata = lines[inicio_datos:]

    for linea_datos in range(len(rawdata)):
        columnas = rawdata[linea_datos].split()
        Array_Campos[i].append(float(columnas[3]))

archivo_coordenadas  = open(rf"Datos_antena\NF_40mmX.txt", 'r')

lines   = archivo_coordenadas.readlines()
rawdata = lines[inicio_datos:]

for linea_datos in range(len(rawdata)):
    columnas = rawdata[linea_datos].split()
    Coordenadas_muestras[0].append(float(columnas[0]))
    Coordenadas_muestras[1].append(float(columnas[1]))

print(Coordenadas_muestras[1][:])
"""
    CALCULO VARIACIÓN FASE CAMPO CERCANO A CERCANO
"""
print("\n")

f = 1.575*1e9

#FFT del campo cercano en Z0
campo_Z0= NF_40mmX

# Delta por el método de resta entre dos muestras consecutivas.
Delta_X = abs(CoordenadaX_40mm[1] - CoordenadaX_40mm[0])
Delta_Y = abs(CoordenadaY_40mm[1] - CoordenadaY_40mm[0])


ckz1 = np.fft.fft(campo_Z0)

Z1 = 50
Z0 = 40

Nx = len(campo_Z0)
Lx = max(CoordenadaX_40mm) + (-1*min(CoordenadaX_40mm))
Ly = max(CoordenadaY_40mm) + (-1*min(CoordenadaY_40mm))

print(Lx, Ly)

# Delta obtenido mediante las fórmulas que aplicamos en fourier
Delta_X = Lx/Nx#abs(CoordenadaX_40mm[1] - CoordenadaX_40mm[0])
Delta_Y = Ly/Nx#abs(CoordenadaY_40mm[1] - CoordenadaY_40mm[0])

print(Delta_X, Delta_Y)

k0 = (2*np.pi)/(3e8/f)

DeltaKx = np.sqrt(k0**2 - (2*np.pi/(Delta_X))**2 - (2*np.pi/(Delta_Y))**2)


r  = np.array(DeltaKx)
#Valor de los modos en Z1
ckz2 = np.exp(-1j*r*(Z1-Z0))

#Cálculo del campo
ckz = ckz1*ckz2

campo = np.fft.ifft(ckz)


#print("El campo en 50mm es",NF_50mmX,"\n")
print("El valor resultante es:",abs(campo),"\n")