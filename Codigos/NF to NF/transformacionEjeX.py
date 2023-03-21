import os
import numpy as np

# Directorio de los archivos
Directorio = r'Datos_antena'

# Lista de los archivos almacenados
lista_archivos = os.listdir(Directorio)

# Variables para almacenar los campos leidos
NF_30mmX = []
NF_50mmX = []

CoordenadaX_30mm = []
CoordenadaY_30mm = []

Array_Campos   = [NF_30mmX,NF_50mmX]
Coordenadas_muestras = [CoordenadaX_30mm, CoordenadaY_30mm]

# Variable que define a partir de qué linea hay datos
inicio_datos = 9

# Leo todos los archivos uno a uno
for i in range(len(lista_archivos)):
    print(rf"{Directorio}\{lista_archivos[i]}")
    archivo  = open(rf"{Directorio}\{lista_archivos[i]}", 'r')

    lines   = archivo.readlines()
    rawdata = lines[inicio_datos:]

    for linea_datos in range(len(rawdata)):
        columnas = rawdata[linea_datos].split()
        Array_Campos[i].append(complex(columnas[3].replace(" ","")))

# Leo un solo archivo de entrada para poder extraer las coordenadas.
archivo_coordenadas  = open(rf"Datos_antena\NF_30mmX.txt", 'r')

lines   = archivo_coordenadas.readlines()
rawdata = lines[inicio_datos:]

for linea_datos in range(len(rawdata)):
    columnas = rawdata[linea_datos].split()
    Coordenadas_muestras[0].append(float(columnas[0]))
    Coordenadas_muestras[1].append(float(columnas[1]))

"""
    CALCULO VARIACIÓN FASE CAMPO CERCANO A CERCANO
"""
print("\n")

f = 1.575*1e9

#FFT del campo cercano en Z1
campo_Z0= NF_30mmX
Nx = len(campo_Z0)

# Delta_X y Delta_Y son iguales al emplear el mismo intervalo de muestreo.
Delta_X = abs(CoordenadaX_30mm[1] - CoordenadaX_30mm[0])
Delta_Y = Delta_X

#Calculo los modos del campo en Z1.
ckz1 = Delta_X*Delta_X*np.fft.fft(campo_Z0)

print("La FFT del campo da lo siguiente: \n",ckz1)

Z2 = 50
Z1 = 30

k0 = (2*np.pi)/(3e8/f)

Omega = []
for mx in range(Nx):
    Kx = (2*np.pi*mx) / (Delta_X * Nx)
    Ky = Kx
    Omega.append(np.sqrt(k0*k0 - Kx*Kx - Ky*Ky)) 

#Calculo los modos del campo en Z2 a partir de los modos del campo en Z1
ckz2 = []
for i in range(Nx):
    ckz2.append(ckz1[i]*np.exp(-1j*Omega[i]*(Z2-Z1)))


Delta_Kx = (2*np.pi) / (Delta_X * Nx)
Delta_Ky = Delta_Kx

#A partir de los modos del campo en Z2, obtengo el campo en Z2
Campo_en_Z2 = Delta_X*Delta_Y*np.fft.ifft(ckz2)
print("\n El campo en Z2 vale lo siguiente: \n",Campo_en_Z2)