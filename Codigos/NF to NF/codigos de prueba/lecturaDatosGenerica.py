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

lines   = archivo_coordenadas.readlines()
rawdata = lines[inicio_datos:]

for linea_datos in range(len(rawdata)):
    columnas = rawdata[linea_datos].split()
    Coordenadas_muestras[0].append(float(columnas[0]))
    Coordenadas_muestras[1].append(float(columnas[1]))

print(Coordenadas_muestras[1][:])
