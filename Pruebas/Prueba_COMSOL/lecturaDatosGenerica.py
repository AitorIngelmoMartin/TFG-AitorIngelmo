import os
import numpy as np

# directorio de los archivos
Directorio = r'Datos_antena'

# lista de los archivos almacenados
lista_archivos = os.listdir(Directorio)

#Variables para almacenar los campos leidos
NF_30mmX = []
NF_30mmY = []
NF_40mmX = []
NF_40mmY = []   

Array_Campos = [NF_30mmX, NF_30mmY, NF_40mmX,NF_40mmY]

#A partir de qué líneas hay datos
inicio_datos = 8

#Leo todos los archivos uno a uno
for i in range(len(lista_archivos)):
    print(rf"{Directorio}\{lista_archivos[i]}")
    archivo  = open(rf"{Directorio}\{lista_archivos[i]}", 'r')

    lines   = archivo.readlines()
    rawdata = lines[inicio_datos:]

    for linea in range(len(rawdata)):
        columnas = rawdata[linea].split()
        Array_Campos[i].append(float(columnas[3]))

    print(Array_Campos[i][0])

