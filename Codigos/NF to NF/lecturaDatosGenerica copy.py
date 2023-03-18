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
Coordenadas_muestras = [CoordenadaX_40mm, CoordenadaY_40mm, CoordenadaZ_40mm,CoordenadaX_50mm,CoordenadaY_50mm,CoordenadaZ_50mm]
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
        Coordenadas_muestras[i].append(float(columnas[1]))

    print(Array_Campos[i][0])
    print(Coordenadas_muestras[i][0])

"""
    CALCULO VARIACIÓN FASE CAMPO CERCANO A CERCANO
"""
print("\n")

f = 1.575*1e9

#FFT del campo cercano en Z0
campo_Z0= NF_40mmY
Delta_X =[]
numero_muestras_X = len(CoordenadaX_40mm)

for i in range(numero_muestras_X-1):
    Delta_X.append(abs(CoordenadaX_40mm[i+1]-CoordenadaX_40mm[i]))

#print("El valor del campoX es:",campo_Z0,"\n")

ckz1 = np.fft.fft(campo_Z0)

Z1 = 50
Z0 = 40

Nx = len(campo_Z0)
Lx = max(Coordenadas_muestras[0]) + (-1*min(Coordenadas_muestras[0]))
Delta_X = Lx/Nx

k0 = (2*np.pi)/(3e8/f)

print(k0)
DeltaKx = np.sqrt(k0**2 - (2*np.pi*Delta_X)**2) 
ckz2    =np.exp(-1j*DeltaKx*(Z1-Z0))



#Cálculo del campo
ckz = ckz1*ckz2

campo = np.fft.ifft(ckz)


#print("El campo en 50mm es",NF_50mmX,"\n")
print("El valor resultante es:",abs(campo),"\n")