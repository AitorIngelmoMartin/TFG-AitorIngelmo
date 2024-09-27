def leerDatos(archivo, posicion_datos: int,numero_variables: int):
    lista_datos = []
    for i in range(numero_variables):
        lista_datos.append([])
    lines   = archivo.readlines()
    rawdata = lines[posicion_datos:]   
    for i in range(len(rawdata)):
        columnas = rawdata[i].split()
        for j in range(numero_variables):
            lista_datos[:][j].append(float(columnas[j]))
    return lista_datos

file  = open('archivo.ASC', 'r')

lista_datos = leerDatos(file,15,6)

Theta = lista_datos[1][:]

print(Theta)

