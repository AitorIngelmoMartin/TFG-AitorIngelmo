"""
        DEFINICION DEL CODIGO
"""
from astropy.io import ascii
from astropy.table import Table

data = ascii.read("archivo.ASC",converters={'': int}, data_start=12)

#print(data)

# https://het.as.utexas.edu/HET/Software/Astropy-1.0/io/ascii/read.html

# aqui saca las columnas
#https://www.great-esf.eu/AstroStats13-Python/input_output/asciifiles.html

file = open('archivo.ASC', 'r')

columns = []


for line in file:
        line = line.strip()
        columns.append(line.split())
print(len(columns))
posicion = 15
for i in (range(posicion,len(columns))):
        print(columns[i])
