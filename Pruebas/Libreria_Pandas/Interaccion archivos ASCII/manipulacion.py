"""
        DEFINICION DEL CODIGO
"""
from astropy.io import ascii
from astropy.table import Table

data = ascii.read("archivo.ASC",converters={'': int}, data_start=12)

print(data)
print(type(data))


# https://het.as.utexas.edu/HET/Software/Astropy-1.0/io/ascii/read.html


# aqui saca las columnas
#https://www.great-esf.eu/AstroStats13-Python/input_output/asciifiles.html