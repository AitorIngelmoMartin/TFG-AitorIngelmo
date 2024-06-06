import numpy as np
import pandas as pd
from pandas import read_csv

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)

xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z
data = f(xg, yg, zg)    


# Definir las cabeceras manualmente
column_names = ['x', 'y', 'z', 'Evalue']

# Leer el archivo, saltando las líneas que comienzan con '%'
data_readed = read_csv('NFtoFF-Trasnformacion JL/Codigos/transformation/file_to_read.txt', comment='%', delim_whitespace=True, names=column_names, skiprows=1)
data_readed_woth_no_nan = data_readed.dropna()
print("No With no NaN \n",data_readed_woth_no_nan)
# Eliminar duplicados
# df = pd.unique(data_readed.values.ravel())   
# print("No duplicates \n",df)