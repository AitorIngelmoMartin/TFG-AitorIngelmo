import pandas as pd
import numpy as np
import numpy.ma as ma

# Ruta del archivo
file_path = 'NFtoFF-Trasnformacion JL/Codigos/transformation/microstrip_patch_antenna_Ex.txt'

# Definir las cabeceras manualmente
column_names = ['x', 'y', 'z', 'Evalue']

# Leer el archivo, saltando las líneas que comienzan con '%'
data = pd.read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
data['Evalue'] = data['Evalue'].apply(lambda x: complex(x.replace('i', 'j')) if 'i' in str(x) else float(x))

if data is not None:
    r = np.ma.masked_invalid(data['x'].to_numpy())
    theta = np.ma.masked_invalid(data['y'].to_numpy())
    phi = np.ma.masked_invalid(data['z'].to_numpy())
    field_value = np.nan_to_num(data['Evalue'].to_numpy(), nan=0)
    field = {
        "r_coordinate":r,
        "theta_coordinate":theta,
        "phi_coordinate":phi,
        "field_value":field_value,
    }
    print(field['r_coordinate'][6])
