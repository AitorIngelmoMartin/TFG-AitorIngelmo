"""File used to perform the NF2FF transformation"""
import os, sys
import numpy as np
import numpy.ma as ma
from math import cos, sin
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


# Import 'funciones' module
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import funciones

#VARIABLES GLOBALES
mu0           = 4*np.pi*1e-7
c0            = 299792458
raw_data      = 9
pauseinterval = 0.01
threshold     = 1e-10
delta_x, delta_y = 0 , 0
file_type = ['Ex','Ey','Ez','normE']
k_calculated = 0

flow_config = { 
    'directory':r'c:\\Users\\aitor\\OneDrive\\Desktop\\TFG-AitorIngelmo\\NFtoFF-Trasnformacion JL\\Codigos\\transformation',
    'files_in_directory': [
        'microstrip_patch_antenna_Ex.txt',
        'microstrip_patch_antenna_Ey.txt',
        'microstrip_patch_antenna_Ez.txt',
        'microstrip_patch_antenna_normE.txt'                
    ],
    'file_type':file_type,
    'work_mode':'NFtoFF',      
    'shape':[100,100,100],
    'freq':1.575e9,
    'length_unit':1e-3,
    'res':2e-3
}

def input_data_process(flow_config: dict):
    """Function used to save inputs from our flow_config dictionary"""
    global k_calculated
    try:
        fields,files = {}, {}
        if len(flow_config['file_type']) == len(flow_config['files_in_directory']):
            for file_type in range(len(flow_config['file_type'])):
                files.update({flow_config['file_type'][file_type]:f"{flow_config['directory']}//{flow_config['files_in_directory'][file_type]}"})
        else:
            raise Exception("El número de ficheros en el directorio es distinto a la cantidad de ficheros introducido")
        fields.update({'files':files})
        fields.update({'file_type':flow_config['file_type']})
        fields.update({"field_readed":{}})
        fields.update({"lines":{}})
        fields.update({"field_readed_masked":{}})
        fields.update({'fields_transformed':{}})
        fields.update({'quantitative_comparison':{}})
        fields.update({'freq':flow_config['freq']})
        fields.update({'length_unit':flow_config['length_unit']})
        fields.update({'res':flow_config['res']})

        k_calculated = (2*np.pi * flow_config['freq'])/c0

    except Exception as exc:
        print(f"ERROR:{exc}")

    return fields

def read_data(fields):
    """Function used to read data from the files defined in 'flow_config'"""
    # Definir las cabeceras manualmente
    column_names = ['x', 'y', 'z', 'E_value']

    fields['field_readed_masked']['values_readed'] = {}
    for file_type, file_path in fields['files'].items():
        with open(file_path) as file:
            fields['lines'][file_type] = file.readlines()

            # Leer el archivo, saltando las líneas que comienzan con '%'
            data = read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
            
            data_with_no_nans = data.dropna()

            x_coordinates = data_with_no_nans['x'].to_numpy()
            y_coordinates = data_with_no_nans['y'].to_numpy()
            z_coordinates = data_with_no_nans['z'].to_numpy()
            E_value = data_with_no_nans['E_value'].to_numpy()

            E_value_str = np.array([str(val) for val in E_value])
            E_value_j = np.char.replace(E_value_str , 'i', 'j')
            complex_E_value = E_value_j.astype(complex)

            fields['field_readed_masked']['values_readed'][file_type] = {}
            fields['field_readed_masked']['values_readed'][file_type]['x_coordinates'] = x_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['y_coordinates'] = y_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['z_coordinates'] = z_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['E_value'] = complex_E_value

def export_data_into_csv(fields):
    """Function used to read data from the files defined in 'flow_config'"""
    # Definir las cabeceras manualmente
    column_names = ['x', 'y', 'z', 'E_value']

    fields['field_readed_masked']['values_readed'] = {}
    for file_type, file_path in fields['files'].items():
        file_name_without_extension = (file_path.split('/')[-1]).split('.')[0] # We get the name without the extension

        # Leer el archivo, saltando las líneas que comienzan con '%'
        data = read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
        
        data_with_no_nans = data.dropna()

        data_with_no_nans.to_csv(f"python_processed_{file_name_without_extension}.txt", sep=';', index=False)

def main():    
    fields = input_data_process(flow_config)
    
    export_data_into_csv(fields)    

    print("FIN PROGRAMA")

main()