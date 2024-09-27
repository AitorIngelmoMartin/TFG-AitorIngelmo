__author__ = "joseluis"

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D

#VARIABLES GLOBALES
mu0           = 4*np.pi*1e-7
c0            = 299792458
raw_data      = 8
pauseinterval = 0.01

k0 ,delta_x,L_x,delta_y,L_y,coordz = 0,0,0,0,0,0
flow_config = { 
    'directory':r'NF to FF/leer polares',
    'files_in_directory': [
        'polar_plot.txt'
    ],
    'file_type':['Enorm'],
    'work_mode':'NFtoFF',      
    'shape':[100,100],
    'freq':1.575e9,
    'length_unit':1e-3,
    'res':2e-3
}

def input_data_process(flow_config):
    """Function used to read data from the JSON file."""
    global k0
    try:
        fields,files = {}, {}     
        if len(flow_config['file_type']) == len(flow_config['files_in_directory']):
            for file_type in range(len(flow_config['file_type'])):
                files.update({flow_config['file_type'][file_type]:f"{flow_config['directory']}//{flow_config['files_in_directory'][file_type]}"})
        else:
            raise Exception("El número de ficheros en el directorio es distinto a la cantidad de ficheros introducido")
        fields.update({'files':files})
        fields.update({'file_type':flow_config['file_type']})
        fields.update({"datavalues":{}})
        fields.update({"lines":{}})
        fields.update({"zValueplane":{}}) 
        fields.update({"zValueMaskedplane":{}}) 
        fields.update({"zValueZeroedplane":{}})
        fields.update({'fields_transformed':{}})
        fields.update({'quantitative_comparison':{}})          
        fields.update({'shape':flow_config['shape']})
        fields.update({'freq':flow_config['freq']})    
        fields.update({'length_unit':flow_config['length_unit']})
        fields.update({'res':flow_config['res']})
        k0 = 2*np.pi*flow_config['freq']/c0
    except Exception as exc:
        print(f"ERROR:{exc}")

    return fields
def read_data(fields,read_type='all'):
    """
    Este método realiza la lectura de todos o parte de los ficheros de salida de Comsol 
    para las componentes del campo eléctrico.
        "fields": Es el diccionario principal sobre el que estamos trabajando
        "read_type": Es una variable opcional empleada para definir el modo de lectura.
                        Por defecto lee todos los ficheros. Pero podemos especificarle que
                        lea algunos en concreto si la igualamos a un diccionario que contenga 
                        los ficheros a leer.                  
    """
    if read_type == 'all':
        for file_type, file_path in fields['files'].items():
            with open(file_path) as file:
                fields['lines'][file_type] = file.readlines()
     
def extract_matrix_data(fields):
    """
    Este método almacena en arrays los datos de los ficheros de Comsol leídos previamente con read_data             
    """
    global delta_x,L_x,delta_y,L_y,coordz

    for file_type in fields['lines'].keys(): #file_type
        datatype = file_type
        fields['datavalues'][datatype] = fields['lines'][file_type][raw_data:]

def represent_polar_diagram(plotnumber: int, r: list, theta: list, field):
    """Function used to generate polar plots"""

    # Crear el gráfico polar
    plt.figure(plotnumber)
    plt.polar(r,np.radians(theta))

    # Mostrar el gráfico
    plt.show()

if __name__ == '__main__':
    plt.close('all')
    #try:
    fields = input_data_process(flow_config)

    read_data(fields)
   
    extract_matrix_data(fields)
    # print(fields['datavalues']['Enorm'])
    array = np.genfromtxt('NF TO FF/leer polares/datos.txt')

    #Plots diagramas de radiación
    represent_polar_diagram(10,r=array[:,0][:],theta=array[:,1][:], field = array)


