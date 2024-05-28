"""File used to perform the NF2FF transformation"""
import os, sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from math import cos, sin

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
delta_x, delta_y, coordz = 0 , 0, 0
file_type = ['Ex','Ey','Ez','normE']

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

    try:
        fields,files,datavalues,lines,zValueplane,zValueMaskedplane,zValueZeroedplane = {}, {}, {}, {}, {} , {}, {}     
        if len(flow_config['file_type']) == len(flow_config['files_in_directory']):
            for file_type in range(len(flow_config['file_type'])):
                files.update({flow_config['file_type'][file_type]:f"{flow_config['directory']}//{flow_config['files_in_directory'][file_type]}"})
        else:
            raise Exception("El número de ficheros en el directorio es distinto a la cantidad de ficheros introducido")
        fields.update({'files':files})
        fields.update({'file_type':flow_config['file_type']})
        fields.update({"datavalues":datavalues})
        fields.update({"lines":lines})
        fields.update({"zValueplane":zValueplane}) 
        fields.update({"zValueMaskedplane":zValueMaskedplane}) 
        fields.update({"zValueZeroedplane":zValueZeroedplane})
        fields.update({'fields_transformed':{}})
        fields.update({'quantitative_comparison':{}})          
        fields.update({'shape':flow_config['shape']})
        fields.update({'freq':flow_config['freq']})    
        fields.update({'length_unit':flow_config['length_unit']})
        fields.update({'res':flow_config['res']})

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
    Este método almacena en arrays los datos de los ficheros de Comsol leídos previamente con readData             
    """
    global delta_x, delta_y, coordz
    #filetypes = lines.keys()
    coord_flag = None
    for file_type in fields['lines'].keys(): #file_type
        datatype     = file_type
        rawdatalines = fields['lines'][file_type][raw_data:]
        if coord_flag == None:
            coord_flag = 1
            coord      = np.array([[float(s) for k,s in zip(range(4),rawdatalines[i].split()) if k<3 ] \
                for i in range(len(rawdatalines))]).reshape(len(rawdatalines),3)
            
            coordx  = np.unique(coord[:,0])*fields['length_unit']
            delta_x = coordx[1]-coordx[0]

            coordy  = np.unique(coord[:,1])*fields['length_unit']
            delta_y = coordy[1]-coordy[0]
            coordz  = np.unique(coord[:,2])*fields['length_unit']
        fields['datavalues'][datatype] = np.array([complex(s.replace('i', 'j')) for i in range(len(rawdatalines)) \
            for k,s in zip(range(4),rawdatalines[i].split()) if k == 3])

def extract_phi_value_cut(fields,field_components,cuts_to_extract):
    """
    Este método nos permite extraer los valores del campo en un cierto número de cortes o valores de z 
        "fields": Es el diccionario que contiene los datos que estamos tratando en el programa.
        "field_components":Es el array que contiene todas las componentes que sobre las cuales 
                           queremos extraer los cortes.
        "cuts_to_extract" : Es el array que contiene los cortes (medidos en metros) que deseamos extraer.       
    """    
    numberOfCuts = len(cuts_to_extract)
    shape_0      = fields['shape'][0]
    shape_1      = fields['shape'][1]

    if numberOfCuts == 1:
        indexarray = list(np.where(np.abs(coordz-cuts_to_extract)<fields['res'])[0])
    else:
        indexarray = [np.where(np.abs(coordz-cuts_to_extract[i])<fields['res'])[0][0] for i in range(numberOfCuts)]
    position0 = np.array([shape_0*shape_1*indexarray[i] for i in range(numberOfCuts)])
    indices   = [range(position0[i],position0[i]+shape_0*shape_1) for i in range(numberOfCuts)]
    
    for field_component in field_components:
        field_component_value = fields['datavalues'][field_component]
        fields['zValueplane'][field_component] = np.array([field_component_value[indices[i]] for i in range(numberOfCuts)]).reshape(numberOfCuts,shape_0,shape_1)

def mask_cut(fields,datatypes):
    for datatype in datatypes:
        fields['zValueZeroedplane'][datatype] = np.nan_to_num(fields['zValueplane'][datatype], nan=0)


def near_field_to_far_field_transformation(fields: dict, cut: int):
    Nx = fields['zValueZeroedplane']['Ex'][cut].shape[0]
    Ny = fields['zValueZeroedplane']['Ex'][cut].shape[1]
    num = 0
    for value in fields['zValueZeroedplane']['Ex'][cut]:

        emn_calculated = calculate_emn_from_dipole_field()

def calculate_emn_from_dipole_field(number_of_points: int, m: int, n: int, r: int, k: int):
    """Function used to obtain a single value of emn from our Dipole field"""
    theta_values = np.linspace(0, np.pi, num=number_of_points)
    phi_values = np.linspace(0, 2*np.pi, num=number_of_points)
    delta_theta = theta_values[1] - theta_values[0]
    delta_phi = phi_values[1] - phi_values[0]

    total_result = 0
    for theta in theta_values:
        for phi in phi_values:
            # total_result += [e_dipole_r_field(r, theta, k, eta, l, Io),e_dipole_theta_field(r,theta,k,eta,l,Io),0]*funciones.b_sin_function(-m,n,theta,phi)
            total_result += np.array([0,e_dipole_theta_field(r,theta,k,eta,l,Io),0])*funciones.b_sin_function(-m,n,theta,phi)
    return total_result*delta_theta*delta_phi


if __name__ == '__main__':
    plt.close('all')
    #try:

    fields = input_data_process(flow_config)
    
    read_data(fields)    
    
    extract_matrix_data(fields)

    extract_phi_value_cut(fields,fields['file_type'],cuts_to_extract = [15.e-3,30.e-3])
    
    # #Este método quita de en medio los NaN.
    mask_cut(fields,fields['file_type'])

    near_field_to_far_field_transformation(fields, cut=0)

    print("FIN PROGRAMA")
 
    #except Exception as exc:
        #print(exc)

