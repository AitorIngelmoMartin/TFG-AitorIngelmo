"""File used to perform the NF2FF transformation"""
import os, sys
import numpy as np
import numpy.ma as ma
from math import cos, sin
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


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
    global delta_x, delta_y
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

        fields['field_readed'][datatype] = np.array([complex(s.replace('i', 'j')) for i in range(len(rawdatalines)) \
            for k,s in zip(range(4),rawdatalines[i].split()) if k == 3])

def mask_values(fields,datatypes):
    for datatype in datatypes:
        field_component_with_no_nan = np.nan_to_num(fields['field_readed'][datatype], nan=0)
        fields['field_readed_masked'][datatype] = field_component_with_no_nan[np.nonzero(field_component_with_no_nan)]

def change_coordinate_system_to_spherical(fields: dict):
    """Function used to change the coordinate system from cartesians to sphericals"""

    make_interpolator(fields['field_readed_masked']['Ex'],1,5,0,np.pi,0,2*np.pi,100)

def make_interpolator(field_component, r_init: int, r_end: int, theta_init: int, theta_end: int, phi_init: int, phi_end: int, number_of_values: int):
    """Function used to make an interpolator to obtain spherical values"""

    # Creation of base coordinates
    # r = np.arange(r_init, r_end)
    r = 1
    theta = np.linspace(theta_init, theta_end, number_of_values)
    phi = np.linspace(phi_init, phi_end, number_of_values)

    spherical_field_component = []
    for field_componenet_value in field_component:
        spherical_field_component.append(field_componenet_value*(np.sin(theta)*np.cos(phi) + np.sin(theta)*np.sin(phi) + np.cos(theta)))
    print(spherical_field_component)
    # Creation of the interpolator
    # interpolador = RegularGridInterpolator((r, theta, phi), valores, bounds_error=False, fill_value=None)

if __name__ == '__main__':
    plt.close('all')
    #try:

    fields = input_data_process(flow_config)
    
    read_data(fields)    
    
    extract_matrix_data(fields)
    

    # Este método quita de en medio los NaN.
    mask_values(fields,fields['file_type'])

    change_coordinate_system_to_spherical(fields)
    # near_field_to_far_field_transformation(fields)
    
    print("FIN PROGRAMA")
 
    #except Exception as exc:
        #print(exc)


def near_field_to_far_field_transformation(fields: dict, cut: int):
    Nx = fields['field_readed_masked']['Ex'][cut].shape[0]
    Ny = fields['field_readed_masked']['Ex'][cut].shape[1]
    num = 0
    for value in fields['field_readed_masked']['Ex'][cut]:

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
